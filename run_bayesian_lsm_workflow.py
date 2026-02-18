#!/usr/bin/env python
"""
End-to-end Bayesian Landslide Susceptibility Mapping workflow for small-N data.

Pipeline:
1) Preprocess + multicollinearity diagnostics (VIF, SPI/TRI/TWI check)
2) Bayesian Neural Network (PyTorch + PyTorch Lightning)
3) Spatial K-Fold CV (groups from coordinate-space clustering)
4) Uncertainty quantification (predictive mean + predictive std)
5) SHAP explainability
6) Susceptibility + uncertainty grid map generation
"""

from __future__ import annotations

try:
    import rasterio
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TARGET_COL = "class"
PREDICTOR_COLS = [
    "longitude",
    "latitude",
    "aspect",
    "curvature",
    "elevation",
    "lulc",
    "ndvi",
    "rainfall",
    "slope",
    "spi",
    "tri",
    "twi",
]
CATEGORICAL_COL = "lulc"
COORD_COLS = ["longitude", "latitude"]
HYDRO_COLS = ["spi", "tri", "twi"]
CONTINUOUS_COLS = [c for c in PREDICTOR_COLS if c != CATEGORICAL_COL]
INTERP_CONTINUOUS_COLS = [c for c in CONTINUOUS_COLS if c not in COORD_COLS]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def compute_vif_table(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    x = df[columns].astype(float).copy()
    for col in x.columns:
        if x[col].std() == 0:
            x[col] = x[col] + 1e-8
    design = x.values
    vifs = [variance_inflation_factor(design, i) for i in range(design.shape[1])]
    return pd.DataFrame({"feature": columns, "vif": vifs}).sort_values(
        "vif", ascending=False
    )


def decide_hydro_strategy(
    df: pd.DataFrame,
    vif_threshold: float = 10.0,
    corr_threshold: float = 0.8,
) -> Tuple[str, pd.DataFrame]:
    hydro_df = df[HYDRO_COLS].astype(float).copy()
    corr = hydro_df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    max_corr = np.nanmax(np.abs(upper.values))
    vif_hydro = compute_vif_table(df, HYDRO_COLS)
    max_vif = vif_hydro["vif"].max()
    strategy = (
        "pca_1_component"
        if (max_corr >= corr_threshold or max_vif >= vif_threshold)
        else "keep_all"
    )
    report = vif_hydro.copy()
    report["max_abs_pairwise_corr"] = max_corr
    report["selected_strategy"] = strategy
    return strategy, report


class LSMPreprocessor:
    def __init__(
        self, hydro_strategy: str = "keep_all", lulc_categories: List[int] | None = None
    ):
        self.hydro_strategy = hydro_strategy
        if lulc_categories is None:
            lulc_encoder_categories = "auto"
        else:
            lulc_encoder_categories = [list(lulc_categories)]
        self.transformer = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), CONTINUOUS_COLS),
                (
                    "onehot_lulc",
                    OneHotEncoder(
                        categories=lulc_encoder_categories,
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="first",
                    ),
                    [CATEGORICAL_COL],
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.feature_names_: List[str] = []
        self.hydro_indices_: List[int] = []
        self.non_hydro_indices_: List[int] = []
        self.hydro_pca_: np.ndarray | None = None
        self.hydro_mean_: np.ndarray | None = None

    def fit(self, x_df: pd.DataFrame) -> "LSMPreprocessor":
        self.transformer.fit(x_df)
        base_feature_names = list(self.transformer.get_feature_names_out())
        self.hydro_indices_ = [base_feature_names.index(c) for c in HYDRO_COLS]
        self.non_hydro_indices_ = [
            i for i in range(len(base_feature_names)) if i not in self.hydro_indices_
        ]

        x_base = self.transformer.transform(x_df)
        if self.hydro_strategy == "pca_1_component":
            hydro_block = x_base[:, self.hydro_indices_]
            self.hydro_mean_ = hydro_block.mean(axis=0, keepdims=True)
            centered = hydro_block - self.hydro_mean_
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            self.hydro_pca_ = vt[0:1, :]
            self.feature_names_ = [
                base_feature_names[i] for i in self.non_hydro_indices_
            ] + ["hydro_pc1"]
        else:
            self.feature_names_ = base_feature_names
        return self

    def transform(self, x_df: pd.DataFrame) -> np.ndarray:
        x_base = self.transformer.transform(x_df)
        if self.hydro_strategy == "pca_1_component":
            hydro_block = x_base[:, self.hydro_indices_]
            centered = hydro_block - self.hydro_mean_
            hydro_pc1 = centered @ self.hydro_pca_.T
            x_non_hydro = x_base[:, self.non_hydro_indices_]
            return np.concatenate([x_non_hydro, hydro_pc1], axis=1)
        return x_base


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.prior_sigma = prior_sigma
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(0.0, 0.05)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features).fill_(-3.0)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0.0, 0.05))
        self.bias_rho = nn.Parameter(torch.empty(out_features).fill_(-3.0))

    @staticmethod
    def _softplus(x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(x))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            w_sigma = self._softplus(self.weight_rho)
            b_sigma = self._softplus(self.bias_rho)
            w = self.weight_mu + w_sigma * torch.randn_like(self.weight_mu)
            b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

    def kl_divergence(self) -> torch.Tensor:
        prior_var = self.prior_sigma**2
        w_sigma = self._softplus(self.weight_rho)
        b_sigma = self._softplus(self.bias_rho)
        prior_sigma_t = torch.tensor(self.prior_sigma, device=self.weight_mu.device)
        w_kl = (
            torch.log(prior_sigma_t / w_sigma)
            + (w_sigma.pow(2) + self.weight_mu.pow(2)) / (2.0 * prior_var)
            - 0.5
        ).sum()
        b_kl = (
            torch.log(prior_sigma_t / b_sigma)
            + (b_sigma.pow(2) + self.bias_mu.pow(2)) / (2.0 * prior_var)
            - 0.5
        ).sum()
        return w_kl + b_kl


class BayesianBNN(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        n_train: int,
        prior_sigma: float = 1.0,
        hidden_dim_1: int = 32,
        hidden_dim_2: int = 16,
        dropout: float = 0.20,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        kl_beta: float = 1.0,
        mc_train_samples: int = 3,
        mc_val_samples: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_train = n_train
        self.mc_train_samples = mc_train_samples
        self.mc_val_samples = mc_val_samples
        self.b1 = BayesianLinear(input_dim, hidden_dim_1, prior_sigma=prior_sigma)
        self.b2 = BayesianLinear(hidden_dim_1, hidden_dim_2, prior_sigma=prior_sigma)
        self.b3 = BayesianLinear(hidden_dim_2, 1, prior_sigma=prior_sigma)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        x = F.relu(self.b1(x, sample=sample))
        x = self.dropout(x)
        x = F.relu(self.b2(x, sample=sample))
        x = self.dropout(x)
        return self.b3(x, sample=sample)

    def kl_divergence(self) -> torch.Tensor:
        return (
            self.b1.kl_divergence() + self.b2.kl_divergence() + self.b3.kl_divergence()
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)
        nll = 0.0
        for _ in range(self.mc_train_samples):
            logits = self.forward(x, sample=True)
            nll = nll + F.binary_cross_entropy_with_logits(logits, y)
        nll = nll / self.mc_train_samples
        kl = self.kl_divergence() / max(1, self.n_train)
        loss = nll + self.hparams.kl_beta * kl
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_nll", nll, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_kl", kl, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().view(-1, 1)
        probs_mc = []
        for _ in range(self.mc_val_samples):
            logits = self.forward(x, sample=True)
            probs_mc.append(torch.sigmoid(logits))
        mean_probs = torch.stack(probs_mc, dim=0).mean(dim=0)
        val_loss = F.binary_cross_entropy(mean_probs, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


@dataclass
class FoldResult:
    fold: int
    roc_auc: float
    pr_auc: float
    brier: float
    train_size: int
    test_size: int


def make_loader(
    x: np.ndarray, y: np.ndarray | None, batch_size: int, shuffle: bool
) -> DataLoader:
    x_t = torch.tensor(x, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(x_t)
    else:
        y_t = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(x_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def predict_prob_std(
    model: BayesianBNN,
    x: np.ndarray,
    mc_samples: int = 200,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = make_loader(x, None, batch_size=batch_size, shuffle=False)
    mean_list: List[np.ndarray] = []
    std_list: List[np.ndarray] = []
    device = model.device
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            probs_mc = []
            for _ in range(mc_samples):
                logits = model.forward(xb, sample=True)
                probs_mc.append(torch.sigmoid(logits).squeeze(-1))
            probs_mc = torch.stack(probs_mc, dim=0)
            mean_list.append(probs_mc.mean(dim=0).cpu().numpy())
            std_list.append(probs_mc.std(dim=0, unbiased=False).cpu().numpy())
    return np.concatenate(mean_list), np.concatenate(std_list)


def predict_prob_posterior_mean(
    model: BayesianBNN, x: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    model.eval()
    loader = make_loader(x, None, batch_size=batch_size, shuffle=False)
    outputs: List[np.ndarray] = []
    device = model.device
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            probs = (
                torch.sigmoid(model.forward(xb, sample=False).squeeze(-1)).cpu().numpy()
            )
            outputs.append(probs)
    return np.concatenate(outputs)


def train_one_bnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int,
    batch_size: int,
    prior_sigma: float,
    kl_beta: float,
    patience: int,
    lr: float,
) -> BayesianBNN:
    model = BayesianBNN(
        input_dim=x_train.shape[1],
        n_train=len(y_train),
        prior_sigma=prior_sigma,
        kl_beta=kl_beta,
        lr=lr,
    )
    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=patience, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        deterministic=True,
        callbacks=[early_stopping],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model


def run_spatial_cv(
    data: pd.DataFrame,
    y: np.ndarray,
    hydro_strategy: str,
    lulc_categories: List[int],
    n_splits: int,
    seed: int,
    max_epochs: int,
    batch_size: int,
    prior_sigma: float,
    kl_beta: float,
    patience: int,
    lr: float,
    mc_eval_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coords = data[COORD_COLS].values
    n_groups = max(n_splits * 4, n_splits)
    n_groups = min(n_groups, len(data))
    groups = KMeans(n_clusters=n_groups, random_state=seed, n_init=20).fit_predict(
        coords
    )
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results: List[FoldResult] = []
    oof_records: List[Dict] = []

    for fold, (train_idx, test_idx) in enumerate(
        sgkf.split(data, y, groups=groups), start=1
    ):
        train_df = data.iloc[train_idx].reset_index(drop=True)
        test_df = data.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]

        tr_idx, val_idx = train_test_split(
            np.arange(len(train_df)),
            test_size=0.2,
            random_state=seed + fold,
            stratify=y_train,
        )
        sub_train_df = train_df.iloc[tr_idx]
        sub_val_df = train_df.iloc[val_idx]
        y_sub_train = y_train[tr_idx]
        y_sub_val = y_train[val_idx]

        preprocessor = LSMPreprocessor(
            hydro_strategy=hydro_strategy,
            lulc_categories=lulc_categories,
        ).fit(sub_train_df[PREDICTOR_COLS])
        x_sub_train = preprocessor.transform(sub_train_df[PREDICTOR_COLS])
        x_sub_val = preprocessor.transform(sub_val_df[PREDICTOR_COLS])
        x_test = preprocessor.transform(test_df[PREDICTOR_COLS])

        model = train_one_bnn(
            x_train=x_sub_train,
            y_train=y_sub_train,
            x_val=x_sub_val,
            y_val=y_sub_val,
            max_epochs=max_epochs,
            batch_size=batch_size,
            prior_sigma=prior_sigma,
            kl_beta=kl_beta,
            patience=patience,
            lr=lr,
        )

        test_prob, test_std = predict_prob_std(
            model, x_test, mc_samples=mc_eval_samples, batch_size=512
        )
        if np.unique(y_test).size > 1:
            fold_roc = roc_auc_score(y_test, test_prob)
        else:
            fold_roc = np.nan
        if np.any(y_test == 1):
            fold_pr = average_precision_score(y_test, test_prob)
        else:
            fold_pr = np.nan
        fold_results.append(
            FoldResult(
                fold=fold,
                roc_auc=fold_roc,
                pr_auc=fold_pr,
                brier=brier_score_loss(y_test, test_prob),
                train_size=len(train_idx),
                test_size=len(test_idx),
            )
        )
        test_orig_idx = data.index[test_idx]
        for i, global_idx in enumerate(test_orig_idx):
            oof_records.append(
                {
                    "index": int(global_idx),
                    "fold": fold,
                    "y_true": int(y_test[i]),
                    "prob_mean": float(test_prob[i]),
                    "prob_std": float(test_std[i]),
                }
            )

    return (
        pd.DataFrame([fr.__dict__ for fr in fold_results]).sort_values("fold"),
        pd.DataFrame(oof_records).sort_values("index"),
    )


def train_final_model(
    data: pd.DataFrame,
    y: np.ndarray,
    hydro_strategy: str,
    lulc_categories: List[int],
    seed: int,
    max_epochs: int,
    batch_size: int,
    prior_sigma: float,
    kl_beta: float,
    patience: int,
    lr: float,
) -> Tuple[LSMPreprocessor, BayesianBNN, np.ndarray]:
    tr_idx, val_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.15,
        random_state=seed,
        stratify=y,
    )
    train_df = data.iloc[tr_idx].reset_index(drop=True)
    val_df = data.iloc[val_idx].reset_index(drop=True)
    y_train = y[tr_idx]
    y_val = y[val_idx]

    preprocessor = LSMPreprocessor(
        hydro_strategy=hydro_strategy,
        lulc_categories=lulc_categories,
    ).fit(train_df[PREDICTOR_COLS])
    x_train = preprocessor.transform(train_df[PREDICTOR_COLS])
    x_val = preprocessor.transform(val_df[PREDICTOR_COLS])
    model = train_one_bnn(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        max_epochs=max_epochs,
        batch_size=batch_size,
        prior_sigma=prior_sigma,
        kl_beta=kl_beta,
        patience=patience,
        lr=lr,
    )
    x_full = preprocessor.transform(data[PREDICTOR_COLS])
    return preprocessor, model, x_full


def build_grid_covariates(data: pd.DataFrame, grid_resolution: int) -> pd.DataFrame:
    lon_min, lon_max = data["longitude"].min(), data["longitude"].max()
    lat_min, lat_max = data["latitude"].min(), data["latitude"].max()
    lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
    lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    grid_df = pd.DataFrame(
        {"longitude": lon_mesh.ravel(), "latitude": lat_mesh.ravel()}
    )

    coords = data[COORD_COLS].values
    query_coords = grid_df[COORD_COLS].values
    for col in INTERP_CONTINUOUS_COLS:
        knr = KNeighborsRegressor(n_neighbors=12, weights="distance")
        knr.fit(coords, data[col].values)
        grid_df[col] = knr.predict(query_coords)
    knc = KNeighborsClassifier(n_neighbors=9, weights="distance")
    knc.fit(coords, data[CATEGORICAL_COL].values)
    grid_df[CATEGORICAL_COL] = knc.predict(query_coords).astype(int)
    return grid_df[PREDICTOR_COLS]


def save_map_png(
    grid_df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    pivot = grid_df.pivot(
        index="latitude", columns="longitude", values=value_col
    ).sort_index()
    extent = [
        grid_df["longitude"].min(),
        grid_df["longitude"].max(),
        grid_df["latitude"].min(),
        grid_df["latitude"].max(),
    ]
    plt.figure(figsize=(9, 7))
    plt.imshow(
        pivot.values,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=value_col)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_map_geotiff(
    grid_df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    nodata: float = -9999.0,
) -> None:
    """Save a gridded value column as a single-band GeoTIFF (EPSG:4326).

    The grid_df must contain 'longitude' and 'latitude' columns produced by
    ``build_grid_covariates`` (i.e. a regular lat/lon mesh).
    """
    if not HAS_RASTERIO:
        print(
            f"[WARN] rasterio not installed – skipping GeoTIFF export for {output_path.name}"
        )
        return

    pivot = grid_df.pivot(
        index="latitude", columns="longitude", values=value_col
    ).sort_index(
        ascending=False  # north-up: largest latitude = first row
    )
    data = pivot.values.astype(np.float32)
    nrows, ncols = data.shape

    lon_vals = np.sort(grid_df["longitude"].unique())
    lat_vals = np.sort(grid_df["latitude"].unique())
    # half-pixel padding so pixel centres align with the original coordinates
    half_dx = (lon_vals[1] - lon_vals[0]) / 2.0
    half_dy = (lat_vals[1] - lat_vals[0]) / 2.0
    west = float(lon_vals.min() - half_dx)
    east = float(lon_vals.max() + half_dx)
    south = float(lat_vals.min() - half_dy)
    north = float(lat_vals.max() + half_dy)

    transform = from_bounds(west, south, east, north, ncols, nrows)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(band_1_description=value_col)

    print(f"GeoTIFF saved: {output_path}")


def run_shap_analysis(
    model: BayesianBNN,
    x_all: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x_all.shape[0]
    bg_idx = rng.choice(np.arange(n), size=min(80, n), replace=False)
    ex_idx = rng.choice(np.arange(n), size=min(180, n), replace=False)
    x_bg = x_all[bg_idx]
    x_ex = x_all[ex_idx]

    def pred_fn(x_np: np.ndarray) -> np.ndarray:
        return predict_prob_posterior_mean(model, x_np, batch_size=1024)

    explainer = shap.Explainer(
        pred_fn,
        x_bg,
        feature_names=feature_names,
        algorithm="permutation",
    )
    shap_values = explainer(x_ex, max_evals=2 * x_ex.shape[1] + 1)
    shap_array = shap_values.values
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_array).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(out_dir / "shap_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_array, x_ex, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_array, x_ex, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap_df = pd.DataFrame(shap_array, columns=feature_names)
    shap_df["row_id"] = np.arange(shap_df.shape[0])
    shap_df.to_csv(out_dir / "shap_values_sample.csv", index=False)
    return imp_df, shap_df, x_ex


def feature_effect_direction(
    feature_name: str,
    feature_names: List[str],
    x_sample: np.ndarray,
    shap_df: pd.DataFrame,
) -> Tuple[float, float]:
    if feature_name not in feature_names:
        return float("nan"), float("nan")
    idx = feature_names.index(feature_name)
    vals = x_sample[:, idx]
    shap_col = shap_df[feature_name].values
    q25 = np.quantile(vals, 0.25)
    q75 = np.quantile(vals, 0.75)
    return float(shap_col[vals <= q25].mean()), float(shap_col[vals >= q75].mean())


def main(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    base_dir = Path(args.data).resolve().parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.data)
    df = df_raw[PREDICTOR_COLS + [TARGET_COL]].dropna().reset_index(drop=True)
    y = df[TARGET_COL].astype(int).values
    lulc_categories = sorted(df[CATEGORICAL_COL].unique().tolist())

    vif_table = compute_vif_table(df, CONTINUOUS_COLS)
    vif_table.to_csv(out_dir / "vif_table.csv", index=False)
    hydro_strategy, hydro_report = decide_hydro_strategy(df)
    hydro_report.to_csv(out_dir / "hydro_multicollinearity_report.csv", index=False)

    fold_metrics, oof = run_spatial_cv(
        data=df[PREDICTOR_COLS],
        y=y,
        hydro_strategy=hydro_strategy,
        lulc_categories=lulc_categories,
        n_splits=args.n_splits,
        seed=args.seed,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        prior_sigma=args.prior_sigma,
        kl_beta=args.kl_beta,
        patience=args.patience,
        lr=args.lr,
        mc_eval_samples=args.mc_eval_samples,
    )
    fold_metrics.to_csv(out_dir / "spatial_cv_fold_metrics.csv", index=False)
    oof.to_csv(out_dir / "spatial_cv_oof_predictions.csv", index=False)

    y_oof = oof["y_true"].values
    p_oof = oof["prob_mean"].values
    overall_metrics = {
        "roc_auc": float(roc_auc_score(y_oof, p_oof)),
        "pr_auc": float(average_precision_score(y_oof, p_oof)),
        "brier": float(brier_score_loss(y_oof, p_oof)),
    }
    pd.DataFrame([overall_metrics]).to_csv(
        out_dir / "spatial_cv_overall_metrics.csv", index=False
    )

    fpr, tpr, _ = roc_curve(y_oof, p_oof)
    precision, recall, _ = precision_recall_curve(y_oof, p_oof)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {overall_metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Spatial CV ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve_spatial_cv.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {overall_metrics['pr_auc']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Spatial CV Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curve_spatial_cv.png", dpi=300)
    plt.close()

    frac_pos, mean_pred = calibration_curve(
        y_oof, p_oof, n_bins=10, strategy="quantile"
    )
    plt.figure(figsize=(7, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="BNN")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration (Spatial CV OOF)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_curve_spatial_cv.png", dpi=300)
    plt.close()

    preprocessor, final_model, x_full = train_final_model(
        data=df[PREDICTOR_COLS],
        y=y,
        hydro_strategy=hydro_strategy,
        lulc_categories=lulc_categories,
        seed=args.seed,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        prior_sigma=args.prior_sigma,
        kl_beta=args.kl_beta,
        patience=args.patience,
        lr=args.lr,
    )
    full_prob, full_std = predict_prob_std(
        final_model, x_full, mc_samples=args.mc_eval_samples, batch_size=512
    )
    train_pred_df = df[PREDICTOR_COLS + [TARGET_COL]].copy()
    train_pred_df["prob_mean"] = full_prob
    train_pred_df["prob_std"] = full_std
    train_pred_df.to_csv(
        out_dir / "training_point_predictions_with_uncertainty.csv", index=False
    )

    shap_imp, shap_df, x_sample = run_shap_analysis(
        model=final_model,
        x_all=x_full,
        feature_names=preprocessor.feature_names_,
        out_dir=out_dir,
        seed=args.seed,
    )

    slope_low, slope_high = feature_effect_direction(
        "slope", preprocessor.feature_names_, x_sample, shap_df
    )
    rain_low, rain_high = feature_effect_direction(
        "rainfall", preprocessor.feature_names_, x_sample, shap_df
    )

    grid_cov = build_grid_covariates(
        df[PREDICTOR_COLS], grid_resolution=args.grid_resolution
    )
    x_grid = preprocessor.transform(grid_cov[PREDICTOR_COLS])
    grid_prob, grid_std = predict_prob_std(
        final_model, x_grid, mc_samples=args.mc_eval_samples, batch_size=1024
    )
    grid_out = grid_cov.copy()
    grid_out["susceptibility_prob"] = grid_prob
    grid_out["prediction_std"] = grid_std
    grid_out.to_csv(out_dir / "susceptibility_grid_lat_lon.csv", index=False)

    save_map_png(
        grid_out,
        value_col="susceptibility_prob",
        output_path=out_dir / "susceptibility_map.png",
        title="Bayesian BNN Landslide Susceptibility",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
    )
    save_map_png(
        grid_out,
        value_col="prediction_std",
        output_path=out_dir / "uncertainty_map.png",
        title="Predictive Uncertainty (Std. Dev.)",
        cmap="viridis",
        vmin=0.0,
        vmax=float(np.quantile(grid_out["prediction_std"], 0.99)),
    )

    # ---- GeoTIFF raster outputs (for ArcGIS Pro / QGIS) ----
    save_map_geotiff(
        grid_out,
        value_col="susceptibility_prob",
        output_path=out_dir / "susceptibility_map.tif",
    )
    save_map_geotiff(
        grid_out,
        value_col="prediction_std",
        output_path=out_dir / "uncertainty_map.tif",
    )

    cv_mean = fold_metrics[["roc_auc", "pr_auc", "brier"]].mean()
    cv_std = fold_metrics[["roc_auc", "pr_auc", "brier"]].std()
    top_features = shap_imp.head(8)

    lines = [
        "# Results and Discussion",
        "",
        "## 1. Multicollinearity and Preprocessing",
        "- VIF diagnostics were run for continuous predictors (`outputs/vif_table.csv`).",
        f"- SPI/TRI/TWI treatment: **{hydro_strategy}** (`outputs/hydro_multicollinearity_report.csv`).",
        "- `lulc` was one-hot encoded and all continuous predictors were standardized within each training fold.",
        "",
        "## 2. Spatial Cross-Validation Performance",
        "- Spatial 5-fold CV used coordinate-space clustering (`longitude`, `latitude`) and `StratifiedGroupKFold`.",
        f"- ROC-AUC: **{cv_mean['roc_auc']:.3f} +/- {cv_std['roc_auc']:.3f}**",
        f"- PR-AUC: **{cv_mean['pr_auc']:.3f} +/- {cv_std['pr_auc']:.3f}**",
        f"- Brier score: **{cv_mean['brier']:.3f} +/- {cv_std['brier']:.3f}**",
        "- Curves: `outputs/roc_curve_spatial_cv.png`, `outputs/precision_recall_curve_spatial_cv.png`, `outputs/calibration_curve_spatial_cv.png`.",
        "",
        "## 3. Bayesian Regularization and Uncertainty",
        "- Weakly informative zero-mean Gaussian priors were used for all network weights.",
        "- ELBO optimization (likelihood + KL penalty) constrains model complexity for small N (N=324).",
        "- Predictive uncertainty was quantified by Monte Carlo posterior predictive standard deviation.",
        "",
        "## 4. SHAP Explainability",
        "- Global feature ranking: `outputs/shap_feature_importance.csv`.",
        "- SHAP plots: `outputs/shap_summary_bar.png` and `outputs/shap_summary_beeswarm.png`.",
        "- Top SHAP features:",
    ]
    for _, row in top_features.iterrows():
        lines.append(f"  - {row['feature']}: mean|SHAP|={row['mean_abs_shap']:.5f}")
    lines.extend(
        [
            f"- Slope effect (Q1 -> Q4 SHAP mean): **{slope_low:.4f} -> {slope_high:.4f}**.",
            f"- Rainfall effect (Q1 -> Q4 SHAP mean): **{rain_low:.4f} -> {rain_high:.4f}**.",
            "",
            "## 5. Susceptibility and Uncertainty Mapping",
            "- Grid prediction table: `outputs/susceptibility_grid_lat_lon.csv`.",
            "- Susceptibility map: `outputs/susceptibility_map.png`.",
            "- Uncertainty map: `outputs/uncertainty_map.png`.",
            "- **GeoTIFF rasters** (EPSG:4326, Float32): `outputs/susceptibility_map.tif`, `outputs/uncertainty_map.tif` – ready for ArcGIS Pro.",
            "- High-uncertainty zones indicate where additional sampling would most improve model reliability.",
            "",
            "## 6. Reproducibility",
            f"- `{json.dumps({'seed': args.seed, 'n_samples': int(len(df)), 'n_splits': args.n_splits, 'max_epochs': args.max_epochs, 'batch_size': args.batch_size, 'prior_sigma': args.prior_sigma, 'kl_beta': args.kl_beta, 'mc_eval_samples': args.mc_eval_samples, 'grid_resolution': args.grid_resolution})}`",
        ]
    )
    (base_dir / "LSM_RESEARCH_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("Workflow complete.")
    print(f"Output directory: {out_dir}")
    print(f"Hydro strategy: {hydro_strategy}")
    print(f"Overall OOF metrics: {overall_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian LSM workflow with spatial CV and SHAP."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="lsm_training_dataset.csv",
        help="Path to training CSV.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--prior_sigma", type=float, default=1.0, help="Std-dev of Gaussian prior."
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=1.0,
        help="KL scaling factor in ELBO objective.",
    )
    parser.add_argument("--mc_eval_samples", type=int, default=120)
    parser.add_argument("--grid_resolution", type=int, default=200)
    main(parser.parse_args())
