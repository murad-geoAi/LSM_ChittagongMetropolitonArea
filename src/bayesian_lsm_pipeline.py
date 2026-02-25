import glob
import json
import os
import random
import warnings
from dataclasses import dataclass

import geopandas as gpd
import jenkspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import scikit_posthocs as sp
import seaborn as sns
import shap
from rasterio.mask import mask
from rasterio.transform import from_origin
from scipy.interpolate import PchipInterpolator, griddata
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import friedmanchisquare, wilcoxon
from shapely.geometry import mapping
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pytorch_lightning.callbacks import EarlyStopping
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing deep-learning dependency. Install requirements.txt before running "
        "the knowledge-guided pipeline (torch, pytorch-lightning, fsspec)."
    ) from exc

warnings.filterwarnings("ignore")

# Paths and reproducibility
RESULTS_DIR = "results"
DATA_OUTPUT_DIR = os.path.join(RESULTS_DIR, "data")
FIG_OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_FILE_CANDIDATES = ("Landslide_Data.csv", "results/data/LSM_Data.csv")
SHAPEFILE = None
RANDOM_SEED = 42

# Dataset assumptions
TARGET_COL = "Landslide"
LON_COL = "Longitude"
LAT_COL = "Latitude"
LEAKAGE_COLS = {"Predicted_Prob", "Predicted_Class", "Uncertainty"}

# Modeling configuration
TEST_SIZE = 0.20
VIF_THRESHOLD = 5.0
CV_SPLITS = 5
CV_REPEATS = 3
NUM_CLASSES = 5
HOLDOUT_BOOTSTRAP_ITER = 1200
BAYESIAN_MODEL_NAMES = ("BNN", "DeepEnsemble", "BLR", "GPC")
BENCHMARK_MODEL_NAMES = ("LR",)
INCLUDE_BENCHMARK_MODELS = True
DEEP_MODEL_MAX_EPOCHS = 90
DEEP_MODEL_PATIENCE = 12
DEEP_MODEL_BATCH_SIZE = 64
BNN_MC_SAMPLES = 80
DEEP_ENSEMBLE_MEMBERS = 3
KNOWLEDGE_GUIDED_LAMBDA_DEEP = 0.15
KNOWLEDGE_GUIDED_LAMBDA_LINEAR = 0.30
KNOWLEDGE_GUIDED_PRIORS = {
    "Elevation": +1.0,
    "Slope": +1.0,
    "Precipitation": +1.0,
    "TRI": +1.0,
    "Drainage_Density": +1.0,
    "Distance_to_River": -1.0,
    "Distance_to_Road": -1.0,
    "NDVI": -1.0,
}


def stable_softplus_np(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def stable_sigmoid_np(x):
    x = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return expit(x)


def build_knowledge_constraints(feature_names):
    constraints = {}
    for i, name in enumerate(feature_names):
        if name in KNOWLEDGE_GUIDED_PRIORS:
            constraints[int(i)] = float(KNOWLEDGE_GUIDED_PRIORS[name])
    return constraints


@dataclass
class DatasetCharacteristics:
    n_samples: int
    n_features: int
    n_positive: int
    n_negative: int
    positive_ratio: float
    total_missing: int
    missing_rate: float


class KnowledgeGuidedLogisticRegression(BaseEstimator, ClassifierMixin):
    """Deterministic logistic regression with soft sign constraints."""

    def __init__(
        self,
        l2_strength=1.0,
        class_weight="balanced",
        max_iter=500,
        tol=1e-6,
        knowledge_constraints=None,
        knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_LINEAR,
        random_state=RANDOM_SEED,
    ):
        self.l2_strength = l2_strength
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.knowledge_constraints = (
            {} if knowledge_constraints is None else dict(knowledge_constraints)
        )
        self.knowledge_lambda = knowledge_lambda
        self.random_state = random_state

    def _prepare_X(self, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr

    def _prepare_targets(self, y):
        y_arr = np.asarray(y, dtype=float).ravel()
        unique = np.unique(y_arr)
        if set(unique.tolist()) - {0.0, 1.0}:
            raise ValueError(
                "KnowledgeGuidedLogisticRegression supports binary labels {0,1} only."
            )
        return y_arr

    def _prepare_sample_weights(self, y_arr):
        if self.class_weight != "balanced":
            return np.ones_like(y_arr, dtype=float)
        n_pos = np.sum(y_arr == 1.0)
        n_neg = np.sum(y_arr == 0.0)
        if n_pos == 0 or n_neg == 0:
            return np.ones_like(y_arr, dtype=float)
        w_pos = len(y_arr) / (2.0 * n_pos)
        w_neg = len(y_arr) / (2.0 * n_neg)
        return np.where(y_arr == 1.0, w_pos, w_neg).astype(float)

    def _constraint_vector(self, n_params):
        signs = np.zeros(n_params, dtype=float)
        for idx, sign in self.knowledge_constraints.items():
            pos = int(idx) + 1  # +1 due intercept in augmented design.
            if 1 <= pos < n_params and sign != 0:
                signs[pos] = np.sign(float(sign))
        return signs

    def fit(self, X, y):
        X_arr = self._prepare_X(X)
        y_arr = self._prepare_targets(y)
        n_samples = X_arr.shape[0]
        X_aug = np.hstack([np.ones((n_samples, 1)), X_arr])
        n_params = X_aug.shape[1]
        sample_w = self._prepare_sample_weights(y_arr)
        sign_vec = self._constraint_vector(n_params)
        sign_mask = sign_vec != 0.0
        n_constraints = max(int(np.sum(sign_mask)), 1)

        def nlp_and_grad(w):
            logits = np.clip(X_aug @ w, -35.0, 35.0)
            probs = stable_sigmoid_np(logits)
            eps = 1e-12
            nll = -np.sum(
                sample_w
                * (
                    y_arr * np.log(probs + eps)
                    + (1.0 - y_arr) * np.log(1.0 - probs + eps)
                )
            )
            grad = X_aug.T @ (sample_w * (probs - y_arr))

            if self.l2_strength > 0:
                nll += 0.5 * float(self.l2_strength) * np.sum(w[1:] ** 2)
                grad[1:] += float(self.l2_strength) * w[1:]

            if self.knowledge_lambda > 0 and np.any(sign_mask):
                z = -sign_vec[sign_mask] * w[sign_mask]
                sp = stable_softplus_np(z)
                sig = stable_sigmoid_np(z)
                nll += float(self.knowledge_lambda) * np.sum(sp**2) / n_constraints
                grad_kg = -2.0 * sign_vec[sign_mask] * sp * sig
                grad[sign_mask] += (
                    float(self.knowledge_lambda) * grad_kg / n_constraints
                )
            return nll, grad

        w0 = np.zeros(n_params, dtype=float)
        opt = minimize(
            fun=lambda w: nlp_and_grad(w)[0],
            x0=w0,
            jac=lambda w: nlp_and_grad(w)[1],
            method="L-BFGS-B",
            options={"maxiter": int(self.max_iter), "ftol": float(self.tol)},
        )
        self.optim_result_ = opt
        if not opt.success:
            print(f"Warning: LR optimizer did not fully converge: {opt.message}")
        self.coef_map_ = opt.x
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = X_arr.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        return self

    def decision_function(self, X):
        X_arr = self._prepare_X(X)
        X_aug = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
        return np.clip(X_aug @ self.coef_map_, -35.0, 35.0)

    def predict_proba(self, X):
        probs = stable_sigmoid_np(self.decision_function(X))
        probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary Bayesian logistic regression with Laplace posterior approximation."""

    def __init__(
        self,
        prior_precision=1.0,
        class_weight="balanced",
        max_iter=500,
        tol=1e-6,
        n_posterior_samples=400,
        knowledge_constraints=None,
        knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_LINEAR,
        random_state=RANDOM_SEED,
    ):
        self.prior_precision = prior_precision
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.n_posterior_samples = n_posterior_samples
        self.knowledge_constraints = (
            {} if knowledge_constraints is None else dict(knowledge_constraints)
        )
        self.knowledge_lambda = knowledge_lambda
        self.random_state = random_state

    def _prepare_X(self, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr

    def _prepare_targets(self, y):
        y_arr = np.asarray(y, dtype=float).ravel()
        unique = np.unique(y_arr)
        if set(unique.tolist()) - {0.0, 1.0}:
            raise ValueError("BayesianLogisticRegression supports binary labels {0,1}.")
        return y_arr

    def _prepare_sample_weights(self, y_arr):
        if self.class_weight != "balanced":
            return np.ones_like(y_arr, dtype=float)
        n_pos = np.sum(y_arr == 1.0)
        n_neg = np.sum(y_arr == 0.0)
        if n_pos == 0 or n_neg == 0:
            return np.ones_like(y_arr, dtype=float)
        w_pos = len(y_arr) / (2.0 * n_pos)
        w_neg = len(y_arr) / (2.0 * n_neg)
        return np.where(y_arr == 1.0, w_pos, w_neg).astype(float)

    def _constraint_vector(self, n_params):
        signs = np.zeros(n_params, dtype=float)
        for idx, sign in self.knowledge_constraints.items():
            pos = int(idx) + 1
            if 1 <= pos < n_params and sign != 0:
                signs[pos] = np.sign(float(sign))
        return signs

    def fit(self, X, y):
        X_arr = self._prepare_X(X)
        y_arr = self._prepare_targets(y)
        n_samples = X_arr.shape[0]
        X_aug = np.hstack([np.ones((n_samples, 1)), X_arr])
        n_params = X_aug.shape[1]
        sample_w = self._prepare_sample_weights(y_arr)

        prior_diag = np.full(n_params, float(self.prior_precision))
        prior_diag[0] = 0.0
        sign_vec = self._constraint_vector(n_params)
        sign_mask = sign_vec != 0.0
        n_constraints = max(int(np.sum(sign_mask)), 1)

        def nlp_and_grad(w):
            logits = np.clip(X_aug @ w, -35.0, 35.0)
            probs = stable_sigmoid_np(logits)
            eps = 1e-12
            nlp = -np.sum(
                sample_w
                * (
                    y_arr * np.log(probs + eps)
                    + (1.0 - y_arr) * np.log(1.0 - probs + eps)
                )
            )
            nlp += 0.5 * np.sum(prior_diag * (w**2))
            grad = X_aug.T @ (sample_w * (probs - y_arr)) + prior_diag * w

            if self.knowledge_lambda > 0 and np.any(sign_mask):
                z = -sign_vec[sign_mask] * w[sign_mask]
                sp = stable_softplus_np(z)
                sig = stable_sigmoid_np(z)
                nlp += float(self.knowledge_lambda) * np.sum(sp**2) / n_constraints
                grad_kg = -2.0 * sign_vec[sign_mask] * sp * sig
                grad[sign_mask] += (
                    float(self.knowledge_lambda) * grad_kg / n_constraints
                )
            return nlp, grad

        w0 = np.zeros(n_params, dtype=float)
        opt = minimize(
            fun=lambda w: nlp_and_grad(w)[0],
            x0=w0,
            jac=lambda w: nlp_and_grad(w)[1],
            method="L-BFGS-B",
            options={"maxiter": int(self.max_iter), "ftol": float(self.tol)},
        )
        self.optim_result_ = opt
        if not opt.success:
            print(f"Warning: BLR optimizer did not fully converge: {opt.message}")
        self.coef_map_ = opt.x

        logits = np.clip(X_aug @ self.coef_map_, -35.0, 35.0)
        probs = stable_sigmoid_np(logits)
        w_diag = sample_w * probs * (1.0 - probs)
        hessian = X_aug.T @ (X_aug * w_diag[:, None]) + np.diag(prior_diag)
        if self.knowledge_lambda > 0 and np.any(sign_mask):
            z = -sign_vec[sign_mask] * self.coef_map_[sign_mask]
            sp = stable_softplus_np(z)
            sig = stable_sigmoid_np(z)
            diag_kg = 2.0 * (sig**2 + sp * sig * (1.0 - sig))
            hessian[np.ix_(np.where(sign_mask)[0], np.where(sign_mask)[0])] += np.diag(
                float(self.knowledge_lambda) * diag_kg / n_constraints
            )
        hessian += np.eye(n_params) * 1e-6
        self.posterior_cov_ = np.linalg.pinv(hessian)

        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = X_arr.shape[1]
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
        return self

    def _predict_mean_var(self, X):
        X_arr = self._prepare_X(X)
        X_aug = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
        mean = X_aug @ self.coef_map_
        var = np.einsum("ij,jk,ik->i", X_aug, self.posterior_cov_, X_aug)
        var = np.clip(var, 0.0, None)
        return mean, var, X_aug

    def predict_proba(self, X):
        mean, var, _ = self._predict_mean_var(X)
        denom = np.sqrt(1.0 + (np.pi * var / 8.0))
        probs = stable_sigmoid_np(mean / np.maximum(denom, 1e-9))
        probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_uncertainty(self, X, n_samples=None):
        n_draws = int(n_samples if n_samples is not None else self.n_posterior_samples)
        n_draws = max(50, n_draws)
        mean, _, X_aug = self._predict_mean_var(X)

        cov = 0.5 * (self.posterior_cov_ + self.posterior_cov_.T)
        cov += np.eye(cov.shape[0]) * 1e-8

        rng = np.random.default_rng(self.random_state)
        draws = rng.multivariate_normal(self.coef_map_, cov, size=n_draws)
        logits_draws = X_aug @ draws.T
        probs_draws = stable_sigmoid_np(np.clip(logits_draws, -35.0, 35.0))

        mean_prob = np.mean(probs_draws, axis=1)
        std_prob = np.std(probs_draws, axis=1)
        laplace_mean = stable_sigmoid_np(
            mean / np.sqrt(1.0 + (np.pi * np.var(logits_draws, axis=1) / 8.0))
        )
        mean_prob = 0.5 * mean_prob + 0.5 * laplace_mean
        return np.clip(mean_prob, 1e-9, 1.0 - 1e-9), std_prob


class SmallMLP(nn.Module):
    """Compact MLP for small tabular datasets."""

    def __init__(self, input_dim, hidden_dims=(32, 16), dropout=0.25):
        super().__init__()
        h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class KnowledgeGuidedLightningModule(pl.LightningModule):
    """Binary classifier module with monotonicity-guided regularization."""

    def __init__(
        self,
        input_dim,
        hidden_dims=(32, 16),
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-3,
        pos_weight=1.0,
        knowledge_constraints=None,
        knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_DEEP,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["knowledge_constraints"])
        self.model = SmallMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.knowledge_lambda = float(knowledge_lambda)
        constraints = {} if knowledge_constraints is None else dict(knowledge_constraints)
        idx = sorted(int(k) for k in constraints.keys())
        signs = [float(np.sign(constraints[i])) for i in idx]
        self.register_buffer("kg_idx", torch.tensor(idx, dtype=torch.long))
        self.register_buffer("kg_sign", torch.tensor(signs, dtype=torch.float32))
        self.register_buffer(
            "pos_weight_tensor", torch.tensor(float(pos_weight), dtype=torch.float32)
        )

    def forward(self, x):
        return self.model(x)

    def _knowledge_penalty(self, x, logits):
        if self.kg_idx.numel() == 0 or self.knowledge_lambda <= 0:
            return torch.zeros((), device=self.device)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=x,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        constrained_grads = grads[:, self.kg_idx]
        signed_grad = constrained_grads * self.kg_sign.unsqueeze(0)
        violations = F.relu(-signed_grad)
        return torch.mean(violations**2)

    def _shared_step(self, batch, stage):
        x, y = batch
        x = x.float()
        y = y.float()
        use_kg = stage == "train" and self.kg_idx.numel() > 0 and self.knowledge_lambda > 0
        if use_kg:
            x = x.requires_grad_(True)
        logits = self.forward(x)
        bce = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=self.pos_weight_tensor
        )
        kg_penalty = self._knowledge_penalty(x, logits) if use_kg else torch.zeros(
            (), device=self.device
        )
        loss = bce + (self.knowledge_lambda * kg_penalty)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=(stage != "train"))
        if stage == "train":
            self.log("train_bce", bce, on_epoch=True, prog_bar=False)
            self.log("train_kg", kg_penalty, on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


def train_val_loaders(
    X_arr,
    y_arr,
    batch_size=DEEP_MODEL_BATCH_SIZE,
    val_fraction=0.15,
    random_state=RANDOM_SEED,
):
    n_samples = len(y_arr)
    batch_size = int(max(8, min(batch_size, n_samples)))
    X_tensor = torch.tensor(X_arr, dtype=torch.float32)
    y_tensor = torch.tensor(y_arr, dtype=torch.float32)

    unique, counts = np.unique(y_arr.astype(int), return_counts=True)
    min_count = int(np.min(counts)) if len(counts) > 0 else 0
    can_split = n_samples >= 40 and len(unique) == 2 and min_count >= 5

    if can_split:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=float(val_fraction), random_state=random_state
        )
        train_idx, val_idx = next(sss.split(X_arr, y_arr))
        train_ds = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    train_ds = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    return train_loader, None


def fit_lightning_model(
    X_arr,
    y_arr,
    knowledge_constraints,
    hidden_dims=(32, 16),
    dropout=0.25,
    lr=1e-3,
    weight_decay=1e-3,
    max_epochs=DEEP_MODEL_MAX_EPOCHS,
    patience=DEEP_MODEL_PATIENCE,
    batch_size=DEEP_MODEL_BATCH_SIZE,
    knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_DEEP,
    random_state=RANDOM_SEED,
):
    pl.seed_everything(int(random_state), workers=True)
    train_loader, val_loader = train_val_loaders(
        X_arr,
        y_arr,
        batch_size=batch_size,
        random_state=random_state,
    )

    n_pos = max(np.sum(y_arr == 1), 1)
    n_neg = max(np.sum(y_arr == 0), 1)
    pos_weight = float(n_neg / n_pos)

    module = KnowledgeGuidedLightningModule(
        input_dim=X_arr.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        pos_weight=pos_weight,
        knowledge_constraints=knowledge_constraints,
        knowledge_lambda=knowledge_lambda,
    )

    callbacks = []
    if val_loader is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=int(patience),
                min_delta=1e-4,
            )
        )

    trainer = pl.Trainer(
        max_epochs=int(max_epochs),
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        deterministic=True,
        gradient_clip_val=1.0,
    )
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    module.eval()
    return module


class BayesianNeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """MC-dropout Bayesian neural network in sklearn-compatible form."""

    def __init__(
        self,
        hidden_dims=(32, 16),
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-3,
        max_epochs=DEEP_MODEL_MAX_EPOCHS,
        patience=DEEP_MODEL_PATIENCE,
        batch_size=DEEP_MODEL_BATCH_SIZE,
        mc_samples=BNN_MC_SAMPLES,
        knowledge_constraints=None,
        knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_DEEP,
        random_state=RANDOM_SEED,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.mc_samples = mc_samples
        self.knowledge_constraints = (
            {} if knowledge_constraints is None else dict(knowledge_constraints)
        )
        self.knowledge_lambda = knowledge_lambda
        self.random_state = random_state

    def _prepare_X(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr

    def fit(self, X, y):
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        unique = np.unique(y_arr)
        if set(unique.tolist()) - {0.0, 1.0}:
            raise ValueError("BNN supports binary labels {0,1} only.")
        self.module_ = fit_lightning_model(
            X_arr=X_arr,
            y_arr=y_arr,
            knowledge_constraints=self.knowledge_constraints,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            knowledge_lambda=self.knowledge_lambda,
            random_state=self.random_state,
        )
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict_uncertainty(self, X, n_samples=None):
        X_arr = self._prepare_X(X)
        x_tensor = torch.tensor(X_arr, dtype=torch.float32)
        n_draws = int(n_samples if n_samples is not None else self.mc_samples)
        n_draws = max(30, n_draws)
        torch.manual_seed(int(self.random_state))

        draws = []
        with torch.no_grad():
            for _ in range(n_draws):
                self.module_.train()
                logits = self.module_(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                draws.append(probs)
        draw_arr = np.vstack(draws)
        mean_prob = np.clip(np.mean(draw_arr, axis=0), 1e-9, 1.0 - 1e-9)
        std_prob = np.std(draw_arr, axis=0)
        self.module_.eval()
        return mean_prob, std_prob

    def predict_proba(self, X):
        mean_prob, _ = self.predict_uncertainty(X, n_samples=self.mc_samples)
        return np.column_stack([1.0 - mean_prob, mean_prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class DeepEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Deep ensemble of compact neural networks."""

    def __init__(
        self,
        ensemble_size=DEEP_ENSEMBLE_MEMBERS,
        hidden_dims=(32, 16),
        dropout=0.15,
        lr=1e-3,
        weight_decay=1e-3,
        max_epochs=DEEP_MODEL_MAX_EPOCHS,
        patience=DEEP_MODEL_PATIENCE,
        batch_size=DEEP_MODEL_BATCH_SIZE,
        knowledge_constraints=None,
        knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_DEEP,
        random_state=RANDOM_SEED,
    ):
        self.ensemble_size = ensemble_size
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.knowledge_constraints = (
            {} if knowledge_constraints is None else dict(knowledge_constraints)
        )
        self.knowledge_lambda = knowledge_lambda
        self.random_state = random_state

    def _prepare_X(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr

    def fit(self, X, y):
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        unique = np.unique(y_arr)
        if set(unique.tolist()) - {0.0, 1.0}:
            raise ValueError("DeepEnsemble supports binary labels {0,1} only.")

        self.members_ = []
        for member_idx in range(int(max(1, self.ensemble_size))):
            seed = int(self.random_state + member_idx * 13)
            member = fit_lightning_model(
                X_arr=X_arr,
                y_arr=y_arr,
                knowledge_constraints=self.knowledge_constraints,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                lr=self.lr,
                weight_decay=self.weight_decay,
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                knowledge_lambda=self.knowledge_lambda,
                random_state=seed,
            )
            self.members_.append(member)

        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict_uncertainty(self, X, n_samples=None):
        _ = n_samples  # kept for interface compatibility
        X_arr = self._prepare_X(X)
        x_tensor = torch.tensor(X_arr, dtype=torch.float32)
        member_probs = []
        with torch.no_grad():
            for member in self.members_:
                member.eval()
                logits = member(x_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                member_probs.append(probs)
        prob_matrix = np.vstack(member_probs)
        mean_prob = np.clip(np.mean(prob_matrix, axis=0), 1e-9, 1.0 - 1e-9)
        std_prob = np.std(prob_matrix, axis=0)
        return mean_prob, std_prob

    def predict_proba(self, X):
        mean_prob, _ = self.predict_uncertainty(X)
        return np.column_stack([1.0 - mean_prob, mean_prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    random.seed(seed)


def ensure_output_dir():
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)


def data_output_path(*parts):
    return os.path.join(DATA_OUTPUT_DIR, *parts)


def figure_output_path(*parts):
    return os.path.join(FIG_OUTPUT_DIR, *parts)


def safe_remove_file(path):
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError as exc:
        print(f"Warning: Could not remove stale file '{path}': {exc}")


def resolve_data_file(filepath_candidates):
    if isinstance(filepath_candidates, (list, tuple)):
        names = list(filepath_candidates)
    else:
        names = [filepath_candidates]

    checked = []
    for filepath in names:
        candidates = [
            filepath,
            os.path.join(os.getcwd(), filepath),
            os.path.join(os.path.dirname(__file__), filepath),
            os.path.join(os.path.dirname(__file__), "..", filepath),
            os.path.join(os.path.dirname(__file__), "..", "..", filepath),
        ]
        for candidate in candidates:
            checked.append(candidate)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    raise FileNotFoundError(
        f"Could not find input dataset. Checked candidates: {checked}"
    )


def is_bayesian_model(model_name):
    return model_name in BAYESIAN_MODEL_NAMES


def get_model_role(model_name):
    if is_bayesian_model(model_name):
        return "Bayesian Deployment Candidate"
    return "Benchmark Performance Model"


def load_dataset(filepath):
    print("Loading empirical dataset...")
    df = pd.read_csv(filepath).drop_duplicates().reset_index(drop=True)

    required = {TARGET_COL, LON_COL, LAT_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns missing in dataset: {sorted(missing)}")

    features = [c for c in df.columns if c not in [TARGET_COL, LON_COL, LAT_COL]]
    leakage_like = [
        c
        for c in features
        if c in LEAKAGE_COLS
        or c.lower().startswith("predicted_")
        or c.lower() in {"uncertainty", "prediction", "pred_prob"}
    ]
    if leakage_like:
        print(f"Dropping leakage-prone columns from predictors: {leakage_like}")
        features = [c for c in features if c not in leakage_like]
    if not features:
        raise ValueError("No predictor columns found.")

    # Convert any non-numeric predictors to category codes.
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print(f"Converting non-numeric features to categorical codes: {non_numeric}")
        for col in non_numeric:
            df[col] = pd.Categorical(df[col]).codes

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df[features].copy()
    y = df[TARGET_COL].copy()
    coords = df[[LON_COL, LAT_COL]].copy()

    print(f"Dataset shape: {df.shape}")
    print("Class distribution:")
    print(y.value_counts().sort_index())
    return df, X, y, coords, features


def review_dataset_characteristics(df, X, y, features):
    print("\n--- Dataset Characteristics Review ---")
    missing_counts = X.isna().sum().sort_values(ascending=False)
    missing_df = pd.DataFrame(
        {
            "Feature": missing_counts.index,
            "Missing_Count": missing_counts.values.astype(int),
            "Missing_Rate": (missing_counts.values / len(df)),
        }
    )
    missing_df.to_csv(data_output_path("dataset_missingness.csv"), index=False)

    X_num = X.apply(pd.to_numeric, errors="coerce")
    effect_sizes = (
        ((X_num[y == 1].mean() - X_num[y == 0].mean()) / X_num.std(ddof=0).replace(0, np.nan))
        .rename("Std_Mean_Diff")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "Feature"})
    )
    effect_sizes["Abs_Std_Mean_Diff"] = effect_sizes["Std_Mean_Diff"].abs()
    effect_sizes = effect_sizes.sort_values("Abs_Std_Mean_Diff", ascending=False)
    effect_sizes.to_csv(data_output_path("dataset_effect_sizes.csv"), index=False)

    corr_target = (
        X_num.corrwith(y).rename("Corr_Target").to_frame().reset_index().rename(
            columns={"index": "Feature"}
        )
    )
    corr_target["Abs_Corr_Target"] = corr_target["Corr_Target"].abs()
    corr_target = corr_target.sort_values("Abs_Corr_Target", ascending=False)
    corr_target.to_csv(data_output_path("dataset_target_correlations.csv"), index=False)

    constraints = build_knowledge_constraints(features)
    prior_rows = []
    for idx, sign in constraints.items():
        feat = features[idx]
        corr = corr_target.loc[corr_target["Feature"] == feat, "Corr_Target"]
        corr_val = float(corr.iloc[0]) if len(corr) else np.nan
        prior_rows.append(
            {
                "Feature": feat,
                "Expected_Direction": int(np.sign(sign)),
                "Observed_Corr_Target": corr_val,
                "Direction_Agreement": bool(np.sign(corr_val) == np.sign(sign))
                if np.isfinite(corr_val)
                else False,
            }
        )
    prior_df = pd.DataFrame(prior_rows)
    prior_df.to_csv(data_output_path("knowledge_priors_review.csv"), index=False)

    stats = DatasetCharacteristics(
        n_samples=int(len(df)),
        n_features=int(len(features)),
        n_positive=int(np.sum(y == 1)),
        n_negative=int(np.sum(y == 0)),
        positive_ratio=float(np.mean(y == 1)),
        total_missing=int(X.isna().sum().sum()),
        missing_rate=float(X.isna().sum().sum() / max(1, X.shape[0] * X.shape[1])),
    )
    with open(data_output_path("dataset_characteristics.json"), "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, indent=2)

    print(
        f"Samples={stats.n_samples} | Features={stats.n_features} | "
        f"Positive ratio={stats.positive_ratio:.3f} | Missing cells={stats.total_missing}"
    )


def calculate_vif(X):
    vif_values = []
    for i in range(X.shape[1]):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.inf
        vif_values.append(vif)

    vif_df = pd.DataFrame({"Feature": X.columns, "VIF": vif_values})
    vif_df["TOL"] = np.where(
        np.isfinite(vif_df["VIF"]) & (vif_df["VIF"] > 0), 1.0 / vif_df["VIF"], 0.0
    )
    return vif_df


def plot_vif_tolerance_curve(vif_df, threshold=VIF_THRESHOLD):
    if vif_df.empty:
        return

    plot_df = (
        vif_df.copy()
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )
    x = np.arange(len(plot_df))

    fig, ax1 = plt.subplots(figsize=(12, 6.5))
    vif_color = "#d55e00"
    tol_color = "#0072b2"

    ax1.plot(
        x,
        plot_df["VIF"].to_numpy(dtype=float),
        color=vif_color,
        marker="o",
        linewidth=2.0,
        label="VIF",
    )
    ax1.axhline(
        threshold,
        color=vif_color,
        linestyle="--",
        linewidth=1.2,
        label=f"VIF threshold ({threshold:g})",
    )
    ax1.set_ylabel("Variance Inflation Factor (VIF)", color=vif_color)
    ax1.tick_params(axis="y", labelcolor=vif_color)
    ax1.set_ylim(bottom=0.0)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        plot_df["TOL"].to_numpy(dtype=float),
        color=tol_color,
        marker="s",
        linewidth=2.0,
        label="Tolerance (1/VIF)",
    )
    tol_threshold = 1.0 / threshold if threshold > 0 else 0.0
    ax2.axhline(
        tol_threshold,
        color=tol_color,
        linestyle="--",
        linewidth=1.2,
        label=f"Tolerance threshold ({tol_threshold:.2f})",
    )
    ax2.set_ylabel("Tolerance", color=tol_color)
    ax2.tick_params(axis="y", labelcolor=tol_color)
    ax2.set_ylim(0.0, 1.05)

    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df["Feature"], rotation=45, ha="right")
    ax1.set_xlabel("Conditioning Factors")
    ax1.set_title("Multicollinearity Profile: VIF and Tolerance by Predictor")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(figure_output_path("vif_tolerance_curve.png"), dpi=300)
    plt.close(fig)


def select_features_vif(X_train, threshold=VIF_THRESHOLD):
    print("Running training-only VIF feature selection...")
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )

    selected = X_imputed.columns.tolist()
    dropped = []

    # Drop near-constant features first.
    low_var_features = X_imputed.var()[X_imputed.var() <= 1e-12].index.tolist()
    if low_var_features:
        dropped.extend(low_var_features)
        selected = [f for f in selected if f not in low_var_features]

    if len(selected) == 0:
        raise ValueError("No features left after dropping near-constant predictors.")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed[selected]), columns=selected, index=X_train.index
    )

    while len(selected) > 1:
        vif_df = calculate_vif(X_scaled[selected])
        vif_values = vif_df["VIF"].to_numpy(dtype=float)
        vif_for_decision = np.where(np.isfinite(vif_values), vif_values, np.inf)
        max_idx = int(np.argmax(vif_for_decision))
        max_vif = float(vif_for_decision[max_idx])

        if max_vif <= threshold:
            break

        feature_to_drop = vif_df.iloc[max_idx]["Feature"]
        dropped.append(feature_to_drop)
        selected.remove(feature_to_drop)
        print(f"Dropping feature {feature_to_drop} with VIF {max_vif:.2f}")

    if len(selected) == 1:
        final_vif = pd.DataFrame({"Feature": selected, "VIF": [1.0], "TOL": [1.0]})
    else:
        final_vif = calculate_vif(X_scaled[selected])

    print(f"Selected features ({len(selected)}): {selected}")
    return selected, dropped, final_vif


def build_model_builders(
    feature_names,
    seed=RANDOM_SEED,
    include_benchmarks=INCLUDE_BENCHMARK_MODELS,
):
    constraints = build_knowledge_constraints(feature_names)
    builders = {
        "BNN": lambda: BayesianNeuralNetworkClassifier(
            hidden_dims=(32, 16),
            dropout=0.28,
            lr=9e-4,
            weight_decay=2e-3,
            max_epochs=DEEP_MODEL_MAX_EPOCHS,
            patience=DEEP_MODEL_PATIENCE,
            batch_size=DEEP_MODEL_BATCH_SIZE,
            mc_samples=BNN_MC_SAMPLES,
            knowledge_constraints=constraints,
            knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_DEEP,
            random_state=seed,
        ),
        "DeepEnsemble": lambda: DeepEnsembleClassifier(
            ensemble_size=DEEP_ENSEMBLE_MEMBERS,
            hidden_dims=(32, 16),
            dropout=0.18,
            lr=9e-4,
            weight_decay=2e-3,
            max_epochs=DEEP_MODEL_MAX_EPOCHS,
            patience=DEEP_MODEL_PATIENCE,
            batch_size=DEEP_MODEL_BATCH_SIZE,
            knowledge_constraints=constraints,
            knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_DEEP,
            random_state=seed,
        ),
        "BLR": lambda: BayesianLogisticRegression(
            prior_precision=1.5,
            class_weight="balanced",
            max_iter=500,
            tol=1e-6,
            n_posterior_samples=500,
            knowledge_constraints=constraints,
            knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_LINEAR,
            random_state=seed,
        ),
        "GPC": lambda: GaussianProcessClassifier(
            kernel=1.0 * Matern(length_scale=1.0, nu=1.5)
            + WhiteKernel(noise_level=1e-5),
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=3,
            random_state=seed,
        ),
        "LR": lambda: KnowledgeGuidedLogisticRegression(
            l2_strength=1.0,
            class_weight="balanced",
            max_iter=500,
            tol=1e-6,
            knowledge_constraints=constraints,
            knowledge_lambda=KNOWLEDGE_GUIDED_LAMBDA_LINEAR,
            random_state=seed,
        ),
    }
    if not include_benchmarks:
        return {name: builders[name] for name in BAYESIAN_MODEL_NAMES}
    return builders


def cleanup_model_artifacts(active_model_names):
    active = set(active_model_names)
    known = set(BAYESIAN_MODEL_NAMES) | set(BENCHMARK_MODEL_NAMES)
    for name in known.difference(active):
        cm_path = figure_output_path(f"confusion_matrix_{name.lower()}.png")
        safe_remove_file(cm_path)


def build_pipeline(estimator):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def safe_roc_auc(y_true, probs):
    try:
        return roc_auc_score(y_true, probs)
    except ValueError:
        return np.nan


def safe_pr_auc(y_true, probs):
    try:
        return average_precision_score(y_true, probs)
    except ValueError:
        return np.nan


def safe_log_loss(y_true, probs):
    try:
        return log_loss(y_true, np.clip(probs, 1e-9, 1.0 - 1e-9), labels=[0, 1])
    except ValueError:
        return np.nan


def expected_calibration_error(y_true, probs, n_bins=10):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins[1:-1], right=True)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(probs[mask])
        ece += np.mean(mask) * np.abs(acc - conf)
    return float(ece)


def compute_metrics(y_true, preds, probs):
    return {
        "acc": accuracy_score(y_true, preds),
        "prec": precision_score(y_true, preds, zero_division=0),
        "rec": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "mcc": matthews_corrcoef(y_true, preds),
        "roc_auc": safe_roc_auc(y_true, probs),
        "pr_auc": safe_pr_auc(y_true, probs),
        "brier": brier_score_loss(y_true, probs),
        "logloss": safe_log_loss(y_true, probs),
        "ece": expected_calibration_error(y_true, probs),
    }


def init_metrics_log(model_names):
    metric_keys = [
        "acc",
        "prec",
        "rec",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier",
        "logloss",
        "ece",
    ]
    metrics_log = {}
    for name in model_names:
        metrics_log[name] = {k: [] for k in metric_keys}
        metrics_log[name]["fold_probs"] = []
        metrics_log[name]["fold_preds"] = []
        metrics_log[name]["fold_y"] = []
    return metrics_log


def cross_validation_and_modeling(X, y, model_builders):
    print(f"Starting repeated CV ({CV_SPLITS} folds x {CV_REPEATS} repeats)...")
    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED
    )

    metrics_log = init_metrics_log(model_builders.keys())

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, builder in model_builders.items():
            model = build_pipeline(builder())
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            fold_metrics = compute_metrics(y_test, preds, probs)
            for metric_name, value in fold_metrics.items():
                metrics_log[name][metric_name].append(value)

            metrics_log[name]["fold_probs"].append(probs)
            metrics_log[name]["fold_preds"].append(preds)
            metrics_log[name]["fold_y"].append(y_test.to_numpy())

        if fold % CV_SPLITS == 0:
            print(f"Completed fold {fold} of {CV_SPLITS * CV_REPEATS}")

    print("\n--- Repeated CV Validation (Train Set) ---")
    for name in model_builders.keys():
        mcc_mean = np.nanmean(metrics_log[name]["mcc"])
        mcc_std = np.nanstd(metrics_log[name]["mcc"], ddof=1)
        roc_mean = np.nanmean(metrics_log[name]["roc_auc"])
        print(
            f"[{name}] MCC: {mcc_mean:.4f} +/- {mcc_std:.4f} | ROC-AUC: {roc_mean:.4f}"
        )

    return metrics_log


def summarize_cv_metrics(metrics_log):
    rows = []
    metric_keys = [
        "acc",
        "prec",
        "rec",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier",
        "logloss",
        "ece",
    ]

    for name, log in metrics_log.items():
        row = {"Model": name, "Model_Role": get_model_role(name)}
        for metric in metric_keys:
            values = np.asarray(log[metric], dtype=float)
            n = np.sum(~np.isnan(values))
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values, ddof=1) if n > 1 else 0.0
            ci95 = 1.96 * std_val / np.sqrt(n) if n > 1 else np.nan
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
            row[f"{metric}_ci95"] = ci95
        rows.append(row)

    summary_df = (
        pd.DataFrame(rows)
        .sort_values("mcc_mean", ascending=False)
        .reset_index(drop=True)
    )
    summary_df.to_csv(data_output_path("cv_metrics_summary.csv"), index=False)
    return summary_df


def plot_holdout_discrimination_curves(fitted_models, X_test, y_test):
    colors = {
        "BNN": "#1f77b4",
        "DeepEnsemble": "#ff7f0e",
        "BLR": "#f28e2b",
        "GPC": "#4e79a7",
        "LR": "#2ca02c",
    }

    # ROC curves from independent holdout predictions.
    plt.figure(figsize=(10, 8))
    for name, model in fitted_models.items():
        probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-9, 1.0 - 1e-9)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = safe_roc_auc(y_test, probs)
        plt.plot(
            fpr,
            tpr,
            color=colors.get(name, "black"),
            linewidth=2.0,
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Holdout Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figure_output_path("roc_curve.png"), dpi=300)
    plt.close()

    # Precision-recall curves from independent holdout predictions.
    plt.figure(figsize=(10, 8))
    for name, model in fitted_models.items():
        probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-9, 1.0 - 1e-9)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = safe_pr_auc(y_test, probs)
        plt.plot(
            recall,
            precision,
            color=colors.get(name, "black"),
            linewidth=2.0,
            label=f"{name} (AUC = {pr_auc:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Holdout Test Set)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(figure_output_path("pr_curve.png"), dpi=300)
    plt.close()


def plot_holdout_confusion_matrices(fitted_models, X_test, y_test):
    for name, model in fitted_models.items():
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix (Holdout Test Set)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(figure_output_path(f"confusion_matrix_{name.lower()}.png"), dpi=300)
        plt.close()


def smooth_calibration_points(mean_pred, frac_pos, n_points=300):
    mean_pred = np.asarray(mean_pred, dtype=float)
    frac_pos = np.asarray(frac_pos, dtype=float)

    valid = np.isfinite(mean_pred) & np.isfinite(frac_pos)
    mean_pred = mean_pred[valid]
    frac_pos = frac_pos[valid]
    if len(mean_pred) < 2:
        return mean_pred, frac_pos

    curve_df = (
        pd.DataFrame({"x": mean_pred, "y": frac_pos})
        .groupby("x", as_index=False)["y"]
        .mean()
        .sort_values("x")
    )
    x = curve_df["x"].to_numpy(dtype=float)
    y = curve_df["y"].to_numpy(dtype=float)

    if len(x) < 2:
        return x, y

    x_smooth = np.linspace(x.min(), x.max(), n_points)
    if len(x) >= 3:
        try:
            y_smooth = PchipInterpolator(x, y, extrapolate=False)(x_smooth)
        except Exception:
            y_smooth = np.interp(x_smooth, x, y)
    else:
        y_smooth = np.interp(x_smooth, x, y)

    return x_smooth, np.clip(y_smooth, 0.0, 1.0)


def statistical_validation(metrics_log, metric_name="f1"):
    models = list(metrics_log.keys())
    if len(models) < 2:
        print("Skipping statistical test: fewer than two models.")
        return

    score_lists = [np.asarray(metrics_log[m][metric_name], dtype=float) for m in models]
    min_len = min(len(s) for s in score_lists)

    if min_len < 3:
        print("Skipping statistical test: not enough folds.")
        return

    trimmed_scores = [s[:min_len] for s in score_lists]
    stats_path = data_output_path("friedman_test.txt")
    nemenyi_path = data_output_path("nemenyi_test_results.csv")

    if len(models) == 2:
        try:
            stat, p_value = wilcoxon(
                trimmed_scores[0], trimmed_scores[1], zero_method="wilcox"
            )
        except ValueError:
            stat, p_value = np.nan, np.nan

        print("\n--- Wilcoxon Signed-Rank Test ---")
        print(
            f"Metric: {metric_name} | Models: {models[0]} vs {models[1]} "
            f"| Statistic: {stat:.4f}, p-value: {p_value:.4e}"
        )
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"test=wilcoxon_signed_rank\n")
            f.write(f"metric={metric_name}\n")
            f.write(f"model_a={models[0]}\n")
            f.write(f"model_b={models[1]}\n")
            f.write(f"statistic={stat:.6f}\n")
            f.write(f"p_value={p_value:.10f}\n")

        safe_remove_file(nemenyi_path)
        return

    stat, p_value = friedmanchisquare(*trimmed_scores)
    print("\n--- Friedman Test ---")
    print(f"Metric: {metric_name} | Statistic: {stat:.4f}, p-value: {p_value:.4e}")

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("test=friedman\n")
        f.write(f"metric={metric_name}\n")
        f.write(f"statistic={stat:.6f}\n")
        f.write(f"p_value={p_value:.10f}\n")

    if p_value < 0.05:
        data = np.vstack(trimmed_scores).T
        nemenyi_df = sp.posthoc_nemenyi_friedman(data)
        nemenyi_df.columns = models
        nemenyi_df.index = models
        try:
            nemenyi_df.to_csv(nemenyi_path)
            print("Significant difference found. Nemenyi post-hoc results saved.")
        except PermissionError:
            fallback = data_output_path(
                f"nemenyi_test_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            nemenyi_df.to_csv(fallback)
            print(
                "Significant difference found. Could not overwrite locked "
                f"file; saved Nemenyi results to {fallback}."
            )
    else:
        print("No statistically significant difference among models at alpha=0.05.")
        safe_remove_file(nemenyi_path)


def evaluate_holdout(model_builders, X_train, y_train, X_test, y_test):
    print("\nTraining on train split and evaluating holdout test set...")
    rows = []
    fitted_models = {}

    for name, builder in model_builders.items():
        model = build_pipeline(builder())
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, preds, probs)

        rows.append(
            {
                "Model": name,
                "Model_Role": get_model_role(name),
                "Accuracy": metrics["acc"],
                "Precision": metrics["prec"],
                "Recall": metrics["rec"],
                "F1": metrics["f1"],
                "MCC": metrics["mcc"],
                "ROC_AUC": metrics["roc_auc"],
                "PR_AUC": metrics["pr_auc"],
                "Brier": metrics["brier"],
                "LogLoss": metrics["logloss"],
                "ECE": metrics["ece"],
            }
        )
        fitted_models[name] = model

    holdout_df = (
        pd.DataFrame(rows).sort_values("MCC", ascending=False).reset_index(drop=True)
    )
    holdout_df.to_csv(data_output_path("holdout_metrics.csv"), index=False)
    print(holdout_df.to_string(index=False))
    return fitted_models, holdout_df


def plot_holdout_calibration(fitted_models, X_test, y_test):
    colors = {
        "BNN": "#1f77b4",
        "DeepEnsemble": "#ff7f0e",
        "BLR": "#f28e2b",
        "GPC": "#4e79a7",
        "LR": "#2ca02c",
    }

    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    rows = []
    for name, model in fitted_models.items():
        probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-9, 1.0 - 1e-9)
        frac_pos, mean_pred = calibration_curve(
            y_test, probs, n_bins=8, strategy="quantile"
        )
        x_smooth, y_smooth = smooth_calibration_points(mean_pred, frac_pos)
        color = colors.get(name, "black")
        if len(x_smooth) >= 2:
            plt.plot(x_smooth, y_smooth, color=color, linewidth=2.0, label=name)
        else:
            plt.plot(mean_pred, frac_pos, color=color, linewidth=2.0, label=name)
        plt.scatter(
            mean_pred,
            frac_pos,
            color=color,
            marker="o",
            s=28,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        rows.append(
            {
                "Model": name,
                "Model_Role": get_model_role(name),
                "ECE": expected_calibration_error(y_test, probs),
            }
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Holdout Calibration Curve (Smoothed)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figure_output_path("calibration_curve_holdout.png"), dpi=300)
    plt.close()

    calibration_df = (
        pd.DataFrame(rows).sort_values("ECE", ascending=True).reset_index(drop=True)
    )
    calibration_df.to_csv(data_output_path("holdout_calibration_summary.csv"), index=False)
    return calibration_df


def choose_final_model(
    cv_summary,
    holdout_metrics,
    spatial_metrics,
    deployment_candidates=BAYESIAN_MODEL_NAMES,
):
    selection_df = cv_summary[
        [
            "Model",
            "Model_Role",
            "mcc_mean",
            "roc_auc_mean",
            "brier_mean",
            "logloss_mean",
            "ece_mean",
        ]
    ].copy()
    selection_df = selection_df.merge(
        holdout_metrics[
            ["Model", "Model_Role", "MCC", "ROC_AUC", "Brier", "LogLoss", "ECE"]
        ].rename(columns={"Model_Role": "Model_Role_Holdout"}),
        on="Model",
        how="left",
    )
    selection_df["Model_Role"] = (
        selection_df["Model_Role"]
        .fillna(selection_df["Model_Role_Holdout"])
        .fillna(selection_df["Model"].map(get_model_role))
    )
    selection_df = selection_df.drop(columns=["Model_Role_Holdout"])

    if spatial_metrics is not None:
        selection_df = selection_df.merge(
            spatial_metrics[
                ["Model", "Model_Role", "Spatial_MCC_Mean", "Spatial_ROC_AUC_Mean"]
            ].rename(columns={"Model_Role": "Model_Role_Spatial"}),
            on="Model",
            how="left",
        )
        selection_df["Model_Role"] = (
            selection_df["Model_Role"]
            .fillna(selection_df["Model_Role_Spatial"])
            .fillna(selection_df["Model"].map(get_model_role))
        )
        selection_df = selection_df.drop(columns=["Model_Role_Spatial"])
    else:
        selection_df["Spatial_MCC_Mean"] = np.nan
        selection_df["Spatial_ROC_AUC_Mean"] = np.nan

    # Rank-based multi-criterion score; lower is better.
    selection_df["rank_cv_mcc"] = selection_df["mcc_mean"].rank(
        ascending=False, method="min"
    )
    selection_df["rank_holdout_mcc"] = selection_df["MCC"].rank(
        ascending=False, method="min"
    )
    selection_df["rank_spatial_mcc"] = selection_df["Spatial_MCC_Mean"].rank(
        ascending=False, method="min"
    )
    selection_df["rank_holdout_brier"] = selection_df["Brier"].rank(
        ascending=True, method="min"
    )
    selection_df["rank_holdout_ece"] = selection_df["ECE"].rank(
        ascending=True, method="min"
    )

    spatial_available = selection_df["Spatial_MCC_Mean"].notna().any()
    w_cv = 0.20
    w_holdout = 0.35
    w_spatial = 0.30 if spatial_available else 0.0
    w_brier = 0.10
    w_ece = 0.05
    total_w = w_cv + w_holdout + w_spatial + w_brier + w_ece

    selection_df["selection_score"] = (
        w_cv * selection_df["rank_cv_mcc"]
        + w_holdout * selection_df["rank_holdout_mcc"]
        + w_spatial
        * selection_df["rank_spatial_mcc"].fillna(selection_df["rank_cv_mcc"])
        + w_brier * selection_df["rank_holdout_brier"]
        + w_ece * selection_df["rank_holdout_ece"]
    ) / total_w

    selection_df = selection_df.sort_values(
        ["selection_score", "MCC", "mcc_mean"], ascending=[True, False, False]
    ).reset_index(drop=True)
    selection_df["Overall_Rank"] = np.arange(1, len(selection_df) + 1, dtype=int)

    deployment_candidates = tuple(deployment_candidates)
    selection_df["Deployment_Eligible"] = selection_df["Model"].isin(
        deployment_candidates
    )

    eligible_df = selection_df.loc[selection_df["Deployment_Eligible"]].copy()
    if eligible_df.empty:
        raise ValueError(
            "No deployment-eligible Bayesian models found in selection table."
        )
    eligible_df = eligible_df.sort_values(
        ["selection_score", "MCC", "mcc_mean"], ascending=[True, False, False]
    ).reset_index(drop=True)

    best_overall_model = selection_df.loc[0, "Model"]
    best_bayesian_model = eligible_df.loc[0, "Model"]
    selection_df["Deployment_Selected"] = selection_df["Model"].eq(best_bayesian_model)

    selection_df.to_csv(data_output_path("model_selection_table.csv"), index=False)
    return best_bayesian_model, best_overall_model, selection_df


def bootstrap_holdout_intervals(fitted_models, X_test, y_test, n_boot=1000):
    y_arr = np.asarray(y_test, dtype=int)
    n = len(y_arr)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for name, model in fitted_models.items():
        probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-9, 1.0 - 1e-9)
        preds = (probs >= 0.5).astype(int)

        metric_store = {"MCC": [], "ROC_AUC": [], "PR_AUC": [], "Brier": [], "ECE": []}
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, n)
            y_b = y_arr[idx]
            p_b = probs[idx]
            pred_b = preds[idx]

            metric_store["MCC"].append(matthews_corrcoef(y_b, pred_b))
            metric_store["ROC_AUC"].append(safe_roc_auc(y_b, p_b))
            metric_store["PR_AUC"].append(safe_pr_auc(y_b, p_b))
            metric_store["Brier"].append(brier_score_loss(y_b, p_b))
            metric_store["ECE"].append(expected_calibration_error(y_b, p_b))

        row = {"Model": name, "Model_Role": get_model_role(name)}
        for metric_name, values in metric_store.items():
            arr = np.asarray(values, dtype=float)
            row[f"{metric_name}_Median"] = np.nanmedian(arr)
            row[f"{metric_name}_CI_Low"] = np.nanpercentile(arr, 2.5)
            row[f"{metric_name}_CI_High"] = np.nanpercentile(arr, 97.5)
        rows.append(row)

    ci_df = (
        pd.DataFrame(rows)
        .sort_values("MCC_Median", ascending=False)
        .reset_index(drop=True)
    )
    ci_df.to_csv(data_output_path("holdout_bootstrap_ci.csv"), index=False)
    return ci_df


def spatial_block_validation(X, y, coords, model_builders):
    print("\nRunning spatial block CV (GroupKFold with KMeans blocks)...")
    valid_mask = coords[[LON_COL, LAT_COL]].notna().all(axis=1)
    X_valid = X.loc[valid_mask]
    y_valid = y.loc[valid_mask]
    coords_valid = coords.loc[valid_mask, [LON_COL, LAT_COL]]

    n_samples = len(X_valid)
    if n_samples < 80:
        print(
            "Skipping spatial block CV: insufficient samples after coordinate filtering."
        )
        return None

    n_blocks = min(10, max(4, n_samples // 30))
    n_splits = min(5, n_blocks)
    if n_splits < 3:
        print("Skipping spatial block CV: not enough blocks.")
        return None

    kmeans = KMeans(n_clusters=n_blocks, random_state=RANDOM_SEED, n_init=10)
    groups = kmeans.fit_predict(coords_valid.values)
    group_cv = GroupKFold(n_splits=n_splits)

    rows = []
    for name, builder in model_builders.items():
        mcc_scores = []
        roc_scores = []
        for train_idx, test_idx in group_cv.split(X_valid, y_valid, groups=groups):
            X_tr, X_te = X_valid.iloc[train_idx], X_valid.iloc[test_idx]
            y_tr, y_te = y_valid.iloc[train_idx], y_valid.iloc[test_idx]

            model = build_pipeline(builder())
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            probs = model.predict_proba(X_te)[:, 1]

            mcc_scores.append(matthews_corrcoef(y_te, preds))
            roc_scores.append(safe_roc_auc(y_te, probs))

        rows.append(
            {
                "Model": name,
                "Model_Role": get_model_role(name),
                "Spatial_MCC_Mean": np.nanmean(mcc_scores),
                "Spatial_MCC_Std": (
                    np.nanstd(mcc_scores, ddof=1) if len(mcc_scores) > 1 else 0.0
                ),
                "Spatial_ROC_AUC_Mean": np.nanmean(roc_scores),
                "Spatial_ROC_AUC_Std": (
                    np.nanstd(roc_scores, ddof=1) if len(roc_scores) > 1 else 0.0
                ),
            }
        )

    spatial_df = (
        pd.DataFrame(rows)
        .sort_values("Spatial_MCC_Mean", ascending=False)
        .reset_index(drop=True)
    )
    spatial_df.to_csv(data_output_path("spatial_cv_metrics.csv"), index=False)
    print(spatial_df.to_string(index=False))
    return spatial_df


def explainable_ai(model, X_reference):
    print("Generating SHAP explanations...")
    if len(X_reference) < 10:
        print("Skipping SHAP: too few samples.")
        return

    for old_file in glob.glob(figure_output_path("shap_dependence_top_*.png")):
        safe_remove_file(old_file)

    try:
        background_n = min(30, len(X_reference))
        explain_n = min(80, len(X_reference))
        background_df = shap.sample(X_reference, background_n, random_state=RANDOM_SEED)
        X_explain_df = shap.sample(X_reference, explain_n, random_state=RANDOM_SEED)

        # SHAP can fail when passing sklearn.Pipeline directly due attribute handling.
        # Wrapping prediction as a function avoids mutating pipeline internals.
        feature_cols = X_reference.columns.tolist()

        def predict_proba_wrapper(data):
            data_df = pd.DataFrame(data, columns=feature_cols)
            return model.predict_proba(data_df)

        explainer = shap.KernelExplainer(
            predict_proba_wrapper, background_df.to_numpy(), link="logit"
        )
        shap_values = explainer.shap_values(X_explain_df.to_numpy(), nsamples=150)

        if isinstance(shap_values, list):
            shap_values_target = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values_target = shap_values[:, :, 1]
        else:
            shap_values_target = shap_values

        plt.figure()
        shap.summary_plot(shap_values_target, X_explain_df, show=False)
        plt.tight_layout()
        plt.savefig(figure_output_path("shap_summary.png"), dpi=300)
        plt.close()

        mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:3]
        top_features = [X_reference.columns[i] for i in top_indices]
        print(f"Top 3 SHAP features: {top_features}")

        for i, feature in enumerate(top_features, start=1):
            plt.figure()
            shap.dependence_plot(feature, shap_values_target, X_explain_df, show=False)
            plt.tight_layout()
            plt.savefig(figure_output_path(f"shap_dependence_top_{i}_{feature}.png"), dpi=300)
            plt.close()
    except Exception as exc:
        print(f"Warning: SHAP explanation skipped due to error: {exc}")


def build_breaks(valid_probs, n_classes=NUM_CLASSES):
    valid_probs = np.asarray(valid_probs, dtype=float)
    valid_probs = valid_probs[np.isfinite(valid_probs)]
    if len(valid_probs) == 0:
        raise ValueError("No valid probabilities available for class breaks.")

    if np.isclose(valid_probs.min(), valid_probs.max()):
        return np.linspace(0.0, 1.0, n_classes + 1).tolist()

    try:
        breaks = jenkspy.jenks_breaks(valid_probs.tolist(), n_classes=n_classes)
    except Exception:
        breaks = np.quantile(valid_probs, np.linspace(0, 1, n_classes + 1)).tolist()

    breaks = np.array(sorted(set(np.round(breaks, 10))), dtype=float)
    if len(breaks) < n_classes + 1:
        breaks = np.quantile(valid_probs, np.linspace(0, 1, n_classes + 1))

    if len(np.unique(breaks)) < n_classes + 1:
        breaks = np.linspace(valid_probs.min(), valid_probs.max(), n_classes + 1)

    breaks[0] = min(breaks[0], valid_probs.min())
    breaks[-1] = max(breaks[-1], valid_probs.max())
    return breaks.tolist()


def save_raster(data, filename, crs, transform):
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)


def predict_probs_and_uncertainty(model, X_df):
    probs = np.clip(model.predict_proba(X_df)[:, 1], 0.0, 1.0)

    # Use Bayesian uncertainty when available (BLR posterior or GPC latent variance),
    # else fallback to decision-boundary confidence.
    if isinstance(model, Pipeline) and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "predict_uncertainty"):
            X_transformed = model[:-1].transform(X_df)
            _, std_prob = clf.predict_uncertainty(X_transformed, n_samples=500)
            std_prob = np.asarray(std_prob, dtype=float)
            std_prob = np.clip(std_prob, 0.0, None)
            max_std = np.nanmax(std_prob)
            if np.isfinite(max_std) and max_std > 0:
                uncertainty = np.clip(std_prob / max_std, 0.0, 1.0)
            else:
                uncertainty = np.zeros_like(probs)
            return probs, uncertainty
        if hasattr(clf, "latent_mean_and_variance"):
            try:
                X_transformed = model[:-1].transform(X_df)
                _, latent_var = clf.latent_mean_and_variance(X_transformed)
                latent_std = np.sqrt(np.clip(np.asarray(latent_var, dtype=float), 0.0, None))
                max_std = np.nanmax(latent_std)
                if np.isfinite(max_std) and max_std > 0:
                    uncertainty = np.clip(latent_std / max_std, 0.0, 1.0)
                else:
                    uncertainty = np.zeros_like(probs)
                return probs, uncertainty
            except Exception:
                pass

    uncertainty = np.clip(1.0 - (np.abs(probs - 0.5) * 2.0), 0.0, 1.0)
    return probs, uncertainty


def generate_spatial_maps(model, df_full, features, shapefile_path):
    print("Generating susceptibility and uncertainty maps...")
    df_raw = df_full.copy()
    if LON_COL not in df_raw.columns or LAT_COL not in df_raw.columns:
        raise ValueError(
            "Longitude/Latitude columns are missing for spatial map generation."
        )

    coords_valid_mask = df_raw[[LON_COL, LAT_COL]].notna().all(axis=1)
    df_coords = df_raw.loc[coords_valid_mask].copy()
    coords = df_coords[[LON_COL, LAT_COL]].values

    probs, uncertainty = predict_probs_and_uncertainty(model, df_coords[features])

    lon_min, lon_max = df_coords[LON_COL].min(), df_coords[LON_COL].max()
    lat_min, lat_max = df_coords[LAT_COL].min(), df_coords[LAT_COL].max()

    grid_lon, grid_lat = np.mgrid[lon_min:lon_max:500j, lat_min:lat_max:500j]
    print("Interpolating points to grid...")
    grid_susceptibility = griddata(coords, probs, (grid_lon, grid_lat), method="linear")
    grid_uncertainty = griddata(
        coords, uncertainty, (grid_lon, grid_lat), method="linear"
    )

    # Fill edge NaNs with nearest interpolation.
    if np.isnan(grid_susceptibility).any():
        nearest = griddata(coords, probs, (grid_lon, grid_lat), method="nearest")
        grid_susceptibility = np.where(
            np.isnan(grid_susceptibility), nearest, grid_susceptibility
        )
    if np.isnan(grid_uncertainty).any():
        nearest = griddata(coords, uncertainty, (grid_lon, grid_lat), method="nearest")
        grid_uncertainty = np.where(
            np.isnan(grid_uncertainty), nearest, grid_uncertainty
        )

    grid_susceptibility = np.clip(grid_susceptibility, 0.0, 1.0)
    grid_uncertainty = np.clip(grid_uncertainty, 0.0, 1.0)

    grid_susc_orig = grid_susceptibility.copy()
    grid_uncert_orig = grid_uncertainty.copy()

    # Raster orientation for geotiff export.
    grid_susceptibility = np.flipud(grid_susceptibility.T).astype(np.float32)
    grid_uncertainty = np.flipud(grid_uncertainty.T).astype(np.float32)

    print("Loading AOI geometry for masking...")
    try:
        if shapefile_path is None or not os.path.exists(shapefile_path):
            raise FileNotFoundError("Shapefile not found.")
        aoi = gpd.read_file(shapefile_path)
        crs = aoi.crs
        geometries = [mapping(shape) for shape in aoi.geometry]
    except Exception as exc:
        print(
            f"Warning: Could not load shapefile; using EPSG:4326 without masking. ({exc})"
        )
        crs = "EPSG:4326"
        geometries = None

    pixel_width = (lon_max - lon_min) / 500
    pixel_height = (lat_max - lat_min) / 500
    transform = from_origin(lon_min, lat_max, pixel_width, pixel_height)

    valid_probs = grid_susceptibility[np.isfinite(grid_susceptibility)]
    rng = np.random.default_rng(RANDOM_SEED)
    if len(valid_probs) > 10000:
        valid_probs = rng.choice(valid_probs, 10000, replace=False)

    breaks = build_breaks(valid_probs, n_classes=NUM_CLASSES)
    print(f"Class breaks: {breaks}")

    grid_reclass = (
        np.digitize(grid_susceptibility, bins=breaks[1:-1], right=True).astype(
            np.float32
        )
        + 1.0
    )
    grid_reclass[np.isnan(grid_susceptibility)] = np.nan

    raw_lsm = data_output_path("LSM_Map_Raw.tif")
    raw_uncert = data_output_path("Uncertainty_Map_Raw.tif")
    raw_class = data_output_path("LSM_Map_Classified.tif")
    save_raster(grid_susceptibility, raw_lsm, crs, transform)
    save_raster(grid_uncertainty, raw_uncert, crs, transform)
    save_raster(grid_reclass, raw_class, crs, transform)

    if geometries:
        for filename in [
            "LSM_Map_Raw.tif",
            "Uncertainty_Map_Raw.tif",
            "LSM_Map_Classified.tif",
        ]:
            in_path = data_output_path(filename)
            out_path = data_output_path(f"{filename}.masked")
            safe_remove_file(out_path)
            safe_remove_file(out_path + ".aux.xml")

            with rasterio.open(in_path) as src:
                out_image, out_transform = mask(src, geometries, crop=True)
                out_meta = src.meta.copy()

            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )
            with rasterio.open(out_path, "w", **out_meta) as dest:
                dest.write(out_image)
            os.replace(out_path, in_path)

    print("\n--- SCAI Calculation ---")
    reclass_path = data_output_path("LSM_Map_Classified.tif")

    scai_log = []
    with rasterio.open(reclass_path) as src:
        reclass_raster = src.read(1)
        valid_mask = np.isfinite(reclass_raster) & (reclass_raster > 0)
        total_valid_pixels = np.sum(valid_mask)

    ls_points = df_raw[df_raw[TARGET_COL] == 1].copy()
    ls_probs, _ = predict_probs_and_uncertainty(model, ls_points[features])
    ls_classes = np.digitize(ls_probs, bins=breaks[1:-1], right=True) + 1

    total_ls_points = len(ls_classes)
    tiers = ["Very Low", "Low", "Moderate", "High", "Very High"]
    alpha = 0.5  # Jeffreys smoothing for stable finite SCAI estimates.
    for i in range(1, NUM_CLASSES + 1):
        area_pixels = np.sum(reclass_raster[valid_mask] == i)
        area_pct = (
            (area_pixels / total_valid_pixels) * 100.0
            if total_valid_pixels > 0
            else 0.0
        )

        ls_count = int(np.sum(ls_classes == i))
        ls_pct = (ls_count / total_ls_points) * 100.0 if total_ls_points > 0 else 0.0
        scai = area_pct / ls_pct if ls_pct > 0 else np.nan
        area_prop_smoothed = (area_pixels + alpha) / (
            total_valid_pixels + (NUM_CLASSES * alpha)
        )
        ls_prop_smoothed = (ls_count + alpha) / (
            total_ls_points + (NUM_CLASSES * alpha)
        )
        scai_smoothed = area_prop_smoothed / ls_prop_smoothed

        scai_log.append(
            {
                "Tier": tiers[i - 1],
                "Area_Pixels": int(area_pixels),
                "Landslide_Count": int(ls_count),
                "Area_Pct": area_pct,
                "Landslide_Pct": ls_pct,
                "SCAI": scai,
                "SCAI_Smoothed": scai_smoothed,
            }
        )

    scai_df = pd.DataFrame(scai_log)
    print(scai_df.to_string(index=False))
    scai_df.to_csv(data_output_path("SCAI_Analysis.csv"), index=False)

    all_probs, all_uncertainty = predict_probs_and_uncertainty(model, df_raw[features])
    all_classes = np.digitize(all_probs, bins=breaks[1:-1], right=True) + 1

    df_export = df_raw.copy()
    df_export["Predicted_Prob"] = all_probs
    df_export["Predicted_Class"] = all_classes
    df_export["Uncertainty"] = all_uncertainty
    output_path = data_output_path("LSM_Data.csv")
    try:
        df_export.to_csv(output_path, index=False)
    except PermissionError:
        fallback = os.path.join(
            DATA_OUTPUT_DIR,
            f"LSM_Data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        df_export.to_csv(fallback, index=False)
        print(
            f"Warning: Could not overwrite {output_path}; saved to {fallback} instead."
        )

    print("Exporting spatial grid data to CSV...")
    grid_classes = (
        np.digitize(grid_susc_orig, bins=breaks[1:-1], right=True).astype(int) + 1
    )
    grid_df = pd.DataFrame(
        {
            "Longitude": grid_lon.ravel(),
            "Latitude": grid_lat.ravel(),
            "Susceptibility": grid_susc_orig.ravel(),
            "Uncertainty": grid_uncert_orig.ravel(),
            "Predicted_Class": grid_classes.ravel(),
        }
    ).dropna()
    grid_df.to_csv(data_output_path("LSM_Grid_Data.csv"), index=False)


def save_model_report(
    selected_features,
    dropped_features,
    cv_summary,
    holdout_metrics,
    holdout_bootstrap_ci,
    spatial_metrics,
    selection_table,
    calibration_summary,
    final_model_name,
    best_overall_model,
    train_size,
    test_size,
):
    report = {
        "seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "cv_repeats": CV_REPEATS,
        "vif_threshold": VIF_THRESHOLD,
        "knowledge_guided_priors": KNOWLEDGE_GUIDED_PRIORS,
        "knowledge_loss_lambda_deep": KNOWLEDGE_GUIDED_LAMBDA_DEEP,
        "knowledge_loss_lambda_linear": KNOWLEDGE_GUIDED_LAMBDA_LINEAR,
        "train_rows": int(train_size),
        "test_rows": int(test_size),
        "selected_features": selected_features,
        "dropped_features_vif": dropped_features,
        "final_model": final_model_name,
        "best_overall_model": best_overall_model,
        "deployment_candidates": list(BAYESIAN_MODEL_NAMES),
        "benchmark_models": list(BENCHMARK_MODEL_NAMES),
        "selection_policy": (
            "All models are benchmarked under identical validation. Final deployment "
            "is restricted to Bayesian and uncertainty-aware candidates "
            "(BNN, DeepEnsemble, BLR, GPC) under the same knowledge-guided "
            "regularization policy."
        ),
        "cv_summary": cv_summary.to_dict(orient="records"),
        "holdout_metrics": holdout_metrics.to_dict(orient="records"),
        "holdout_bootstrap_ci": holdout_bootstrap_ci.to_dict(orient="records"),
        "spatial_cv_metrics": (
            spatial_metrics.to_dict(orient="records")
            if spatial_metrics is not None
            else None
        ),
        "model_selection_table": selection_table.to_dict(orient="records"),
        "holdout_calibration": calibration_summary.to_dict(orient="records"),
    }
    with open(data_output_path("model_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    set_seed()
    ensure_output_dir()

    data_file = resolve_data_file(DATA_FILE_CANDIDATES)
    print(f"Using input dataset: {data_file}")
    df, X, y, coords, features = load_dataset(data_file)
    review_dataset_characteristics(df, X, y, features)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        coords_train,
        coords_test,
    ) = train_test_split(
        X,
        y,
        coords,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("Train class distribution:")
    print(y_train.value_counts().sort_index())
    print("Test class distribution:")
    print(y_test.value_counts().sort_index())

    selected_features, dropped_features, vif_df = select_features_vif(
        X_train, threshold=VIF_THRESHOLD
    )
    vif_df.to_csv(data_output_path("vif_selected_features.csv"), index=False)
    plot_vif_tolerance_curve(vif_df, threshold=VIF_THRESHOLD)

    model_builders = build_model_builders(
        selected_features,
        include_benchmarks=INCLUDE_BENCHMARK_MODELS,
    )
    cleanup_model_artifacts(model_builders.keys())
    print(f"Active model scope: {list(model_builders.keys())}")
    print(f"Bayesian deployment candidates: {list(BAYESIAN_MODEL_NAMES)}")
    print(f"Benchmark-only models: {list(BENCHMARK_MODEL_NAMES)}")

    metrics_log = cross_validation_and_modeling(
        X_train[selected_features], y_train, model_builders
    )
    cv_summary = summarize_cv_metrics(metrics_log)
    print("\nCV summary (sorted by MCC mean):")
    print(
        cv_summary[
            [
                "Model",
                "mcc_mean",
                "roc_auc_mean",
                "f1_mean",
                "brier_mean",
                "logloss_mean",
                "ece_mean",
            ]
        ].to_string(index=False)
    )

    statistical_validation(metrics_log, metric_name="f1")

    spatial_metrics = spatial_block_validation(
        X_train[selected_features], y_train, coords_train, model_builders
    )

    fitted_models, holdout_metrics = evaluate_holdout(
        model_builders,
        X_train[selected_features],
        y_train,
        X_test[selected_features],
        y_test,
    )
    plot_holdout_discrimination_curves(fitted_models, X_test[selected_features], y_test)
    plot_holdout_confusion_matrices(fitted_models, X_test[selected_features], y_test)
    calibration_summary = plot_holdout_calibration(
        fitted_models, X_test[selected_features], y_test
    )
    holdout_bootstrap_ci = bootstrap_holdout_intervals(
        fitted_models, X_test[selected_features], y_test, n_boot=HOLDOUT_BOOTSTRAP_ITER
    )
    print("\nHoldout bootstrap 95% CI summary:")
    print(
        holdout_bootstrap_ci[
            [
                "Model",
                "MCC_Median",
                "MCC_CI_Low",
                "MCC_CI_High",
                "ROC_AUC_Median",
                "ROC_AUC_CI_Low",
                "ROC_AUC_CI_High",
            ]
        ].to_string(index=False)
    )

    final_model_name, best_overall_model, selection_table = choose_final_model(
        cv_summary,
        holdout_metrics,
        spatial_metrics,
        deployment_candidates=BAYESIAN_MODEL_NAMES,
    )
    print(
        f"\nBest overall benchmark model by ranking (all models): {best_overall_model}"
    )
    print(
        f"Final deployment model (restricted to Bayesian candidates): {final_model_name}"
    )
    print(
        selection_table[
            [
                "Model",
                "Model_Role",
                "selection_score",
                "Overall_Rank",
                "Deployment_Eligible",
                "Deployment_Selected",
                "MCC",
                "Spatial_MCC_Mean",
                "mcc_mean",
            ]
        ].to_string(index=False)
    )
    if best_overall_model != final_model_name:
        print(
            "Deployment justification: benchmark models may score higher on "
            "discrimination, but final map production is restricted to Bayesian "
            "models for posterior uncertainty representation, calibration-aware "
            "probability interpretation, and alignment with the uncertainty-aware "
            "research objective."
        )

    explainable_ai(fitted_models[final_model_name], X_train[selected_features])

    print(f"\nTraining final {final_model_name} on all data for map generation...")
    final_model = build_pipeline(model_builders[final_model_name]())
    final_model.fit(X[selected_features], y)
    generate_spatial_maps(final_model, df, selected_features, SHAPEFILE)

    save_model_report(
        selected_features=selected_features,
        dropped_features=dropped_features,
        cv_summary=cv_summary,
        holdout_metrics=holdout_metrics,
        holdout_bootstrap_ci=holdout_bootstrap_ci,
        spatial_metrics=spatial_metrics,
        selection_table=selection_table,
        calibration_summary=calibration_summary,
        final_model_name=final_model_name,
        best_overall_model=best_overall_model,
        train_size=len(X_train),
        test_size=len(X_test),
    )

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    main()
