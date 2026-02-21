import json
import os
import random
import warnings

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
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import friedmanchisquare
from shapely.geometry import mapping
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
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
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# Paths and reproducibility
OUTPUT_DIR = "results"
DATA_FILE = "Landslide_Data.csv"
SHAPEFILE = None
RANDOM_SEED = 42

# Dataset assumptions
TARGET_COL = "Landslide"
LON_COL = "Longitude"
LAT_COL = "Latitude"

# Modeling configuration
TEST_SIZE = 0.20
VIF_THRESHOLD = 5.0
CV_SPLITS = 5
CV_REPEATS = 3
NUM_CLASSES = 5
HOLDOUT_BOOTSTRAP_ITER = 1200


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary Bayesian logistic regression using Laplace approximation."""

    def __init__(
        self,
        prior_precision=1.0,
        max_iter=400,
        tol=1e-6,
        n_posterior_samples=400,
        random_state=RANDOM_SEED,
    ):
        self.prior_precision = prior_precision
        self.max_iter = max_iter
        self.tol = tol
        self.n_posterior_samples = n_posterior_samples
        self.random_state = random_state

    def _prepare_X(self, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        return X_arr

    def fit(self, X, y):
        X_arr = self._prepare_X(X)
        y_arr = np.asarray(y, dtype=float).ravel()
        unique = np.unique(y_arr)
        if set(unique.tolist()) - {0.0, 1.0}:
            raise ValueError(
                "BayesianLogisticRegression supports binary labels {0,1} only."
            )

        n_samples = X_arr.shape[0]
        X_aug = np.hstack([np.ones((n_samples, 1)), X_arr])
        n_params = X_aug.shape[1]

        prior_diag = np.full(n_params, float(self.prior_precision))
        prior_diag[0] = 0.0  # Unpenalized intercept.

        def nlp_and_grad(w):
            logits = np.clip(X_aug @ w, -35.0, 35.0)
            probs = expit(logits)
            eps = 1e-12
            nlp = -np.sum(
                y_arr * np.log(probs + eps) + (1.0 - y_arr) * np.log(1.0 - probs + eps)
            )
            nlp += 0.5 * np.sum(prior_diag * (w**2))
            grad = X_aug.T @ (probs - y_arr) + prior_diag * w
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
        probs = expit(logits)
        w_diag = probs * (1.0 - probs)
        hessian = X_aug.T @ (X_aug * w_diag[:, None]) + np.diag(prior_diag)
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
        probs = expit(mean / np.maximum(denom, 1e-9))
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
        probs_draws = expit(np.clip(logits_draws, -35.0, 35.0))

        mean_prob = np.mean(probs_draws, axis=1)
        std_prob = np.std(probs_draws, axis=1)
        # Keep deterministic center consistent with Laplace predictive mean when close.
        laplace_mean = expit(
            mean / np.sqrt(1.0 + (np.pi * np.var(logits_draws, axis=1) / 8.0))
        )
        mean_prob = 0.5 * mean_prob + 0.5 * laplace_mean
        return np.clip(mean_prob, 1e-9, 1.0 - 1e-9), std_prob


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    random.seed(seed)


def ensure_output_dir(path=OUTPUT_DIR):
    os.makedirs(path, exist_ok=True)


def load_dataset(filepath):
    print("Loading empirical dataset...")
    df = pd.read_csv(filepath).drop_duplicates().reset_index(drop=True)

    required = {TARGET_COL, LON_COL, LAT_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns missing in dataset: {sorted(missing)}")

    features = [c for c in df.columns if c not in [TARGET_COL, LON_COL, LAT_COL]]
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


def save_correlation_plot(df_features, title, output_filename):
    corr = df_features.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=300)
    plt.close()


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


def build_model_builders(seed=RANDOM_SEED):
    return {
        "BLR": lambda: BayesianLogisticRegression(
            prior_precision=1.5,
            max_iter=500,
            tol=1e-6,
            n_posterior_samples=500,
            random_state=seed,
        ),
        "GPC": lambda: GaussianProcessClassifier(
            kernel=1.0 * Matern(length_scale=1.0, nu=1.5)
            + WhiteKernel(noise_level=1e-5),
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=5,
            random_state=seed,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "SVC": lambda: SVC(
            C=2.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=seed,
        ),
    }


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
        row = {"Model": name}
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
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cv_metrics_summary.csv"), index=False)
    return summary_df


def evaluate_and_visualize_cv(metrics_log, best_baseline_name):
    colors = {"BLR": "#f28e2b", "GPC": "blue", "RandomForest": "green", "SVC": "red"}

    # ROC curves from out-of-fold predictions.
    plt.figure(figsize=(10, 8))
    for name, log in metrics_log.items():
        all_y = np.concatenate(log["fold_y"])
        all_probs = np.concatenate(log["fold_probs"])
        fpr, tpr, _ = roc_curve(all_y, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=colors.get(name, "black"),
            label=f"{name} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Out-of-Fold, Train CV)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=300)
    plt.close()

    # PR curves from out-of-fold predictions.
    plt.figure(figsize=(10, 8))
    for name, log in metrics_log.items():
        all_y = np.concatenate(log["fold_y"])
        all_probs = np.concatenate(log["fold_probs"])
        precision, recall, _ = precision_recall_curve(all_y, all_probs)
        pr_auc = auc(recall, precision)
        plt.plot(
            recall,
            precision,
            color=colors.get(name, "black"),
            label=f"{name} (AUC = {pr_auc:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Out-of-Fold, Train CV)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=300)
    plt.close()

    # Confusion matrices from out-of-fold predictions.
    for name in ["GPC", best_baseline_name]:
        all_y = np.concatenate(metrics_log[name]["fold_y"])
        all_preds = np.concatenate(metrics_log[name]["fold_preds"])
        cm = confusion_matrix(all_y, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix (Out-of-Fold, Train CV)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f"confusion_matrix_{name.lower()}.png"), dpi=300
        )
        plt.close()


def statistical_validation(metrics_log, metric_name="f1"):
    models = list(metrics_log.keys())
    score_lists = [np.asarray(metrics_log[m][metric_name], dtype=float) for m in models]
    min_len = min(len(s) for s in score_lists)

    if min_len < 3:
        print("Skipping Friedman test: not enough folds.")
        return

    trimmed_scores = [s[:min_len] for s in score_lists]
    stat, p_value = friedmanchisquare(*trimmed_scores)
    print("\n--- Friedman Test ---")
    print(f"Metric: {metric_name} | Statistic: {stat:.4f}, p-value: {p_value:.4e}")

    stats_path = os.path.join(OUTPUT_DIR, "friedman_test.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"metric={metric_name}\n")
        f.write(f"statistic={stat:.6f}\n")
        f.write(f"p_value={p_value:.10f}\n")

    if p_value < 0.05:
        data = np.vstack(trimmed_scores).T
        nemenyi_df = sp.posthoc_nemenyi_friedman(data)
        nemenyi_df.columns = models
        nemenyi_df.index = models
        nemenyi_df.to_csv(os.path.join(OUTPUT_DIR, "nemenyi_test_results.csv"))
        print("Significant difference found. Nemenyi post-hoc results saved.")
    else:
        print("No statistically significant difference among models at alpha=0.05.")


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
    holdout_df.to_csv(os.path.join(OUTPUT_DIR, "holdout_metrics.csv"), index=False)
    print(holdout_df.to_string(index=False))
    return fitted_models, holdout_df


def plot_holdout_calibration(fitted_models, X_test, y_test):
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    rows = []
    for name, model in fitted_models.items():
        probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-9, 1.0 - 1e-9)
        frac_pos, mean_pred = calibration_curve(
            y_test, probs, n_bins=10, strategy="quantile"
        )
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=1.5, label=name)
        rows.append({"Model": name, "ECE": expected_calibration_error(y_test, probs)})

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Holdout Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "calibration_curve_holdout.png"), dpi=300)
    plt.close()

    calibration_df = (
        pd.DataFrame(rows).sort_values("ECE", ascending=True).reset_index(drop=True)
    )
    calibration_df.to_csv(
        os.path.join(OUTPUT_DIR, "holdout_calibration_summary.csv"), index=False
    )
    return calibration_df


def choose_final_model(cv_summary, holdout_metrics, spatial_metrics):
    selection_df = cv_summary[
        ["Model", "mcc_mean", "roc_auc_mean", "brier_mean", "logloss_mean", "ece_mean"]
    ].copy()
    selection_df = selection_df.merge(
        holdout_metrics[["Model", "MCC", "ROC_AUC", "Brier", "LogLoss", "ECE"]],
        on="Model",
        how="left",
    )

    if spatial_metrics is not None:
        selection_df = selection_df.merge(
            spatial_metrics[["Model", "Spatial_MCC_Mean", "Spatial_ROC_AUC_Mean"]],
            on="Model",
            how="left",
        )
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
    selection_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_selection_table.csv"), index=False
    )
    best_model = selection_df.loc[0, "Model"]
    return best_model, selection_df


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

        row = {"Model": name}
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
    ci_df.to_csv(os.path.join(OUTPUT_DIR, "holdout_bootstrap_ci.csv"), index=False)
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
    spatial_df.to_csv(os.path.join(OUTPUT_DIR, "spatial_cv_metrics.csv"), index=False)
    print(spatial_df.to_string(index=False))
    return spatial_df


def explainable_ai(model, X_reference):
    print("Generating SHAP explanations...")
    if len(X_reference) < 10:
        print("Skipping SHAP: too few samples.")
        return

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
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=300)
        plt.close()

        mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:3]
        top_features = [X_reference.columns[i] for i in top_indices]
        print(f"Top 3 SHAP features: {top_features}")

        for i, feature in enumerate(top_features, start=1):
            plt.figure()
            shap.dependence_plot(feature, shap_values_target, X_explain_df, show=False)
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUTPUT_DIR, f"shap_dependence_top_{i}_{feature}.png"),
                dpi=300,
            )
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

    # Use epistemic uncertainty when available (BLR posterior), else fallback.
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

    raw_lsm = os.path.join(OUTPUT_DIR, "LSM_Map_Raw.tif")
    raw_uncert = os.path.join(OUTPUT_DIR, "Uncertainty_Map_Raw.tif")
    raw_class = os.path.join(OUTPUT_DIR, "LSM_Map_Classified.tif")
    save_raster(grid_susceptibility, raw_lsm, crs, transform)
    save_raster(grid_uncertainty, raw_uncert, crs, transform)
    save_raster(grid_reclass, raw_class, crs, transform)

    if geometries:
        for fin, fout in [
            ("LSM_Map_Raw.tif", "LSM_Map.tif"),
            ("Uncertainty_Map_Raw.tif", "Uncertainty_Map.tif"),
            ("LSM_Map_Classified.tif", "LSM_Map_Reclassified.tif"),
        ]:
            in_path = os.path.join(OUTPUT_DIR, fin)
            out_path = os.path.join(OUTPUT_DIR, fout)
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
            os.remove(in_path)

    print("\n--- SCAI Calculation ---")
    reclass_path = os.path.join(OUTPUT_DIR, "LSM_Map_Reclassified.tif")
    if not os.path.exists(reclass_path):
        reclass_path = os.path.join(OUTPUT_DIR, "LSM_Map_Classified.tif")

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
    scai_df.to_csv(os.path.join(OUTPUT_DIR, "SCAI_Analysis.csv"), index=False)

    all_probs, all_uncertainty = predict_probs_and_uncertainty(model, df_raw[features])
    all_classes = np.digitize(all_probs, bins=breaks[1:-1], right=True) + 1

    df_export = df_raw.copy()
    df_export["Predicted_Prob"] = all_probs
    df_export["Predicted_Class"] = all_classes
    df_export["Uncertainty"] = all_uncertainty
    output_path = os.path.join(OUTPUT_DIR, "LSM_Data.csv")
    try:
        df_export.to_csv(output_path, index=False)
    except PermissionError:
        fallback = os.path.join(
            OUTPUT_DIR,
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
    grid_df.to_csv(os.path.join(OUTPUT_DIR, "LSM_Grid_Data.csv"), index=False)


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
    train_size,
    test_size,
):
    report = {
        "seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "cv_repeats": CV_REPEATS,
        "vif_threshold": VIF_THRESHOLD,
        "train_rows": int(train_size),
        "test_rows": int(test_size),
        "selected_features": selected_features,
        "dropped_features_vif": dropped_features,
        "final_model": final_model_name,
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
    with open(
        os.path.join(OUTPUT_DIR, "model_report.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)


def main():
    set_seed()
    ensure_output_dir()

    df, X, y, coords, features = load_dataset(DATA_FILE)

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

    save_correlation_plot(
        X_train[features],
        "Correlation Matrix Before VIF Feature Selection (Train Split)",
        "correlation_before.png",
    )

    selected_features, dropped_features, vif_df = select_features_vif(
        X_train, threshold=VIF_THRESHOLD
    )
    vif_df.to_csv(os.path.join(OUTPUT_DIR, "vif_selected_features.csv"), index=False)

    save_correlation_plot(
        X_train[selected_features],
        "Correlation Matrix After VIF Feature Selection (Train Split)",
        "correlation_after.png",
    )

    model_builders = build_model_builders()

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

    baseline_rows = cv_summary[cv_summary["Model"].isin(["RandomForest", "SVC", "BLR"])]
    best_baseline_name = (
        baseline_rows.iloc[0]["Model"] if not baseline_rows.empty else "RandomForest"
    )
    print(f"Best baseline model (by CV MCC): {best_baseline_name}")

    evaluate_and_visualize_cv(metrics_log, best_baseline_name)
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

    final_model_name, selection_table = choose_final_model(
        cv_summary, holdout_metrics, spatial_metrics
    )
    print(f"\nFinal model selected by multi-criteria ranking: {final_model_name}")
    print(
        selection_table[
            ["Model", "selection_score", "MCC", "Spatial_MCC_Mean", "mcc_mean"]
        ].to_string(index=False)
    )

    explainable_ai(fitted_models[final_model_name], X_train[selected_features])

    print(f"\nTraining final {final_model_name} on all data for map generation...")
    final_model = build_pipeline(build_model_builders()[final_model_name]())
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
        train_size=len(X_train),
        test_size=len(X_test),
    )

    print("\nPipeline execution completed successfully.")


if __name__ == "__main__":
    main()
