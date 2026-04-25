"""
Generate additional publication-grade figures from existing outputs.

Outputs
-------
- results/figures/radar_holdout_metrics.png
- results/figures/shap_waterfall_example.png

Run (from repo root):
    python src/generate_additional_figures.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = "results"
DATA_DIR = os.path.join(RESULTS_DIR, "data")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

MODEL_REPORT_PATH = os.path.join(DATA_DIR, "model_report.json")
DATASET_PATH = os.path.join(DATA_DIR, "LSM_Data.csv")

TARGET_COL = "Landslide"


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


def _build_gpc_pipeline(seed: int) -> Pipeline:
    kernel = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-5)
    clf = GaussianProcessClassifier(
        kernel=kernel,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=5,
        random_state=seed,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def generate_radar_holdout_metrics(report: dict[str, Any]) -> str:
    """
    Radar plot comparing holdout metrics across all benchmarked models.

    Notes:
    - Calibration metrics are inverted (1 - value), so higher is better everywhere.
    """

    holdout_rows = report.get("holdout_metrics", [])
    if not holdout_rows:
        raise RuntimeError("model_report.json is missing 'holdout_metrics'.")

    metrics = ["Accuracy", "F1", "MCC", "ROC_AUC", "PR_AUC", "Brier", "ECE"]
    axis_labels = ["Accuracy", "F1", "MCC", "ROC-AUC", "PR-AUC", "1−Brier", "1−ECE"]

    values_by_model: dict[str, list[float]] = {}
    for row in holdout_rows:
        model_name = str(row["Model"])
        vals: list[float] = []
        for m in metrics:
            v = float(row[m])
            if m in {"Brier", "ECE"}:
                v = 1.0 - v
            vals.append(float(np.clip(v, 0.0, 1.0)))
        values_by_model[model_name] = vals

    preferred_order = ["BLR", "GPC", "LogReg_L2", "RandomForest"]
    model_order = [m for m in preferred_order if m in values_by_model] + [
        m for m in values_by_model.keys() if m not in preferred_order
    ]

    colors = {
        "RandomForest": "#1f77b4",
        "GPC": "#ff7f0e",
        "LogReg_L2": "#2ca02c",
        "BLR": "#d62728",
    }

    n_axes = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), axis_labels)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, color="#d1d5db", linestyle="--", linewidth=0.7)
    ax.xaxis.grid(True, color="#d1d5db", linestyle="--", linewidth=0.7)

    for model_name in model_order:
        vals = values_by_model[model_name]
        series = vals + vals[:1]
        color = colors.get(model_name, None)
        ax.plot(angles, series, linewidth=2.0, label=model_name, color=color)
        ax.fill(angles, series, alpha=0.12, color=color)

    ax.set_title("Holdout Performance Radar (higher is better)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.36, 1.15), frameon=False)

    out_path = os.path.join(FIG_DIR, "radar_holdout_metrics.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_path}")
    return out_path


def generate_shap_waterfall_example(report: dict[str, Any]) -> str:
    """
    SHAP waterfall plot for a representative holdout sample using the final deployment model.

    The sample is selected from the holdout test subset as the highest predicted probability
    among true landslide samples (fallback: overall highest probability).
    """

    seed = int(report.get("seed", 42))
    test_size = float(report.get("test_size", 0.2))
    selected_features = report.get("selected_features", [])
    final_model_name = str(report.get("final_model", "GPC"))
    if not selected_features:
        raise RuntimeError("model_report.json is missing 'selected_features'.")
    if final_model_name != "GPC":
        print(
            f"Warning: final_model in report is '{final_model_name}'. "
            "This script currently generates SHAP waterfall for GPC only."
        )

    df = pd.read_csv(DATASET_PATH)
    X = df[selected_features].copy()
    y = df[TARGET_COL].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    model = _build_gpc_pipeline(seed=seed)
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    candidate_idx = np.where(y_test == 1)[0]
    if len(candidate_idx) > 0:
        idx = int(candidate_idx[np.argmax(test_probs[candidate_idx])])
    else:
        idx = int(np.argmax(test_probs))

    x_instance = X_test.iloc[[idx]]

    background_n = min(30, len(X_train))
    background_df = shap.sample(X_train, background_n, random_state=seed)

    feature_cols = X_train.columns.tolist()

    def predict_proba_wrapper(data: np.ndarray) -> np.ndarray:
        data_df = pd.DataFrame(data, columns=feature_cols)
        return model.predict_proba(data_df)

    explainer = shap.KernelExplainer(
        predict_proba_wrapper, background_df.to_numpy(), link="logit"
    )
    shap_values = explainer.shap_values(x_instance.to_numpy(), nsamples=150)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1][0], dtype=float)
        base_values = float(np.asarray(explainer.expected_value)[1])
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        values = np.asarray(shap_values[0, :, 1], dtype=float)
        base_values = float(np.asarray(explainer.expected_value)[1])
    else:
        values = np.asarray(shap_values[0], dtype=float)
        base_values = float(np.asarray(explainer.expected_value))

    explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=x_instance.iloc[0].to_numpy(dtype=float),
        feature_names=feature_cols,
    )

    shap.plots.waterfall(explanation, show=False, max_display=min(14, len(feature_cols)))
    plt.title("SHAP Waterfall (Representative Holdout Sample, GPC)")
    out_path = os.path.join(FIG_DIR, "shap_waterfall_example.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")
    return out_path


def main() -> None:
    _ensure_dirs()
    report = _load_json(MODEL_REPORT_PATH)
    generate_radar_holdout_metrics(report)
    generate_shap_waterfall_example(report)


if __name__ == "__main__":
    main()
