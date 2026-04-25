import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _ensure_dirs():
    os.makedirs("results/figures", exist_ok=True)


def generate_cv_summary():
    cv_df = pd.read_csv("results/data/cv_metrics_summary.csv")
    sp_df = pd.read_csv("results/data/spatial_cv_metrics.csv")

    df = pd.merge(cv_df, sp_df, on=["Model", "Model_Role"])

    # Sort models: BLR, GPC, LogReg_L2, RandomForest
    preferred_order = ["BLR", "GPC", "LogReg_L2", "RandomForest"]
    df["Model_Cat"] = pd.Categorical(
        df["Model"], categories=preferred_order, ordered=True
    )
    df = df.sort_values("Model_Cat").reset_index(drop=True)

    models = df["Model"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(models))
    width = 0.35

    # Plot AUC
    axes[0].bar(
        x - width / 2,
        df["roc_auc_mean"],
        width,
        yerr=df["roc_auc_std"],
        label="Repeated CV",
        capsize=5,
        color="#1f77b4",
    )
    axes[0].bar(
        x + width / 2,
        df["Spatial_ROC_AUC_Mean"],
        width,
        yerr=df["Spatial_ROC_AUC_Std"],
        label="Spatial block CV",
        capsize=5,
        color="#ff7f0e",
    )
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Repeated CV vs. Spatial block CV: ROC-AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Plot MCC
    axes[1].bar(
        x - width / 2,
        df["mcc_mean"],
        width,
        yerr=df["mcc_std"],
        label="Repeated CV",
        capsize=5,
        color="#1f77b4",
    )
    axes[1].bar(
        x + width / 2,
        df["Spatial_MCC_Mean"],
        width,
        yerr=df["Spatial_MCC_Std"],
        label="Spatial block CV",
        capsize=5,
        color="#ff7f0e",
    )
    axes[1].set_ylabel("MCC")
    axes[1].set_title("Repeated CV vs. Spatial block CV: MCC")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Repeated CV and spatial block CV summary", fontsize=16)
    plt.tight_layout()
    out_path = "results/figures/cv_summary_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


def generate_scai_plot():
    df = pd.read_csv("results/data/SCAI_Analysis.csv")
    df = df.dropna(subset=["Tier"])

    # Ensure tier order
    tier_order = ["Very Low", "Low", "Moderate", "High", "Very High"]
    df["Tier_Cat"] = pd.Categorical(df["Tier"], categories=tier_order, ordered=True)
    df = df.sort_values("Tier_Cat").reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df["Tier"]))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        df["Area_Pct"],
        width,
        label="Area (%)",
        color="#2ca02c",
        alpha=0.7,
    )
    bars2 = ax1.bar(
        x + width / 2,
        df["Landslide_Pct"],
        width,
        label="Landslide (%)",
        color="#d62728",
        alpha=0.7,
    )

    ax1.set_xlabel("Susceptibility Tier", fontsize=12)
    ax1.set_ylabel("Percentage (%)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Tier"], fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    line1 = ax2.plot(
        x,
        df["SCAI"],
        color="black",
        marker="o",
        linewidth=2,
        markersize=8,
        label="SCAI",
    )
    if "SCAI_Smoothed" in df.columns:
        line2 = ax2.plot(
            x,
            df["SCAI_Smoothed"],
            color="blue",
            linestyle="--",
            marker="s",
            linewidth=2,
            markersize=8,
            label="SCAI (Smoothed)",
        )
    else:
        line2 = []

    ax2.set_ylabel("SCAI Value", fontsize=12)
    ax2.set_ylim(bottom=0)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines + lines2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        fontsize=11,
    )

    plt.title("SCAI by susceptibility tier", fontsize=14)
    plt.tight_layout()

    out_path = "results/figures/scai_by_tier.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    _ensure_dirs()
    generate_cv_summary()
    generate_scai_plot()
