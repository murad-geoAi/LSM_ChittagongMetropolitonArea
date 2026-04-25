"""
Create a submission-ready graphical abstract for the Natural Hazards paper.

Outputs:
    results/figures/graphical_abstract_natural_hazards.png
    results/figures/graphical_abstract_natural_hazards.pdf
    results/figures/graphical_abstract_natural_hazards.svg
    results/figures/graphical_abstract_natural_hazards.tiff

The layout distills the methodological figure into a wide graphical abstract:
study area and inputs -> common benchmarking workflow -> Bayesian-only
deployment -> susceptibility and uncertainty outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "results" / "figures"
DATA_DIR = ROOT / "results" / "data"

STUDY_AREA_PATH = ROOT / "Study Area Map" / "Study area map.jpeg"
SUSCEPTIBILITY_PATH = FIGURES_DIR / "susceptibility_map_classified.png"
UNCERTAINTY_PATH = FIGURES_DIR / "uncertainty_map.png"

OUTPUT_STEM = FIGURES_DIR / "graphical_abstract_natural_hazards"


Image.MAX_IMAGE_PIXELS = None


PALETTE = {
    "bg": "#fbfaf7",
    "ink": "#10233f",
    "muted": "#5b6779",
    "border": "#cfd6df",
    "shadow": "#dbe2ea",
    "navy": "#153e75",
    "navy_soft": "#eaf2fb",
    "teal": "#0f766e",
    "teal_soft": "#e6f7f5",
    "amber": "#c2410c",
    "amber_soft": "#fff1e8",
    "violet": "#6d28d9",
    "violet_soft": "#f3e8ff",
    "green": "#2f855a",
    "green_soft": "#eaf7ef",
    "gold": "#9a6700",
    "gold_soft": "#fff7db",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.unicode_minus": False,
    }
)


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_summary_data() -> dict:
    lsm_df = pd.read_csv(DATA_DIR / "LSM_Data.csv")
    selection_df = pd.read_csv(DATA_DIR / "model_selection_table.csv")
    scai_df = pd.read_csv(DATA_DIR / "SCAI_Analysis.csv")

    total_samples = int(len(lsm_df))
    landslide_count = int(lsm_df["Landslide"].sum())
    non_landslide_count = total_samples - landslide_count

    exclude_cols = {
        "Landslide",
        "Longitude",
        "Latitude",
        "Predicted_Prob",
        "Predicted_Class",
        "Uncertainty",
    }
    factor_count = int(sum(col not in exclude_cols for col in lsm_df.columns))

    best_overall = selection_df.sort_values("Overall_Rank").iloc[0]
    deployed = selection_df.loc[selection_df["Deployment_Selected"]].iloc[0]
    very_high = scai_df.loc[scai_df["Tier"] == "Very High"].iloc[0]

    friedman_stats = {}
    friedman_path = DATA_DIR / "friedman_test.txt"
    if friedman_path.exists():
        for line in friedman_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            friedman_stats[key.strip()] = value.strip()

    return {
        "total_samples": total_samples,
        "landslide_count": landslide_count,
        "non_landslide_count": non_landslide_count,
        "factor_count": factor_count,
        "best_overall": best_overall,
        "deployed": deployed,
        "very_high": very_high,
        "friedman": friedman_stats,
    }


def load_image(path: Path, crop: tuple[float, float, float, float]) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        left = int(width * crop[0])
        top = int(height * crop[1])
        right = int(width * crop[2])
        bottom = int(height * crop[3])
        img = img.crop((left, top, right, bottom))

        max_dim = 2200
        scale = min(max_dim / img.size[0], max_dim / img.size[1], 1.0)
        if scale < 1.0:
            resized = (
                max(1, int(img.size[0] * scale)),
                max(1, int(img.size[1] * scale)),
            )
            img = img.resize(resized, Image.Resampling.LANCZOS)
        return np.asarray(img)


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.2,
    radius: float = 0.028,
    shadow: bool = True,
    zorder: int = 1,
) -> None:
    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.006, y - 0.008),
                w,
                h,
                boxstyle=f"round,pad=0.006,rounding_size={radius}",
                linewidth=0,
                facecolor=PALETTE["shadow"],
                alpha=0.35,
                transform=ax.transAxes,
                zorder=zorder,
            )
        )

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.006,rounding_size={radius}",
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            transform=ax.transAxes,
            zorder=zorder + 1,
        )
    )


def add_pill(
    ax,
    x: float,
    y: float,
    text: str,
    facecolor: str,
    textcolor: str,
    fontsize: float = 9,
    padding: float = 0.006,
    radius: float = 0.018,
    edgecolor: str | None = None,
    linewidth: float = 0.8,
    zorder: int = 20,
) -> None:
    char_w = 0.0048
    width = max(0.04, len(text) * char_w + padding * 4)
    height = 0.038

    ax.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle=f"round,pad=0.004,rounding_size={radius}",
            linewidth=linewidth if edgecolor else 0,
            edgecolor=edgecolor if edgecolor else "none",
            facecolor=facecolor,
            transform=ax.transAxes,
            zorder=zorder,
        )
    )
    ax.text(
        x + width / 2,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight="bold",
        zorder=zorder + 1,
    )


def add_image(
    fig, image: np.ndarray, rect: tuple[float, float, float, float], aspect: str = "auto"
) -> None:
    img_ax = fig.add_axes(rect)
    img_ax.imshow(image, aspect=aspect)
    img_ax.axis("off")


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=24,
            linewidth=2.0,
            color=PALETTE["navy"],
            alpha=0.85,
            transform=ax.transAxes,
            zorder=25,
        )
    )


def metric_text(value: float) -> str:
    return f"{value:.3f}"


def render_graphical_abstract() -> None:
    ensure_dirs()
    summary = load_summary_data()

    best_overall = summary["best_overall"]
    deployed = summary["deployed"]
    very_high = summary["very_high"]
    friedman = summary["friedman"]

    study_img = load_image(STUDY_AREA_PATH, crop=(0.39, 0.02, 0.98, 0.98))
    susceptibility_img = load_image(SUSCEPTIBILITY_PATH, crop=(0.04, 0.08, 0.82, 0.95))
    uncertainty_img = load_image(UNCERTAINTY_PATH, crop=(0.04, 0.08, 0.82, 0.95))

    fig = plt.figure(figsize=(14, 8.4), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    add_box(ax, 0.03, 0.87, 0.94, 0.10, "#fdfcf9", PALETTE["border"], radius=0.032)
    ax.text(
        0.05,
        0.935,
        "Bayesian benchmark-to-deployment workflow for landslide susceptibility mapping",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=30,
    )
    ax.text(
        0.05,
        0.895,
        "Chittagong Metropolitan Area, Bangladesh | benchmark all models under one protocol, deploy only the best Bayesian model to preserve uncertainty information",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        color=PALETTE["muted"],
        zorder=30,
    )

    left = (0.03, 0.47, 0.28, 0.35)
    middle = (0.35, 0.47, 0.34, 0.35)
    right = (0.73, 0.47, 0.24, 0.35)
    outputs = (0.03, 0.07, 0.94, 0.33)

    add_box(ax, *left, PALETTE["navy_soft"], "#b8cbea")
    add_box(ax, *middle, PALETTE["amber_soft"], "#f3c4a7")
    add_box(ax, *right, PALETTE["violet_soft"], "#d7b7ff")
    add_box(ax, *outputs, PALETTE["green_soft"], "#b5ddc6")

    add_pill(ax, 0.05, 0.805, "1  Study area and inputs", PALETTE["navy"], "white")
    add_pill(ax, 0.37, 0.805, "2  Benchmark workflow", PALETTE["amber"], "white")
    add_pill(ax, 0.75, 0.805, "3  Bayesian-only deployment", PALETTE["violet"], "white")
    add_pill(ax, 0.05, 0.375, "4  Deployed map products", PALETTE["green"], "white")

    draw_arrow(ax, (0.315, 0.645), (0.35, 0.645))
    draw_arrow(ax, (0.69, 0.645), (0.73, 0.645))
    draw_arrow(ax, (0.85, 0.47), (0.85, 0.405))

    add_image(fig, study_img, (0.055, 0.55, 0.22, 0.185))
    ax.text(
        0.055,
        0.765,
        "Study area: Chittagong Metropolitan Area",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.9,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=30,
    )
    ax.text(
        0.055,
        0.728,
        "369 labelled points | 14 conditioning factors",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        color=PALETTE["muted"],
        zorder=30,
    )
    ax.text(
        0.055,
        0.495,
        "155 landslide and 214 non-landslide samples.\nFactors span terrain, hydrology, climate, land cover,\nand road-river proximity under a stratified 80/20 split.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.1,
        color=PALETTE["muted"],
        linespacing=1.35,
        zorder=30,
    )

    ax.text(
        0.37,
        0.765,
        "All four models use the same\nvalidation framework",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.9,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=30,
    )
    add_pill(ax, 0.37, 0.715, "BLR", "#fde2d2", PALETTE["amber"], edgecolor="#f3c4a7")
    add_pill(ax, 0.43, 0.715, "GPC", "#dce9fb", PALETTE["navy"], edgecolor="#b8cbea")
    add_pill(ax, 0.49, 0.715, "LR-L2", "#fde2d2", PALETTE["amber"], edgecolor="#f3c4a7")
    add_pill(ax, 0.58, 0.715, "RF", "#fde2d2", PALETTE["amber"], edgecolor="#f3c4a7")
    add_pill(ax, 0.37, 0.672, "Repeated CV: 5 x 3", "#fff6ef", PALETTE["amber"], fontsize=8.3, edgecolor="#f3c4a7")
    add_pill(ax, 0.49, 0.672, "Holdout: 20%", "#fff6ef", PALETTE["amber"], fontsize=8.3, edgecolor="#f3c4a7")
    add_pill(ax, 0.59, 0.672, "Spatial block CV", "#fff6ef", PALETTE["amber"], fontsize=8.3, edgecolor="#f3c4a7")
    add_pill(ax, 0.37, 0.63, "Calibration", "#fff6ef", PALETTE["amber"], fontsize=8.3, edgecolor="#f3c4a7")
    add_pill(ax, 0.46, 0.63, "Friedman p = 0.0005", "#fff6ef", PALETTE["amber"], fontsize=8.3, edgecolor="#f3c4a7")

    add_box(ax, 0.37, 0.50, 0.14, 0.10, "#fffaf6", "#f3c4a7", shadow=False, radius=0.022)
    add_box(ax, 0.525, 0.50, 0.14, 0.10, "#f8fbff", "#b8cbea", shadow=False, radius=0.022)
    ax.text(
        0.38,
        0.58,
        "Best overall benchmark",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=PALETTE["amber"],
        fontweight="bold",
        zorder=30,
    )
    ax.text(
        0.38,
        0.545,
        f"{best_overall['Model']}\nMCC {metric_text(best_overall['MCC'])} | ROC-AUC {metric_text(best_overall['ROC_AUC'])}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.3,
        color=PALETTE["ink"],
        linespacing=1.4,
        zorder=30,
    )
    ax.text(
        0.535,
        0.58,
        "Best Bayesian candidate",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=PALETTE["navy"],
        fontweight="bold",
        zorder=30,
    )
    ax.text(
        0.535,
        0.545,
        f"{deployed['Model']}\nMCC {metric_text(deployed['MCC'])} | ECE {metric_text(deployed['ECE'])}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.3,
        color=PALETTE["ink"],
        linespacing=1.4,
        zorder=30,
    )

    ax.text(
        0.75,
        0.765,
        "Weighted ranking across CV,\nholdout, spatial MCC, Brier, and ECE",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.3,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=30,
    )
    add_box(ax, 0.75, 0.64, 0.20, 0.09, "#fcf8ff", "#d7b7ff", shadow=False, radius=0.022)
    ax.text(
        0.765,
        0.695,
        "Decision rule",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.4,
        color=PALETTE["violet"],
        fontweight="bold",
        zorder=30,
    )
    ax.text(
        0.765,
        0.662,
        "Final map export is restricted to\nBayesian models only.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.0,
        color=PALETTE["ink"],
        linespacing=1.3,
        zorder=30,
    )
    add_box(ax, 0.75, 0.515, 0.20, 0.095, "#ffffff", "#d7b7ff", shadow=False, radius=0.022)
    ax.text(
        0.765,
        0.575,
        "Selected for deployment",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.4,
        color=PALETTE["violet"],
        fontweight="bold",
        zorder=30,
    )
    ax.text(
        0.765,
        0.545,
        f"{deployed['Model']} | Overall rank #{int(deployed['Overall_Rank'])}\nChosen to retain model-based uncertainty.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.2,
        color=PALETTE["ink"],
        linespacing=1.35,
        zorder=30,
    )

    ax.text(
        0.05,
        0.345,
        f"Deployed {deployed['Model']} map: Very High susceptibility covers {very_high['Area_Pct']:.2f}% of the area, captures {very_high['Landslide_Pct']:.2f}% of landslides, and yields SCAI = {very_high['SCAI']:.2f}.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12.2,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=30,
    )

    add_box(ax, 0.05, 0.12, 0.22, 0.19, "#f7fff9", "#b5ddc6", shadow=False, radius=0.022)
    ax.text(
        0.065,
        0.285,
        "Hazard-screening message",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["green"],
        zorder=30,
    )
    ax.text(
        0.065,
        0.245,
        "1. Benchmarking identifies RF as the\n   strongest discriminator.\n2. Bayesian policy deploys GPC for\n   susceptibility plus uncertainty mapping.\n3. High-risk zones are concentrated into a\n   limited spatial footprint.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.7,
        color=PALETTE["muted"],
        linespacing=1.45,
        zorder=30,
    )

    add_box(ax, 0.31, 0.12, 0.30, 0.19, "#ffffff", "#b5ddc6", shadow=False, radius=0.022)
    add_box(ax, 0.64, 0.12, 0.30, 0.19, "#ffffff", "#b5ddc6", shadow=False, radius=0.022)
    ax.text(
        0.325,
        0.285,
        "Deployed susceptibility map",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=30,
    )
    ax.text(
        0.655,
        0.285,
        "Associated uncertainty map",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["ink"],
        zorder=30,
    )
    add_image(fig, susceptibility_img, (0.325, 0.135, 0.27, 0.155), aspect="auto")
    add_image(fig, uncertainty_img, (0.655, 0.135, 0.27, 0.155), aspect="auto")
    add_pill(ax, 0.325, 0.155, "Five susceptibility tiers", PALETTE["gold_soft"], PALETTE["gold"], fontsize=8.2, edgecolor="#e6cd89")
    add_pill(ax, 0.655, 0.155, "Latent variance proxy", PALETTE["gold_soft"], PALETTE["gold"], fontsize=8.2, edgecolor="#e6cd89")

    if friedman:
        ax.text(
            0.37,
            0.475,
            f"Friedman statistic = {friedman.get('statistic', 'NA')} | significant performance differences among models",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color=PALETTE["muted"],
            zorder=30,
        )

    for suffix in (".png", ".pdf", ".svg", ".tiff"):
        save_kwargs = {"dpi": 300, "facecolor": PALETTE["bg"]}
        if suffix == ".tiff":
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUTPUT_STEM.with_suffix(suffix), **save_kwargs)

    plt.close(fig)


if __name__ == "__main__":
    render_graphical_abstract()
