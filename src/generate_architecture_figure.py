"""
Generate a publication-grade model architecture infographic using matplotlib.
Output: results/figures/model_architecture.png (300 DPI, high-resolution)
        results/figures/model_architecture.pdf (vector format)
        results/figures/model_architecture.svg (vector format)

This script produces a clean, print-ready figure suitable for Q1 journal papers.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Configuration & Color Palette
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 8,
        "axes.unicode_minus": False,
        "mathtext.default": "regular",
    }
)

# Colors
C = {
    "bg": "#ffffff",
    "text_dark": "#1a1a2e",
    "text_mid": "#374151",
    "text_light": "#6b7280",
    "arrow": "#64748b",
    # Stage backgrounds & borders
    "s1_bg": "#eff6ff",
    "s1_bd": "#93c5fd",
    "s1_badge": "#3b82f6",
    "s2_bg": "#f0fdf4",
    "s2_bd": "#86efac",
    "s2_badge": "#16a34a",
    "s3_bg": "#f8f8fc",
    "s3_bd": "#c7d2fe",
    "s3_badge": "#6366f1",
    "s4_bg": "#fffbeb",
    "s4_bd": "#fcd34d",
    "s4_badge": "#b45309",
    "s5_bg": "#f5f3ff",
    "s5_bd": "#c4b5fd",
    "s5_badge": "#7c3aed",
    "s6_bg": "#f8fafc",
    "s6_bd": "#94a3b8",
    "s6_badge": "#334155",
    # Model card colors
    "blr_bg": "#fff7ed",
    "blr_bd": "#fdba74",
    "blr_acc": "#c2410c",
    "gpc_bg": "#eff6ff",
    "gpc_bd": "#93c5fd",
    "gpc_acc": "#1d4ed8",
    "rf_bg": "#f0fdf4",
    "rf_bd": "#86efac",
    "rf_acc": "#15803d",
    "svc_bg": "#fef2f2",
    "svc_bd": "#fca5a5",
    "svc_acc": "#b91c1c",
}


def draw_box(ax, x, y, w, h, bg, bd, radius=0.12, lw=1.2, zorder=1):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg,
        edgecolor=bd,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(box)


def draw_arrow(ax, x, y_start, y_end, color=None, lw=1.4):
    if color is None:
        color = C["arrow"]
    ax.annotate(
        "",
        xy=(x, y_end),
        xytext=(x, y_start),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
        zorder=5,
    )


def draw_badge(ax, x, y, text, color, w=0.65, h=0.17, fontsize=6.2):
    badge = FancyBboxPatch(
        (x, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=color,
        edgecolor="none",
        zorder=10,
    )
    ax.add_patch(badge)
    ax.text(
        x + w / 2,
        y,
        text,
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=11,
        fontfamily="sans-serif",
    )


def icon_circle(ax, cx, cy, r, bg_color, text, text_color="white", fontsize=8):
    """Draw a small icon circle with a letter/symbol inside."""
    circle = plt.Circle((cx, cy), r, facecolor=bg_color, edgecolor="none", zorder=8)
    ax.add_patch(circle)
    ax.text(
        cx,
        cy,
        text,
        fontsize=fontsize,
        fontweight="bold",
        color=text_color,
        ha="center",
        va="center",
        zorder=9,
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Create Figure
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig, ax = plt.subplots(1, 1, figsize=(10.5, 14.8))
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 14.8)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C["bg"])

LM = 0.3
W = 9.9
CX = LM + W / 2

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Title
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ax.text(
    CX,
    14.5,
    "Fig. 1.",
    fontsize=11,
    fontweight="bold",
    color=C["text_dark"],
    ha="center",
    va="center",
)
ax.text(
    CX,
    14.18,
    "Proposed Bayesian Machine Learning Framework for\n"
    "Landslide Susceptibility Mapping (LSM) â€” Chittagong Metropolitan Area",
    fontsize=9,
    fontweight="normal",
    color=C["text_dark"],
    ha="center",
    va="center",
    linespacing=1.4,
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STAGE 1: Data Input
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
s1_y = 12.65
s1_h = 1.2
draw_box(ax, LM, s1_y, W, s1_h, C["s1_bg"], C["s1_bd"])
draw_badge(ax, LM + 0.12, s1_y + s1_h - 0.16, "STAGE 1", C["s1_badge"])
ax.text(
    LM + 0.92,
    s1_y + s1_h - 0.16,
    "Geospatial Dataset Acquisition",
    fontsize=9,
    fontweight="bold",
    color=C["text_dark"],
    va="center",
)

# Feature tags
features = [
    "Elevation",
    "Slope",
    "Aspect",
    "Curvature",
    "SPI",
    "TWI",
    "TRI",
    "LULC",
    "NDVI",
    "Precipitation",
    "Drainage Density",
    "Dist. to River",
    "Dist. to Road",
    "Soil Texture",
]

ax.text(
    LM + 0.2,
    s1_y + s1_h - 0.42,
    "14 Conditioning Factors:",
    fontsize=7,
    fontweight="bold",
    color=C["text_mid"],
    va="center",
)

tag_x = LM + 0.2
tag_y = s1_y + s1_h - 0.62
tag_h = 0.18
gap = 0.05

for feat in features:
    tag_w = len(feat) * 0.072 + 0.18
    if tag_x + tag_w > LM + W - 2.0:
        tag_x = LM + 0.2
        tag_y -= tag_h + gap + 0.04
    box = FancyBboxPatch(
        (tag_x, tag_y - tag_h / 2),
        tag_w,
        tag_h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor="#dbeafe",
        edgecolor="#93c5fd",
        linewidth=0.5,
        zorder=6,
    )
    ax.add_patch(box)
    ax.text(
        tag_x + tag_w / 2,
        tag_y,
        feat,
        fontsize=5.8,
        fontweight="medium",
        color="#1e40af",
        ha="center",
        va="center",
        zorder=7,
        family="monospace",
    )
    tag_x += tag_w + gap

# Right side info
ax.text(
    LM + W - 0.15,
    s1_y + s1_h - 0.42,
    "Target: Landslide (0 / 1)",
    fontsize=6.8,
    color=C["text_mid"],
    ha="right",
    va="center",
    fontweight="semibold",
)
ax.text(
    LM + W - 0.15,
    s1_y + s1_h - 0.6,
    "Coordinates: Latitude, Longitude",
    fontsize=6.5,
    color=C["text_light"],
    ha="right",
    va="center",
)

draw_arrow(ax, CX, s1_y - 0.02, s1_y - 0.3)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STAGE 2: Preprocessing
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
s2_y = 11.1
s2_h = 1.22
draw_box(ax, LM, s2_y, W, s2_h, C["s2_bg"], C["s2_bd"])
draw_badge(ax, LM + 0.12, s2_y + s2_h - 0.16, "STAGE 2", C["s2_badge"])
ax.text(
    LM + 0.92,
    s2_y + s2_h - 0.16,
    "Data Preprocessing & Feature Engineering",
    fontsize=9,
    fontweight="bold",
    color=C["text_dark"],
    va="center",
)

prep_items = [
    ("a", "VIF Feature\nSelection", "threshold = 5.0"),
    ("b", "Median\nImputation", "SimpleImputer"),
    ("c", "Z-Score\nNormalization", "StandardScaler"),
    ("d", "Stratified\nTrain/Test Split", "80% / 20%"),
]
colors_prep = ["#059669", "#0d9488", "#0891b2", "#0284c7"]
prep_w = 2.18
prep_gap = 0.2
prep_start_x = LM + (W - 4 * prep_w - 3 * prep_gap) / 2
prep_y = s2_y + 0.1
prep_h = 0.78

for i, (letter, name, detail) in enumerate(prep_items):
    px = prep_start_x + i * (prep_w + prep_gap)
    draw_box(
        ax,
        px,
        prep_y,
        prep_w,
        prep_h,
        "#ffffff",
        "#bbf7d0",
        radius=0.08,
        lw=0.7,
        zorder=3,
    )
    # Icon circle
    icon_circle(
        ax,
        px + prep_w / 2,
        prep_y + prep_h - 0.15,
        0.1,
        colors_prep[i],
        letter,
        fontsize=7,
    )
    ax.text(
        px + prep_w / 2,
        prep_y + prep_h / 2 - 0.06,
        name,
        fontsize=7,
        fontweight="bold",
        color="#166534",
        ha="center",
        va="center",
        zorder=4,
        linespacing=1.2,
    )
    ax.text(
        px + prep_w / 2,
        prep_y + 0.1,
        detail,
        fontsize=6,
        color=C["text_light"],
        ha="center",
        va="center",
        zorder=4,
        family="monospace",
    )
    if i < 3:
        ax.annotate(
            "",
            xy=(px + prep_w + prep_gap * 0.2, prep_y + prep_h / 2),
            xytext=(px + prep_w + 0.01, prep_y + prep_h / 2),
            arrowprops=dict(arrowstyle="->", color="#86efac", lw=1.0),
            zorder=5,
        )

draw_arrow(ax, CX, s2_y - 0.02, s2_y - 0.3)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STAGE 3: Models
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
s3_y = 8.3
s3_h = 2.48
draw_box(ax, LM, s3_y, W, s3_h, C["s3_bg"], C["s3_bd"])
draw_badge(ax, LM + 0.12, s3_y + s3_h - 0.16, "STAGE 3", C["s3_badge"])
ax.text(
    LM + 0.92,
    s3_y + s3_h - 0.16,
    "Bayesian Deployment + Benchmark Models",
    fontsize=9,
    fontweight="bold",
    color=C["text_dark"],
    va="center",
)

# Pipeline label
ax.text(
    CX,
    s3_y + s3_h - 0.45,
    "Each model wrapped in:  Pipeline  â†’  Imputer  â†’  Scaler  â†’  Classifier",
    fontsize=6.5,
    color=C["text_light"],
    ha="center",
    va="center",
    family="monospace",
    bbox=dict(
        boxstyle="round,pad=0.15", facecolor="#f1f5f9", edgecolor="#e2e8f0", lw=0.5
    ),
)

model_cards = [
    {
        "abbr": "BLR",
        "name": "Bayesian Logistic\nRegression",
        "type": "BAYESIAN",
        "bg": C["blr_bg"],
        "bd": C["blr_bd"],
        "acc": C["blr_acc"],
        "details": [
            "Laplace Approximation",
            "Prior Precision = 1.5",
            "Posterior Sampling (500)",
            "L-BFGS-B Optimizer",
        ],
    },
    {
        "abbr": "GPC",
        "name": "Gaussian Process\nClassifier",
        "type": "BAYESIAN",
        "bg": C["gpc_bg"],
        "bd": C["gpc_bd"],
        "acc": C["gpc_acc"],
        "details": [
            "Matern (nu = 3/2) Kernel",
            "+ White Kernel",
            "n_restarts = 5",
            "Full Bayesian Inference",
        ],
    },
    {
        "abbr": "LR-L2",
        "name": "L2-Regularized\nLogistic Regression",
        "type": "BENCHMARK",
        "bg": C["svc_bg"],
        "bd": C["svc_bd"],
        "acc": C["svc_acc"],
        "details": [
            "Penalty = L2 (C = 1.0)",
            "Class Weight = balanced",
            "Solver = lbfgs",
            "max_iter = 3000",
        ],
    },
    {
        "abbr": "RF",
        "name": "Random\nForest",
        "type": "BENCHMARK",
        "bg": C["rf_bg"],
        "bd": C["rf_bd"],
        "acc": C["rf_acc"],
        "details": [
            "n_estimators = 300",
            "min_samples_leaf = 2",
            "balanced_subsample",
            "n_jobs = -1",
        ],
    },
]

mc_w = 2.18
mc_gap = 0.2
mc_start_x = LM + (W - len(model_cards) * mc_w - (len(model_cards) - 1) * mc_gap) / 2
mc_y = s3_y + 0.1
mc_h = 1.68

for i, mc in enumerate(model_cards):
    mx = mc_start_x + i * (mc_w + mc_gap)
    draw_box(ax, mx, mc_y, mc_w, mc_h, mc["bg"], mc["bd"], radius=0.1, zorder=3)

    # Type badge
    bw = 0.65
    bh = 0.14
    bx = mx + (mc_w - bw) / 2
    by = mc_y + mc_h - 0.22
    badge = FancyBboxPatch(
        (bx, by),
        bw,
        bh,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=mc["acc"],
        edgecolor="none",
        zorder=8,
    )
    ax.add_patch(badge)
    ax.text(
        bx + bw / 2,
        by + bh / 2,
        mc["type"],
        fontsize=5,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=9,
    )

    # Abbreviation
    ax.text(
        mx + mc_w / 2,
        mc_y + mc_h - 0.55,
        mc["abbr"],
        fontsize=16,
        fontweight="bold",
        color=mc["acc"],
        ha="center",
        va="center",
        zorder=4,
    )

    # Full name
    ax.text(
        mx + mc_w / 2,
        mc_y + mc_h - 0.88,
        mc["name"],
        fontsize=6.8,
        fontweight="bold",
        color=C["text_dark"],
        ha="center",
        va="center",
        zorder=4,
        linespacing=1.2,
    )

    # Details
    for j, detail in enumerate(mc["details"]):
        ax.text(
            mx + mc_w / 2,
            mc_y + 0.52 - j * 0.13,
            detail,
            fontsize=5.5,
            color=C["text_light"],
            ha="center",
            va="center",
            zorder=4,
        )

draw_arrow(ax, CX, s3_y - 0.02, s3_y - 0.3)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STAGE 4: Validation
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
s4_y = 5.78
s4_h = 2.2
draw_box(ax, LM, s4_y, W, s4_h, C["s4_bg"], C["s4_bd"])
draw_badge(ax, LM + 0.12, s4_y + s4_h - 0.16, "STAGE 4", C["s4_badge"])
ax.text(
    LM + 0.92,
    s4_y + s4_h - 0.16,
    "Rigorous Multi-Strategy Validation",
    fontsize=9,
    fontweight="bold",
    color=C["text_dark"],
    va="center",
)

val_items = [
    ("i", "Repeated Stratified\nK-Fold CV", "5 folds Ã— 3 repeats"),
    ("ii", "Spatial Block\nCross-Validation", "KMeans grouping"),
    ("iii", "Independent\nHoldout Test", "20% test split"),
    ("iv", "Bootstrap\nConfidence Intervals", "1,200 iterations"),
]
val_colors = ["#b45309", "#a16207", "#92400e", "#78350f"]
vi_w = 2.18
vi_gap = 0.2
vi_start_x = LM + (W - 4 * vi_w - 3 * vi_gap) / 2
vi_y = s4_y + 0.65
vi_h = 0.88

for i, (num, name, detail) in enumerate(val_items):
    vx = vi_start_x + i * (vi_w + vi_gap)
    draw_box(
        ax, vx, vi_y, vi_w, vi_h, "#ffffff", "#fde68a", radius=0.08, lw=0.7, zorder=3
    )
    # Circled number
    icon_circle(
        ax, vx + vi_w / 2, vi_y + vi_h - 0.15, 0.1, val_colors[i], num, fontsize=6
    )
    ax.text(
        vx + vi_w / 2,
        vi_y + vi_h / 2 - 0.04,
        name,
        fontsize=6.8,
        fontweight="bold",
        color="#854d0e",
        ha="center",
        va="center",
        zorder=4,
        linespacing=1.2,
    )
    ax.text(
        vx + vi_w / 2,
        vi_y + 0.1,
        detail,
        fontsize=5.8,
        color=C["text_light"],
        ha="center",
        va="center",
        zorder=4,
        family="monospace",
    )

# Metrics row label
ax.text(
    LM + 0.35,
    s4_y + 0.38,
    "Evaluation Metrics:",
    fontsize=6.5,
    fontweight="bold",
    color="#713f12",
    va="center",
)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "ROC-AUC",
    "PR-AUC",
    "Brier",
    "LogLoss",
    "ECE",
]
chip_y = s4_y + 0.22
chip_x = LM + 0.35
for m in metrics:
    cw = len(m) * 0.06 + 0.22
    ch = 0.18
    box = FancyBboxPatch(
        (chip_x, chip_y - ch / 2),
        cw,
        ch,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#fef9c3",
        edgecolor="#fcd34d",
        linewidth=0.5,
        zorder=6,
    )
    ax.add_patch(box)
    ax.text(
        chip_x + cw / 2,
        chip_y,
        m,
        fontsize=5.2,
        fontweight="medium",
        color="#713f12",
        ha="center",
        va="center",
        zorder=7,
        family="monospace",
    )
    chip_x += cw + 0.06

draw_arrow(ax, CX, s4_y - 0.02, s4_y - 0.3)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STAGE 5: Model Selection
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
s5_y = 4.38
s5_h = 1.08
draw_box(ax, LM, s5_y, W, s5_h, C["s5_bg"], C["s5_bd"])
draw_badge(ax, LM + 0.12, s5_y + s5_h - 0.16, "STAGE 5", C["s5_badge"])
ax.text(
    LM + 0.92,
    s5_y + s5_h - 0.16,
    "Multi-Criteria Rank-Based Model Selection",
    fontsize=9,
    fontweight="bold",
    color=C["text_dark"],
    va="center",
)

# Weighted formula
ax.text(
    CX,
    s5_y + s5_h - 0.45,
    "Weighted composite score  =",
    fontsize=7,
    fontweight="semibold",
    color=C["text_mid"],
    ha="center",
    va="center",
)

terms = [
    ("20%", "CV MCC"),
    ("35%", "Holdout MCC"),
    ("30%", "Spatial MCC"),
    ("10%", "Brier"),
    ("5%", "ECE"),
]

term_w = 1.55
term_gap = 0.28
total_tw = len(terms) * term_w + (len(terms) - 1) * term_gap
tx = CX - total_tw / 2
term_y = s5_y + 0.26

for i, (weight, metric) in enumerate(terms):
    fw = term_w
    fh = 0.3
    box = FancyBboxPatch(
        (tx, term_y - fh / 2),
        fw,
        fh,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor="#ffffff",
        edgecolor="#c4b5fd",
        linewidth=0.7,
        zorder=6,
    )
    ax.add_patch(box)
    ax.text(
        tx + 0.3,
        term_y,
        weight,
        fontsize=7.5,
        fontweight="bold",
        color="#7c3aed",
        ha="center",
        va="center",
        zorder=7,
        family="monospace",
    )
    ax.text(
        tx + fw / 2 + 0.13,
        term_y,
        metric,
        fontsize=6.5,
        fontweight="medium",
        color=C["text_mid"],
        ha="center",
        va="center",
        zorder=7,
    )
    tx += fw
    if i < len(terms) - 1:
        ax.text(
            tx + term_gap / 2,
            term_y,
            "+",
            fontsize=11,
            color="#d1d5db",
            ha="center",
            va="center",
            zorder=5,
            fontweight="light",
        )
        tx += term_gap

draw_arrow(ax, CX, s5_y - 0.02, s5_y - 0.3)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  STAGE 6: Outputs
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
s6_y = 2.78
s6_h = 1.28
draw_box(ax, LM, s6_y, W, s6_h, C["s6_bg"], C["s6_bd"])
draw_badge(ax, LM + 0.12, s6_y + s6_h - 0.16, "STAGE 6", C["s6_badge"])
ax.text(
    LM + 0.92,
    s6_y + s6_h - 0.16,
    "Final Outputs & Deliverables",
    fontsize=9,
    fontweight="bold",
    color=C["text_dark"],
    va="center",
)

outputs = [
    (
        "S",
        "#6366f1",
        "Explainable AI\n(SHAP Analysis)",
        "Feature importance &\ndependence plots",
    ),
    (
        "M",
        "#0891b2",
        "Susceptibility &\nUncertainty Maps",
        "Pixel-wise GeoTIFF\nrasters & classified zones",
    ),
    (
        "V",
        "#b45309",
        "SCAI & Statistical\nValidation",
        "Friedman/Nemenyi +\nBayesian deployment rule",
    ),
]
ow = 3.0
o_gap = 0.2
o_start_x = LM + (W - 3 * ow - 2 * o_gap) / 2
o_y = s6_y + 0.08
o_h = 0.88

for i, (letter, lcolor, name, detail) in enumerate(outputs):
    ox = o_start_x + i * (ow + o_gap)
    draw_box(ax, ox, o_y, ow, o_h, "#ffffff", "#e2e8f0", radius=0.08, lw=0.7, zorder=3)
    icon_circle(ax, ox + ow / 2, o_y + o_h - 0.16, 0.12, lcolor, letter, fontsize=8)
    ax.text(
        ox + ow / 2,
        o_y + o_h / 2 - 0.06,
        name,
        fontsize=7,
        fontweight="bold",
        color=C["text_dark"],
        ha="center",
        va="center",
        zorder=4,
        linespacing=1.2,
    )
    ax.text(
        ox + ow / 2,
        o_y + 0.14,
        detail,
        fontsize=5.5,
        color=C["text_light"],
        ha="center",
        va="center",
        zorder=4,
        linespacing=1.25,
    )

# Footer
ax.text(
    CX,
    2.4,
    "Proposed methodological framework for Bayesian landslide susceptibility mapping.\n"
    "All models share identical preprocessing; only Bayesian candidates are eligible for final deployment maps.",
    fontsize=6.2,
    color=C["text_light"],
    ha="center",
    va="center",
    fontstyle="italic",
    linespacing=1.35,
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  Save
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "figures"
)
os.makedirs(out_dir, exist_ok=True)

# High-res PNG
out_png = os.path.join(out_dir, "model_architecture.png")
fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.2, facecolor="white")
print(f"Saved: {out_png}")

# Vector PDF
out_pdf = os.path.join(out_dir, "model_architecture.pdf")
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2, facecolor="white")
print(f"Saved: {out_pdf}")

# Vector SVG
out_svg = os.path.join(out_dir, "model_architecture.svg")
fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.2, facecolor="white")
print(f"Saved: {out_svg}")

plt.close()
print(
    "\nPublication-grade model architecture figure generated in PNG (300 DPI), PDF, and SVG."
)

