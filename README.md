# Landslide Susceptibility Mapping (Bayesian LSM Pipeline)

Welcome to the project repository for the Q1 Journal-grade Bayesian Landslide Susceptibility Modeling (LSM) pipeline. This repository has been restructured to facilitate reproducibility, isolated virtual environments, and modularity.

## Repository Structure Overview

```text
LSM_ChittagongMetropolitonArea/
│
├── data/                    # Contains Raw and Processed Datasets
│   └── Landslide_Data.csv   # The primary structured dataset
│
├── src/                     # Source codes and notebooks
│   └── (Your Scripts Here)  # Place all execution pipelines or model architecture scripts here (e.g., bayesian_lsm_pipeline.py)
│
├── results/                 # Automated output files and pipeline artifacts
│   ├── figures/             # Q1 Journal ready visualizations (.png, .jpg)
│   └── data/                # Processed outputs, geospatial raster/grids (.tif, .csv)
│
├── docs/                    # Related project documentation and write-ups
│   └── DEMO_RESEARCH_PAPER.md
│
├── requirements.txt         # Dependency file for python isolated environment
└── README.md                # This user guide
```

---

## 🛠️ Environment Setup

To ensure strict reproducibility and avoid package conflicts with your other software, you must create an isolated virtual environment containing all required libraries.

1. **Navigate to the Project Directory**:

   ```bash
   cd "h:\Baysesian Machine Learning-LSM\LSM_ChittagongMetropolitonArea"
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv lsm_env
   ```

3. **Activate the Environment**:
   - On Windows:
     ```bash
     .\lsm_env\Scripts\activate
     ```

4. **Install the Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Figure & Asset Guide for Q1 Journal Tasks

Using the outputs found in `results/`, here is a guide linking the exact figures to the relevant sections of your research paper:

### 1. Feature Engineering & Multicollinearity Checking

**Objective**: Demonstrate that continuous features do not severely intercorrelate or bias the algorithms.

- `results/figures/correlation_before.png`: Pearson correlation heatmap before Variance Inflation Factor (VIF) selection.
- `results/figures/correlation_after.png`: Purified heatmap after excluding highly collinear variables.
- `results/data/vif_selected_features.csv`: VIF score tabular report for the methodology section.

### 2. Model Performance Evaluation

**Objective**: Show that the Bayesian implementation (e.g., Gaussian Process Classifier or Bayesian Logistic Regression) reliably detects landslides globally and spatially.

- `results/figures/roc_curve.png`: Test dataset ROC curves demonstrating Sensitivity vs Specificity.
- `results/figures/pr_curve.png`: Precision-Recall evaluating class imbalance robustness.
- `results/figures/calibration_curve_holdout.png`: Brier score reliability mapping demonstrating the Bayesian method’s uncertainty precision versus baselines.
- `results/figures/confusion_matrix_gpc.png`: Matrix showing True/False predictions.
- `results/data/cv_metrics_summary.csv` & `spatial_cv_metrics.csv`: Aggregate statistics (Accuracy, Precision, Recall, F1, MCC, Spatial MCC).

### 3. Explainable AI (XAI) and Interpretability

**Objective**: Analyze what physical features contribute most drastically to landslides globally and locally.

- `results/figures/shap_summary.png`: SHAP bee-swarm plot showing total feature impact and directionality (e.g., how high rainfall scales susceptibility).
- `results/figures/shap_dependence_top_[1-3]_[Feature].png`: Interaction dependencies highlighting non-linear relationships.

### 4. Advanced Statistical Validation

**Objective**: Prove rigorous statistical significance ensuring model superiority isn't due to random chance.

- `results/data/friedman_test.txt`: The general score verifying performance variation across 5 models.
- `results/data/nemenyi_test_results.csv`: Pairwise p-value matrices (insert this table to prove statistical advantage).
- `results/data/holdout_bootstrap_ci.csv`: 95% Confidence Interval (CI) limits on model metrics.

### 5. Final Geospatial Output

**Objective**: Visualize spatial zones of vulnerability and report logical hazard consistency using SCAI.

- `results/data/LSM_Grid_Data.csv`: Pixel-by-pixel spatial coordinates (Lat/Lon) matched with `Susceptibility` and `Uncertainty` arrays. Use this to explicitly generate maps manually in **ArcGIS Pro** or QGIS as requested.
- `results/data/LSM_Map_Raw.tif` & `Uncertainty_Map_Raw.tif`: Ready-made geographical TIFFs.
- `results/data/SCAI_Analysis.csv`: Proof that actual landslide points fall correctly into ‘Very High`susceptibility zones rather than`Low` zones.

---

## 🚀 Execution Guide

If you write new iterations of the pipeline:

1. Store datasets exclusively in `data/`.
2. Construct pipelines inside `src/`.
3. Assure your Python script directs logging and graph exports strictly to `results/data/` or `results/figures/`.
