# Bayesian LSM Pipeline with Benchmark Comparison

This repository runs a landslide susceptibility workflow that compares Bayesian and deterministic models under the same validation protocol, while restricting final map deployment to Bayesian models.

## Model Policy

- **Bayesian deployment candidates:** `BLR`, `GPC`
- **Benchmark performance models:** `LogReg_L2`, `RandomForest`
- All four models use the same preprocessing pipeline:
  - median imputation
  - standard scaling
  - classifier
- Final susceptibility and uncertainty maps are generated **only** from the best Bayesian model under the weighted selection criteria.

## Validation Framework

The workflow applies identical validation to all models:

1. repeated stratified CV (`5` folds x `3` repeats)
2. independent holdout testing (`20%`)
3. spatial block CV (`GroupKFold` with KMeans-derived blocks)
4. full metrics: Accuracy, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC, Brier, LogLoss, ECE

Statistical model-comparison test:

- `Friedman` test, and `Nemenyi` post-hoc when significant.

## Run

```bash
cd "h:\Baysesian Machine Learning-LSM\LSM_ChittagongMetropolitonArea"
python __pycache__/bayesian_lsm_pipeline.py
python src/generate_maps.py
python src/generate_architecture_figure.py
```

## Key Outputs

### Metrics and rankings (`results/data/`)

- `cv_metrics_summary.csv`
- `spatial_cv_metrics.csv`
- `holdout_metrics.csv`
- `holdout_bootstrap_ci.csv`
- `holdout_calibration_summary.csv`
- `model_selection_table.csv`
- `friedman_test.txt`
- `nemenyi_test_results_*.csv` (fallback timestamped file if lock prevents overwrite)
- `model_report.json`

### Figures (`results/figures/`)

- `roc_curve.png`
- `pr_curve.png`
- `calibration_curve_holdout.png`
- `confusion_matrix_blr.png`
- `confusion_matrix_gpc.png`
- `confusion_matrix_logreg_l2.png`
- `confusion_matrix_randomforest.png`
- `shap_summary.png`
- `model_architecture.png`
- `susceptibility_map.png`
- `uncertainty_map.png`

### Spatial products (`results/data/`)

- `LSM_Map_Raw.tif`
- `LSM_Map_Classified.tif`
- `Uncertainty_Map_Raw.tif`
- `LSM_Grid_Data.csv`
- `SCAI_Analysis.csv`

## Selection Interpretation

- `model_selection_table.csv` includes both benchmark and Bayesian models.
- `Deployment_Eligible` marks Bayesian candidates only.
- `Deployment_Selected` marks the final deployed Bayesian model.
- If a benchmark model ranks higher overall, deployment still remains Bayesian by design to preserve posterior uncertainty representation and uncertainty-aware interpretation.
