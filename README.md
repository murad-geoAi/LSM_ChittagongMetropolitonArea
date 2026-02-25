# Knowledge-Guided Bayesian LSM Pipeline

This repository runs a landslide susceptibility workflow with a knowledge-guided loss design, standardized validation, and Bayesian-only deployment for final susceptibility/uncertainty map production.

## Model Policy

- **Bayesian deployment candidates:** `BNN`, `DeepEnsemble`, `BLR`, `GPC`
- **Benchmark model:** `LR`
- Deep models (`BNN`, `DeepEnsemble`) are implemented with **PyTorch + PyTorch Lightning**.
- Traditional models (`BLR`, `GPC`, `LR`) are implemented with sklearn-compatible estimators.
- All models share preprocessing:
  - median imputation
  - standard scaling
  - classifier

## Knowledge-Guided Loss

- The pipeline uses a monotonicity-based regularizer tied to geomorphological priors.
- Penalized directional constraints are applied to factors such as slope/elevation/precipitation (positive-risk direction) and distance-to-river/distance-to-road/NDVI (negative-risk direction).
- Dataset diagnostics and prior-direction agreement reports are exported in `results/data/`.

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
cd "g:\KGBNN\LSM_ChittagongMetropolitonArea"
python src/bayesian_lsm_pipeline.py
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
- `nemenyi_test_results_*.csv`
- `model_report.json`

### Dataset diagnostics (`results/data/`)

- `dataset_characteristics.json`
- `dataset_missingness.csv`
- `dataset_effect_sizes.csv`
- `dataset_target_correlations.csv`
- `knowledge_priors_review.csv`

### Figures (`results/figures/`)

- `roc_curve.png`
- `pr_curve.png`
- `calibration_curve_holdout.png`
- `confusion_matrix_bnn.png`
- `confusion_matrix_deepensemble.png`
- `confusion_matrix_blr.png`
- `confusion_matrix_gpc.png`
- `confusion_matrix_lr.png`
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

- `model_selection_table.csv` includes all benchmarked models.
- `Deployment_Eligible` marks Bayesian/uncertainty-aware deployment candidates only.
- `Deployment_Selected` marks the final deployed mapping model.
- If the deterministic benchmark ranks first overall, deployment still remains Bayesian/uncertainty-aware by design.
