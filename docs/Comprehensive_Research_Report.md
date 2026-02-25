# Comprehensive Research Report (Current Pipeline)

**Run date:** February 25, 2026  
**Repository:** `LSM_ChittagongMetropolitonArea`

## 1. Active Pipeline Scope

- Bayesian deployment candidates: `BNN`, `DeepEnsemble`, `BLR`, `GPC`
- Deterministic benchmark: `LR`
- Deep models implemented with PyTorch + PyTorch Lightning
- Knowledge-guided loss enabled with monotonic directional priors
- Final deployment restricted to Bayesian candidates only

## 2. Dataset Characteristics

From `results/data/dataset_characteristics.json`:

- Total samples: `369`
- Predictors: `14`
- Landslide: `155`
- Non-landslide: `214`
- Positive ratio: `0.4201`
- Missing predictor cells: `126` (`2.44%`)

Knowledge-prior agreement summary is exported to:

- `results/data/knowledge_priors_review.csv`

## 3. Validation Design

- Repeated stratified CV: `5 folds x 3 repeats`
- Holdout split: `80/20`
- Spatial block CV: `GroupKFold` over KMeans blocks
- Metrics: Accuracy, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC, Brier, LogLoss, ECE
- Statistical test: Friedman on CV F1 + Nemenyi post-hoc

## 4. Holdout Performance

From `results/data/holdout_metrics.csv`:

| Model | Role | Accuracy | F1 | MCC | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPC | Bayesian | 0.743 | 0.708 | 0.481 | 0.821 | 0.814 | 0.172 | 0.084 |
| BNN | Bayesian | 0.703 | 0.686 | 0.420 | 0.794 | 0.711 | 0.184 | 0.119 |
| BLR | Bayesian | 0.703 | 0.686 | 0.420 | 0.773 | 0.722 | 0.201 | 0.115 |
| LR | Benchmark | 0.703 | 0.686 | 0.420 | 0.769 | 0.668 | 0.203 | 0.110 |
| DeepEnsemble | Bayesian | 0.703 | 0.676 | 0.411 | 0.795 | 0.714 | 0.187 | 0.129 |

## 5. CV and Spatial Findings

- Best CV MCC: `GPC` (`0.450`)
- Best spatial MCC: `DeepEnsemble` (`0.355`)
- Best spatial ROC-AUC: `GPC` (`0.769`)

Sources:

- `results/data/cv_metrics_summary.csv`
- `results/data/spatial_cv_metrics.csv`

## 6. Statistical Significance

From `results/data/friedman_test.txt`:

- Test: Friedman
- Metric: F1
- Statistic: `12.145985`
- p-value: `0.0162980630` (significant at 0.05)

Nemenyi post-hoc matrix:

- `results/data/nemenyi_test_results.csv`

## 7. Model Selection and Deployment

From `results/data/model_selection_table.csv`:

- Overall rank 1: `GPC` (selection score `1.60`)
- `Deployment_Selected = True` for `GPC`
- `best_overall_model = GPC`
- `final_model = GPC`

## 8. Spatial Products and SCAI

Generated outputs:

- `results/data/LSM_Map_Raw.tif`
- `results/data/LSM_Map_Classified.tif`
- `results/data/Uncertainty_Map_Raw.tif`
- `results/data/LSM_Grid_Data.csv`
- `results/data/SCAI_Analysis.csv`

SCAI (`Very High` class):

- Area: `9.466%`
- Landslide capture: `78.710%`
- SCAI: `0.120`

## 9. Figures Generated

- `results/figures/roc_curve.png`
- `results/figures/pr_curve.png`
- `results/figures/calibration_curve_holdout.png`
- `results/figures/confusion_matrix_bnn.png`
- `results/figures/confusion_matrix_deepensemble.png`
- `results/figures/confusion_matrix_blr.png`
- `results/figures/confusion_matrix_gpc.png`
- `results/figures/confusion_matrix_lr.png`
- `results/figures/shap_summary.png`
- `results/figures/model_architecture.png`
- `results/figures/susceptibility_map.png`
- `results/figures/susceptibility_map_classified.png`
- `results/figures/uncertainty_map.png`
