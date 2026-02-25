# Knowledge-Guided Bayesian Machine Learning for Landslide Susceptibility Mapping in Chittagong Metropolitan Area

## Abstract

This study implements a knowledge-guided Bayesian landslide susceptibility mapping (LSM) framework for the Chittagong Metropolitan Area, Bangladesh. The dataset contains 369 samples with 14 conditioning factors. Five models were benchmarked under identical preprocessing and validation: Bayesian Neural Network (BNN), Deep Ensemble Neural Network, Bayesian Logistic Regression (BLR), Gaussian Process Classifier (GPC), and Logistic Regression (LR) as the deterministic benchmark. A monotonicity-guided regularization term was used to encode domain priors for key factors (e.g., elevation, slope, precipitation, distance-to-river). Validation included repeated stratified cross-validation (5x3), holdout testing (20%), and spatial block CV. On holdout, GPC achieved the best overall performance (MCC=0.481, ROC-AUC=0.821, ECE=0.084) and was selected for deployment. Friedman testing on CV F1 scores indicated significant differences among models (p=0.0163). Final mapping products include susceptibility, uncertainty, and SCAI analysis, with the Very High susceptibility class capturing 78.71% of observed landslides within 9.47% of mapped area.

## 1. Study Objective

The objective is to produce calibrated susceptibility and uncertainty maps under a Bayesian deployment policy while preserving transparent benchmark comparison against a deterministic baseline.

## 2. Data and Predictors

- Total samples: 369
- Landslide / non-landslide: 155 / 214
- Predictors: 14 conditioning factors
- Coordinates used for spatial validation and map interpolation
- Missingness: 126 predictor cells (2.44%)

## 3. Modeling Framework

### 3.1 Compared Models

1. Bayesian Neural Network (BNN)
2. Deep Ensemble Neural Network
3. Bayesian Logistic Regression (BLR)
4. Gaussian Process Classifier (GPC)
5. Logistic Regression (LR, benchmark)

### 3.2 Shared Pipeline

- Median imputation
- Standard scaling
- Classifier

### 3.3 Knowledge-Guided Loss

The training objective uses:

\[
\mathcal{L} = \mathcal{L}_{BCE} + \lambda_{KG}\mathcal{R}_{mono}
\]

where the regularizer penalizes violations of expected monotonic directions for selected factors. Prior configuration and empirical direction agreement are exported in:

- `results/data/knowledge_priors_review.csv`

## 4. Validation Strategy

- Repeated stratified CV: 5 folds x 3 repeats
- Holdout split: 80/20
- Spatial block CV: GroupKFold over KMeans-derived spatial groups
- Metrics: Accuracy, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC, Brier, LogLoss, ECE
- Statistical test: Friedman + Nemenyi post-hoc

## 5. Results

### 5.1 Holdout Performance

| Model | Accuracy | F1 | MCC | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPC | 0.743 | 0.708 | 0.481 | 0.821 | 0.814 | 0.172 | 0.084 |
| BNN | 0.703 | 0.686 | 0.420 | 0.794 | 0.711 | 0.184 | 0.119 |
| BLR | 0.703 | 0.686 | 0.420 | 0.773 | 0.722 | 0.201 | 0.115 |
| LR | 0.703 | 0.686 | 0.420 | 0.769 | 0.668 | 0.203 | 0.110 |
| DeepEnsemble | 0.703 | 0.676 | 0.411 | 0.795 | 0.714 | 0.187 | 0.129 |

### 5.2 Spatial CV and Selection

- Best spatial MCC: DeepEnsemble (0.355)
- Best rank-based overall selection score: GPC (1.60)
- Deployment-selected model: GPC

### 5.3 Statistical Significance

Friedman test on CV F1:

- statistic = 12.145985
- p-value = 0.0162980630

Significant differences were detected; Nemenyi matrix is saved in `results/data/nemenyi_test_results.csv`.

### 5.4 SCAI Summary

Very High susceptibility class:

- Area percentage: 9.466%
- Landslide percentage: 78.710%
- SCAI: 0.120

## 6. Generated Products

### Data outputs

- `results/data/model_report.json`
- `results/data/model_selection_table.csv`
- `results/data/holdout_metrics.csv`
- `results/data/spatial_cv_metrics.csv`
- `results/data/LSM_Map_Raw.tif`
- `results/data/LSM_Map_Classified.tif`
- `results/data/Uncertainty_Map_Raw.tif`
- `results/data/SCAI_Analysis.csv`

### Figure outputs

- `results/figures/model_architecture.png`
- `results/figures/roc_curve.png`
- `results/figures/pr_curve.png`
- `results/figures/calibration_curve_holdout.png`
- `results/figures/shap_summary.png`
- `results/figures/susceptibility_map.png`
- `results/figures/uncertainty_map.png`

## 7. Reproducibility

Run from repository root:

```bash
python src/bayesian_lsm_pipeline.py
python src/generate_maps.py
python src/generate_architecture_figure.py
```
