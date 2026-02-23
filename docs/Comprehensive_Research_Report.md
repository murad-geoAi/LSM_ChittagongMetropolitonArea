# Comprehensive Research Report: Bayesian Deployment with Benchmark Model Comparison

**Case Study:** Chittagong Metropolitan Area, Bangladesh

## 1. Objective

This workflow evaluates four models under identical validation settings:

- Bayesian deployment candidates: `BLR`, `GPC`
- Benchmark performance models: `LogReg_L2`, `RandomForest`

The final susceptibility and uncertainty maps are constrained to Bayesian candidates only.

## 2. Method Summary

- Dataset: 369 samples (`155` landslide, `214` non-landslide)
- Predictors: 14 conditioning factors
- Split: stratified `80/20` train-test
- Shared preprocessing: median imputation + standard scaling
- Validation: repeated stratified CV (`5x3`), independent holdout, spatial block CV
- Metrics: Accuracy, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC, Brier, LogLoss, ECE
- Statistical test: Friedman on CV F1 with Nemenyi post-hoc when significant

## 3. Performance Results

### Holdout test metrics

| Model | Role | Accuracy | F1 | MCC | ROC-AUC | PR-AUC | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForest | Benchmark | 0.811 | 0.774 | 0.611 | 0.866 | 0.865 | 0.153 | 0.120 |
| GPC | Bayesian candidate | 0.743 | 0.708 | 0.481 | 0.821 | 0.814 | 0.172 | 0.084 |
| LogReg_L2 | Benchmark | 0.703 | 0.686 | 0.420 | 0.768 | 0.667 | 0.203 | 0.110 |
| BLR | Bayesian candidate | 0.676 | 0.625 | 0.340 | 0.775 | 0.723 | 0.193 | 0.112 |

### Repeated CV summary

| Model | Role | CV MCC (mean) | CV ROC-AUC (mean) | CV F1 (mean) |
| --- | --- | ---: | ---: | ---: |
| RandomForest | Benchmark | 0.498 | 0.849 | 0.711 |
| GPC | Bayesian candidate | 0.450 | 0.836 | 0.681 |
| LogReg_L2 | Benchmark | 0.433 | 0.756 | 0.684 |
| BLR | Bayesian candidate | 0.388 | 0.757 | 0.631 |

### Spatial CV summary

| Model | Role | Spatial MCC (mean) | Spatial ROC-AUC (mean) |
| --- | --- | ---: | ---: |
| RandomForest | Benchmark | 0.323 | 0.805 |
| GPC | Bayesian candidate | 0.259 | 0.769 |
| BLR | Bayesian candidate | 0.226 | 0.709 |
| LogReg_L2 | Benchmark | 0.214 | 0.709 |

## 4. Statistical Significance

- Friedman test on CV F1:
  - statistic = `17.7273`
  - p-value = `0.0005006466`
- Nemenyi post-hoc table saved as:
  - `results/data/nemenyi_test_results_20260224_012009.csv` (timestamped fallback due lock on legacy file)

## 5. Model Ranking and Deployment Decision

Weighted ranking (`results/data/model_selection_table.csv`) produced:

1. RandomForest (benchmark, overall rank 1, selection score 1.15)
2. GPC (Bayesian candidate, overall rank 2, selection score 1.95)
3. LogReg_L2 (benchmark, overall rank 3, selection score 3.35)
4. BLR (Bayesian candidate, overall rank 4, selection score 3.55)

Final deployment model: **GPC** (`Deployment_Selected = True`).

### Justification for Bayesian deployment

Even though RandomForest ranks first in pure discrimination benchmarks, deployment is restricted to Bayesian candidates to preserve:

- posterior uncertainty representation for uncertainty maps
- probabilistic interpretability aligned with Bayesian inference
- calibration-aware risk communication under uncertainty-focused research goals

## 6. Final Susceptibility and Uncertainty Products

Generated from the final Bayesian model (**GPC**) only:

- `results/data/LSM_Map_Raw.tif`
- `results/data/LSM_Map_Classified.tif`
- `results/data/Uncertainty_Map_Raw.tif`
- `results/data/LSM_Grid_Data.csv`

## 7. SCAI Plausibility

| Tier | Area % | Landslide % | Landslide Count | SCAI |
| --- | ---: | ---: | ---: | ---: |
| Very Low | 29.37 | 0.65 | 1 | 45.53 |
| Low | 23.16 | 0.65 | 1 | 35.90 |
| Moderate | 24.04 | 5.81 | 9 | 4.14 |
| High | 13.96 | 14.19 | 22 | 0.98 |
| Very High | 9.47 | 78.71 | 122 | 0.12 |

The Very High class captures most observed landslides inside a limited mapped area fraction.
