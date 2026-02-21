# Demo Research Paper (Template)

**Project:** Bayesian / probabilistic machine-learning pipeline for Landslide Susceptibility Mapping (LSM)  
**Study area (in this repo):** Chittagong (Chattogram) Metropolitan Area, Bangladesh *(update if different)*  
**Purpose of this document:** A “filled” demo manuscript you can later edit into your final research paper. Bracketed text like `[ADD …]` indicates where you should customize.

---

# Probabilistic and Bayesian Machine Learning for Landslide Susceptibility Mapping: A Reproducible Pipeline with Spatial Block Validation, Calibration, and Explainable AI (Demo Manuscript)

**[Author 1]**, **[Author 2]**  
[Affiliation(s)]  
Corresponding author: [email]

## Abstract
Landslide susceptibility mapping (LSM) is increasingly performed using machine-learning models; however, reported performance can be optimistic when spatial autocorrelation and probability calibration are not evaluated. This demo study presents a reproducible LSM workflow that combines (i) multicollinearity screening via variance inflation factor (VIF), (ii) probabilistic/bayesian and non-bayesian classifiers, (iii) repeated stratified cross-validation on the training split, (iv) spatial block validation using KMeans-derived blocks with GroupKFold, (v) holdout testing with bootstrap confidence intervals, (vi) calibration assessment via expected calibration error (ECE), and (vii) SHAP-based explainability. Using a deduplicated inventory of **369** samples (155 landslide and 214 non-landslide) and 14 conditioning factors, the best overall model selected by a multi-criteria ranking was **Random Forest**. On the holdout test set (n=74), Random Forest achieved **ROC-AUC=0.866**, **MCC=0.611**, **F1=0.774**, and **Brier=0.153**. Under spatial block validation, performance decreased (Random Forest **spatial ROC-AUC=0.805**, **spatial MCC=0.323**), highlighting the importance of spatially-aware evaluation. Susceptibility zoning into five classes (Jenks natural breaks) concentrated **98.7%** of mapped landslide points into the “Very High” class occupying **9.47%** of the study-area pixels (SCAI=0.096). The template emphasizes methodological reporting and interprets results in terms of spatial generalization, calibration, explainability, and practical use for hazard screening.

**Keywords:** landslide susceptibility mapping; probabilistic ML; Bayesian logistic regression; Gaussian process classification; spatial cross-validation; calibration; SHAP; uncertainty

---

## 1. Introduction
Landslides are a recurring natural hazard in many mountainous and hilly regions, causing loss of life, infrastructure damage, and long-term environmental impacts. LSM aims to estimate the spatial probability (or relative likelihood) of landslide occurrence, supporting planning, early warning, and risk reduction.

Machine-learning approaches often report strong predictive accuracy, but three issues commonly limit real-world reliability: (i) *spatial autocorrelation* can inflate performance if nearby samples appear in both training and testing; (ii) *probability calibration* is rarely checked even though maps are interpreted probabilistically; and (iii) *interpretability and uncertainty* are often insufficiently communicated to decision makers.

This demo manuscript documents a full end-to-end workflow implemented in the project pipeline (`__pycache__/bayesian_lsm_pipeline.py`) and summarizes the quantitative outputs saved in `results/`. It is intended as a detailed starting point for your final research paper.

---

## 2. Materials and Methods

### 2.1 Study area
[ADD a standard study-area subsection: location, administrative boundary, relief/elevation range, climate/rainfall regime, lithology/soil, land cover, drainage/river network, roads/settlements, and known landslide triggers.]

**Note (from the dataset coordinates):** sample locations span approximately **Longitude 91.7589–91.8425** and **Latitude 22.3269–22.4907** (WGS84), consistent with the Chittagong/Chattogram metropolitan region.

### 2.2 Landslide inventory and conditioning factors
The dataset contains a binary target variable (`Landslide` ∈ {0,1}) and coordinates (`Longitude`, `Latitude`). Fourteen conditioning factors are used as predictors:

- Topography / terrain: `Elevation`, `Slope`, `Aspect`, `Curvature`, `SPI` (stream power index), `TWI` (topographic wetness index), `TRI` (terrain ruggedness index)
- Surface properties: `LULC` (land use/land cover), `NDVI`, `SoilTexture`
- Hydro-meteorology and proximity: `Precipitation`, `Drainage_Density`, `Distance_to_River`, `Distance_to_Road`

[ADD data sources for each layer: DEM product, rainfall dataset, LULC/NDVI product, soil map, road/river vectors; include spatial resolution, projection, and acquisition dates.]

### 2.3 Data preparation
The pipeline applies the following steps:

1. **Deduplication:** raw samples are deduplicated (7 duplicates removed).
2. **Type handling:** any non-numeric predictor columns are converted to categorical codes.
3. **Train/test split:** 80/20 stratified split with `random_state=42`.
4. **Missing values:** median imputation is performed *within each modeling pipeline*.
5. **Scaling:** standard scaling is applied (primarily needed for SVC/BLR/GPC).

**Dataset and split summary (as executed in this repo):**

| Split              |   Rows |   Neg (0) |   Pos (1) |
|:-------------------|-------:|----------:|----------:|
| All (deduplicated) |    369 |       214 |       155 |
| Train              |    295 |       171 |       124 |
| Test               |     74 |        43 |        31 |

### 2.4 Multicollinearity screening with VIF
To reduce redundancy and stabilize estimation, multicollinearity is assessed using the **variance inflation factor (VIF)** on the *training split only*. Predictors are imputed and standardized, then VIF is computed iteratively; if the maximum VIF exceeds a threshold (VIF > 5.0), the corresponding predictor is dropped and VIF is recomputed.

In this run, all predictors remained below the threshold and were retained:

| Feature           |   VIF |   TOL |
|:------------------|------:|------:|
| Elevation         | 2.82  | 0.355 |
| Slope             | 3.342 | 0.299 |
| Aspect            | 1.162 | 0.861 |
| Curvature         | 1.237 | 0.808 |
| SPI               | 1.611 | 0.621 |
| TWI               | 2.17  | 0.461 |
| TRI               | 1.826 | 0.548 |
| LULC              | 2.419 | 0.413 |
| NDVI              | 2.081 | 0.481 |
| Precipitation     | 1.14  | 0.877 |
| Drainage_Density  | 1.851 | 0.54  |
| Distance_to_River | 1.89  | 0.529 |
| Distance_to_Road  | 1.267 | 0.789 |
| SoilTexture       | 1.329 | 0.753 |

### 2.5 Candidate models
Four probabilistic classifiers are compared using consistent preprocessing (median imputation + standard scaling):

1. **BLR (Bayesian Logistic Regression):** Bayesian logistic regression fit with a Laplace approximation (Gaussian prior; `prior_precision=1.5`). Predictive probabilities incorporate an approximation of posterior uncertainty; optional uncertainty is estimated via posterior sampling.
2. **GPC (Gaussian Process Classifier):** GaussianProcessClassifier with a Matern kernel (ν=1.5) plus a small WhiteKernel noise term; multiple optimizer restarts improve fit stability.
3. **Random Forest:** 300 trees (`n_estimators=300`), `min_samples_leaf=2`, class-weighting via `balanced_subsample`.
4. **SVC:** RBF-kernel SVC (`C=2.0`, `gamma=scale`) with class weights and probability estimation enabled.

[OPTIONAL: add a short rationale for selecting these models and how each supports probabilistic LSM.]

### 2.6 Validation design
This project reports performance under three complementary validation regimes.

**(A) Repeated stratified CV (training split):**  
RepeatedStratifiedKFold with **5 folds × 3 repeats** (15 train-CV test partitions) on the training set only.

**(B) Holdout test evaluation:**  
Models are trained on the full training split (n=295) and evaluated on the holdout test split (n=74).

**(C) Spatial block validation:**  
To approximate spatially independent evaluation, coordinates are clustered into spatial “blocks” using KMeans. Blocks are then used as groups in GroupKFold:

- `n_blocks = min(10, max(4, n_samples // 30))`
- `n_splits = min(5, n_blocks)`

For this run (n_samples=295), this yields **9 blocks** and **5 spatial folds**. Metrics reported: spatial MCC and spatial ROC-AUC.

### 2.7 Metrics reported
The pipeline computes metrics capturing both *classification* and *probabilistic* quality:

- **Discrimination:** ROC-AUC, PR-AUC
- **Thresholded classification:** Accuracy, Precision, Recall, F1, Matthews correlation coefficient (MCC)
- **Probability quality / calibration:** Brier score, LogLoss, Expected Calibration Error (ECE; 10 bins)

### 2.8 Statistical comparison of models
To test whether performance differences among models are statistically meaningful (on the training CV), the pipeline performs:

1. **Friedman test** on per-fold scores (default metric: F1).
2. If significant (p < 0.05), **Nemenyi post-hoc** pairwise comparisons.

### 2.9 Bootstrap confidence intervals on holdout metrics
Holdout uncertainty is quantified via nonparametric bootstrap resampling on the test set. For each model, **1200** bootstrap replicates are drawn (with replacement), and median + 95% percentile intervals are reported for MCC, ROC-AUC, PR-AUC, Brier, and ECE.

### 2.10 Multi-criteria model selection
Rather than selecting solely by a single metric, the project uses a **rank-based selection score** combining:

- CV MCC rank (weight 0.20)
- Holdout MCC rank (weight 0.35)
- Spatial MCC rank (weight 0.30)
- Holdout Brier rank (weight 0.10; lower is better)
- Holdout ECE rank (weight 0.05; lower is better)

The model with the *lowest* weighted rank score is selected as the final model.

### 2.11 Explainability (SHAP)
SHAP KernelExplainer is applied to a random subsample of the training data (up to 80 explained samples, 30 background samples) to produce:

- SHAP summary plot (`results/shap_summary.png`)
- SHAP dependence plots for the top features (see `results/shap_dependence_*.png`)

[In your final paper, interpret these in the context of geomorphological processes and local conditions.]

### 2.12 Susceptibility map generation, uncertainty, and SCAI
After model selection, the final model is trained on **all deduplicated samples** and used to generate continuous susceptibility and uncertainty surfaces:

1. Predict per-sample susceptibility probabilities.
2. Interpolate probabilities to a **500 × 500** grid using `scipy.interpolate.griddata` (linear; edges filled by nearest).
3. Classify susceptibility into **five classes** using **Jenks natural breaks** (fallback to quantiles if Jenks fails).
4. Export GeoTIFFs (e.g., `results/LSM_Map_Raw.tif`, `results/LSM_Map_Classified.tif`, `results/Uncertainty_Map_Raw.tif`).

**Uncertainty definition:**  
- If the model exposes a posterior predictive uncertainty function (BLR), the pipeline uses normalized posterior standard deviation.
- Otherwise, uncertainty is approximated as closeness to the decision boundary: `1 - 2*|p - 0.5|` (scaled to [0,1]).

**SCAI (seed cell area index):**  
SCAI is computed as `Area% / Landslide%` for each susceptibility class. Because some classes may contain zero landslide points, a Jeffreys smoothing (α=0.5) is also reported (`SCAI_Smoothed`) to keep finite values.

---

## 3. Results

### 3.1 Repeated CV results (training split)
Across repeated stratified CV on the training split, Random Forest showed the strongest overall discrimination and MCC, with SVC and GPC close behind. BLR underperformed the other models in both discrimination and MCC.

| Model        |   mcc_mean |   mcc_std |   roc_auc_mean |   roc_auc_std |   f1_mean |   f1_std |   brier_mean |   brier_std |   ece_mean |   ece_std |
|:-------------|-----------:|----------:|---------------:|--------------:|----------:|---------:|-------------:|------------:|-----------:|----------:|
| RandomForest |      0.498 |     0.091 |          0.849 |         0.04  |     0.711 |    0.053 |        0.159 |       0.017 |      0.115 |     0.029 |
| SVC          |      0.475 |     0.08  |          0.843 |         0.027 |     0.705 |    0.053 |        0.165 |       0.014 |      0.113 |     0.03  |
| GPC          |      0.45  |     0.105 |          0.836 |         0.034 |     0.681 |    0.069 |        0.168 |       0.013 |      0.13  |     0.022 |
| BLR          |      0.388 |     0.133 |          0.757 |         0.057 |     0.631 |    0.086 |        0.201 |       0.025 |      0.142 |     0.047 |

Associated figures saved by the pipeline (training out-of-fold):
- ROC curve: `results/roc_curve.png`
- PR curve: `results/pr_curve.png`
- Confusion matrices (selected): `results/confusion_matrix_randomforest.png`, `results/confusion_matrix_gpc.png`

### 3.2 Holdout test performance
Holdout testing confirms Random Forest as the strongest classifier overall (highest MCC and ROC-AUC), while GPC shows the best calibration (lowest ECE).

| Model        |   Accuracy |    F1 |   MCC |   ROC_AUC |   PR_AUC |   Brier |   ECE |
|:-------------|-----------:|------:|------:|----------:|---------:|--------:|------:|
| RandomForest |      0.811 | 0.774 | 0.611 |     0.866 |    0.865 |   0.153 | 0.12  |
| GPC          |      0.743 | 0.708 | 0.481 |     0.821 |    0.814 |   0.172 | 0.084 |
| SVC          |      0.743 | 0.708 | 0.481 |     0.809 |    0.809 |   0.173 | 0.164 |
| BLR          |      0.676 | 0.625 | 0.34  |     0.775 |    0.723 |   0.193 | 0.112 |

Calibration artifacts:
- Holdout calibration curve: `results/calibration_curve_holdout.png`
- Holdout calibration ECE ranking: `results/holdout_calibration_summary.csv`

### 3.3 Holdout bootstrap confidence intervals
Bootstrap intervals quantify performance uncertainty on the holdout split. Random Forest maintains the highest median MCC and AUC with relatively tight intervals.

| Model        |   MCC_Median |   MCC_CI_Low |   MCC_CI_High |   ROC_AUC_Median |   ROC_AUC_CI_Low |   ROC_AUC_CI_High |   PR_AUC_Median |   PR_AUC_CI_Low |   PR_AUC_CI_High |   Brier_Median |   Brier_CI_Low |   Brier_CI_High |   ECE_Median |   ECE_CI_Low |   ECE_CI_High |
|:-------------|-------------:|-------------:|--------------:|-----------------:|-----------------:|------------------:|----------------:|----------------:|-----------------:|---------------:|---------------:|----------------:|-------------:|-------------:|--------------:|
| RandomForest |        0.616 |        0.408 |         0.791 |            0.871 |            0.768 |             0.944 |           0.872 |           0.766 |            0.944 |          0.151 |          0.116 |           0.194 |        0.156 |        0.093 |         0.224 |
| SVC          |        0.5   |        0.29  |         0.69  |            0.813 |            0.694 |             0.904 |           0.817 |           0.67  |            0.907 |          0.172 |          0.125 |           0.227 |        0.18  |        0.112 |         0.253 |
| GPC          |        0.482 |        0.281 |         0.675 |            0.823 |            0.711 |             0.917 |           0.824 |           0.686 |            0.91  |          0.171 |          0.133 |           0.215 |        0.151 |        0.097 |         0.224 |
| BLR          |        0.331 |        0.114 |         0.539 |            0.773 |            0.652 |             0.868 |           0.733 |           0.546 |            0.864 |          0.196 |          0.153 |           0.243 |        0.153 |        0.088 |         0.235 |

### 3.4 Spatial block validation
Spatial block validation reduces performance for all models relative to random CV, consistent with reduced spatial leakage and increased difficulty of extrapolation.

| Model        |   Spatial_MCC_Mean |   Spatial_MCC_Std |   Spatial_ROC_AUC_Mean |   Spatial_ROC_AUC_Std |
|:-------------|-------------------:|------------------:|-----------------------:|----------------------:|
| RandomForest |              0.323 |             0.098 |                  0.805 |                 0.072 |
| SVC          |              0.319 |             0.126 |                  0.76  |                 0.074 |
| GPC          |              0.259 |             0.171 |                  0.769 |                 0.064 |
| BLR          |              0.226 |             0.191 |                  0.709 |                 0.084 |

### 3.5 Statistical comparison (training CV)
The Friedman test on CV-fold F1 scores indicates statistically significant differences among models:

- Friedman statistic = **17.898**, p-value = **0.000462** (alpha = 0.05)

Nemenyi post-hoc results suggest BLR differs significantly from Random Forest (p=0.00135) and SVC (p=0.00493), while Random Forest and SVC are not significantly different in this test (p≈0.985).

### 3.6 Model selection outcome
The multi-criteria weighted rank score selected **Random Forest** as the final model:

| Model        |   selection_score |   mcc_mean |   MCC |   Spatial_MCC_Mean |   Brier |   ECE |
|:-------------|------------------:|-----------:|------:|-------------------:|--------:|------:|
| RandomForest |              1.1  |      0.498 | 0.611 |              0.323 |   0.153 | 0.12  |
| SVC          |              2.2  |      0.475 | 0.481 |              0.319 |   0.173 | 0.164 |
| GPC          |              2.45 |      0.45  | 0.481 |              0.259 |   0.172 | 0.084 |
| BLR          |              3.9  |      0.388 | 0.34  |              0.226 |   0.193 | 0.112 |

### 3.7 Explainability (SHAP)
The project produces SHAP summary and dependence plots (see `results/shap_summary.png` and `results/shap_dependence_*.png`). Based on the saved dependence-plot filenames in this repository run, the most influential features included **Elevation**, **TWI**, **Slope**, **Precipitation**, and **Distance_to_Road** (interpretation should be completed using the plots and domain knowledge).

[ADD: narrative interpretation of partial dependence patterns; relate to slope instability processes, drainage concentration, anthropogenic road cuts, rainfall triggering, vegetation cover, etc.]

### 3.8 Susceptibility zoning and SCAI
Susceptibility was reclassified into five tiers (Very Low → Very High). The seed cell area index (SCAI) results are:

| Tier      |   Area_Pct |   Landslide_Pct | SCAI   |   SCAI_Smoothed |
|:----------|-----------:|----------------:|:-------|----------------:|
| Very Low  |     29.372 |            0    |        |          92.523 |
| Low       |     23.163 |            0    |        |          72.962 |
| Moderate  |     24.036 |            0    |        |          75.715 |
| High      |     13.963 |            1.29 | 10.821 |           8.797 |
| Very High |      9.466 |           98.71 | 0.096  |           0.097 |

Interpretation: the **Very High** class occupies ~**9.47%** of pixels while containing ~**98.7%** of landslide points, producing SCAI ≪ 1 (desirable concentration). However, note that SCAI here is computed after training the final model on all samples; therefore, it is not an independent validation and should be interpreted alongside holdout and spatial-block results.

Generated rasters and artifacts:
- Susceptibility (continuous): `results/LSM_Map_Raw.tif`
- Susceptibility (classified): `results/LSM_Map_Classified.tif` (and `results/LSM_Map_Reclassified.tif` if present)
- Uncertainty: `results/Uncertainty_Map_Raw.tif` and `results/Uncertainty_Map.tif`

---

## 4. Discussion

### 4.1 Model performance and practical implications
Across random CV, holdout testing, and spatial validation, Random Forest consistently provides the strongest overall discrimination and MCC. The margin over SVC/GPC is larger on the holdout split than in repeated CV, suggesting that relative model ranking can depend on the evaluation regime and sample split.

In practical hazard screening, the combination of high ROC-AUC and strong MCC suggests Random Forest can separate landslide from non-landslide locations while maintaining balance under class imbalance. The high PR-AUC is particularly relevant because landslide inventories often contain many more stable locations than failures.

### 4.2 Spatial generalization and the need for spatial CV
Performance drops under spatial block CV (e.g., Random Forest MCC from 0.498 in repeated CV to 0.323 in spatial CV). This is expected when spatial autocorrelation exists: neighboring samples share similar terrain and land-cover properties, inflating random-CV performance. Spatial validation provides a more realistic estimate of performance when the model is applied to new geographic areas or future mapping campaigns.

[ADD: discussion of spatial sampling design, inventory completeness, and potential improvements such as larger spatial blocks, alternative blocking strategies, or spatial leave-one-area-out tests.]

### 4.3 Probability calibration and reliability
Probability calibration matters when susceptibility is communicated as “probability of occurrence” or used for threshold-based zoning. In this run, GPC achieved the lowest holdout ECE (0.084), indicating the most reliable probabilities among the compared models. Random Forest achieved better overall MCC and AUC but slightly worse calibration than GPC.

Two practical options follow:
1. Use Random Forest for *ranking and zoning* but apply post-hoc calibration (e.g., isotonic or Platt scaling) if calibrated probabilities are required.
2. Use GPC (or a calibrated RF) where decision thresholds require reliable probabilities.

### 4.4 Explainability and geomorphological plausibility
SHAP analyses indicate that terrain and hydrologic indices (e.g., elevation, slope, wetness index), precipitation, and road proximity are influential in the model. This aligns with common landslide controls: steep slopes and terrain roughness increase driving forces; drainage concentration and wetness affect pore pressure; rainfall triggers failures; and roads can introduce slope cuts and altered drainage.

[ADD: cite relevant literature supporting each factor and compare signs/patterns with prior studies in Bangladesh/SE Asia.]

### 4.5 Uncertainty mapping: interpretation and limitations
The project provides an uncertainty surface, but its meaning depends on the model:

- For BLR, uncertainty is more clearly linked to parameter/posterior uncertainty via posterior sampling.
- For non-bayesian models, uncertainty is approximated as “closeness to 0.5 probability,” which is useful as a *confidence heuristic* but is not a full epistemic uncertainty quantification.

Future improvements could include ensemble-based epistemic uncertainty (e.g., RF variance across trees, bootstrap ensembles), conformal prediction, or fully bayesian spatial models depending on data availability.

### 4.6 Map generation approach and opportunities to strengthen spatial modeling
This pipeline interpolates predicted susceptibility from sample points onto a grid using `griddata`. This is a pragmatic approach when a full raster stack for predictors is not available in the workflow; however, it may smooth sharp terrain transitions and can be sensitive to sample density. A stronger “production” approach is to compute conditioning factors as rasters and apply the trained model per pixel/cell.

[ADD: proposed production workflow with raster stacks, tiling, and masking by AOI polygons; include projection/area-accurate CRS for areal statistics.]

---

## 5. Conclusion
This demo manuscript documents a complete LSM workflow emphasizing rigorous evaluation (random CV, holdout testing, and spatial block validation), calibration assessment, uncertainty mapping, and explainability. For the Chittagong metropolitan study area dataset used here, Random Forest achieved the best overall predictive performance and spatial generalization, while GPC provided the best-calibrated probabilities. Spatial validation revealed a meaningful performance decrease relative to random CV, underscoring the need for spatially-aware model assessment in landslide susceptibility studies.

---

## 6. Data and code availability (template)
- **Code:** `__pycache__/bayesian_lsm_pipeline.py` (pipeline implementation)  
- **Outputs:** `results/` (metrics tables, plots, rasters)  
- **Input dataset:** `../Landslide_Data.csv` (landslide inventory + factors; update path as needed)

[ADD: DOI/Zenodo link or institutional repository link if you publish the dataset and code.]

---

## Declarations (placeholders)
- **Ethics approval:** [ADD if applicable]
- **Consent to participate:** [ADD if applicable]
- **Competing interests:** [ADD]
- **Funding:** [ADD]
- **Author contributions:** [ADD]
- **Acknowledgements:** [ADD]

---

## References (placeholders)
[ADD: references for landslide susceptibility mapping, spatial CV, calibration metrics, SHAP, and datasets used.]

