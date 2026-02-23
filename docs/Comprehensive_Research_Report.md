# Comprehensive Research Report: Landslide Susceptibility Mapping (LSM)

**Case Study**: Chittagong Metropolitan Area

## 1. Introduction

This report summarizes the findings, data processing, model evaluation, and spatial mapping results for the Landslide Susceptibility Modeling pipeline. A major focus of this research was to generate robust susceptibility and uncertainty indicators, comparing Bayesian models (Gaussian Process Classifier, Bayesian Logistic Regression) with frequentist counterparts (Random Forest, SVM).

## 2. Methodology & Feature Engineering

**Dataset Description:**

- The models were trained and validated on a comprehensive geospatial dataset, engineered with 14 topographical, hydrological, and environmental features.
- Initial preprocessing involved assessing Variance Inflation Factor (VIF) to drop highly collinear predictors (threshold = 5.0).
- **Features Used:** Elevation, Slope, Aspect, Curvature, SPI, TWI, TRI, LULC, NDVI, Precipitation, Drainage_Density, Distance_to_River, Distance_to_Road, and SoilTexture.

## 3. Model Performance Evaluation

Four models were robustly checked using 5-fold cross-validation with 3 repeats, spatial cross-validation, and an independent 20% holdout test set.

### Holdout Test Set Performance Highlights:

| Model            | Accuracy | Precision | Recall | F1 Score | MCC   | ROC-AUC | Brier Score | ECE   |
| ---------------- | -------- | --------- | ------ | -------- | ----- | ------- | ----------- | ----- |
| **RandomForest** | 81.1%    | 77.4%     | 77.4%  | 77.4%    | 0.611 | 0.866   | 0.153       | 0.120 |
| **GPC**          | 74.3%    | 67.6%     | 74.2%  | 70.8%    | 0.481 | 0.821   | 0.172       | 0.084 |
| **SVC**          | 74.3%    | 67.6%     | 74.2%  | 70.8%    | 0.481 | 0.809   | 0.173       | 0.164 |
| **BLR**          | 67.6%    | 60.6%     | 64.5%  | 62.5%    | 0.340 | 0.775   | 0.193       | 0.112 |

**Spatial Cross-Validation Validation:**

- To prevent spatial autocorrelation over-optimism, spatial-block CV was performed. RandomForest maintained a spatial MCC of 0.322, with GPC exhibiting 0.259.

**Model Selection Outcome:**

- **RandomForest** achieved the highest overall predictive power and ranking.
- **Gaussian Process Classifier (GPC)** achieved the lowest (best) Expected Calibration Error (ECE = 0.084), successfully representing the uncertainty distribution needed for Bayesian predictions.

## 4. Final Susceptibility and Uncertainty Mapping

Using the trained algorithm, predicting over the entire spatial grid (extracted to `LSM_Grid_Data.csv`) generated spatial models that were converted to GeoTIFF files and map visualisations.

- `LSM_Map_Raw.tif` / `susceptibility_map.png`: Final pixel-wise Landslide Susceptibility map highlighting vulnerable zones scaling from 0 (very low) to 1.0 (very high).
- `Uncertainty_Map_Raw.tif` / `uncertainty_map.png`: Captures the model's confidence boundary at each coordinate.

## 5. Spatial Validation via SCAI

The Seed Cell Area Index (SCAI) analysis provides critical external evaluation on the logical consistency of the map. Higher SCAI in Low Susceptibility zones and lower SCAI in High Susceptibility zones dictates excellent real-world validity.

| Susceptibility Tier |    Area % | Landslide % | Confirmed Landslides | SCAI Factor |
| ------------------- | --------: | ----------: | -------------------: | ----------: |
| Very Low            |    29.37% |        0.0% |                    0 |   Very High |
| Low                 |    23.16% |        0.0% |                    0 |   Very High |
| Moderate            |    24.04% |        0.0% |                    0 |   Very High |
| High                |    13.96% |       1.29% |                    2 |       10.82 |
| **Very High**       | **9.47%** |  **98.71%** |              **153** |   **0.096** |

**Conclusion:**
With 98.71% of known historical landslides captured inside the "Very High" susceptibility zone—which represents only 9.47% of the total study area—the models hold exceptional consistency and real-world applicability for urban zoning and risk mitigation strategies in that region.

## 6. Project Artifacts

- **Maps:** TIFFs and visual PNGs stored under `results/data/` and `results/figures/`.
- **Explainable AI (SHAP):** Visual representations stored within `results/figures/` identifying core features controlling predicted vulnerabilities.
- **Statistical Significance tests:** Nemenyi tests validating performance superiority beyond random chance.
