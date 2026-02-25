# Knowledge-Guided Loss Justification (Reviewer-Facing)

## 1. Dataset Evidence (from `results/data/LSM_Data.csv`)

- Samples: `369`
- Target balance: `155` landslide (`42.0%`) vs `214` non-landslide (`58.0%`)
- Conditioning predictors used for modeling: `14` (plus coordinates for spatial CV/mapping only)
- Missing predictor values: `126` cells total (about `2.1%` of predictor matrix), distributed sparsely across factors
- Strongest observed target associations (absolute point-biserial correlation):
  - `Elevation` (`|r| = 0.359`)
  - `Slope` (`|r| = 0.322`)
  - `Precipitation` (`|r| = 0.276`)
  - `TWI` (`|r| = 0.260`)
  - `LULC` (`|r| = 0.207`)
- Notable predictor collinearity exists (examples):
  - `LULC` vs `NDVI`: `|r| ≈ 0.70`
  - `Elevation` vs `Slope`: `|r| ≈ 0.66`
  - `Drainage_Density` vs `Distance_to_River`: `|r| ≈ 0.61`

Interpretation: this is a small, moderately imbalanced, correlated tabular dataset where unconstrained deep models are at risk of overfitting or learning unstable/non-physical decision boundaries.

## 2. Selected Knowledge-Guided Loss

The selected loss is a **monotonicity-guided regularized binary loss**:

\[
\mathcal{L} = \mathcal{L}_{BCE} + \lambda_{KG}\,\mathcal{R}_{mono}
\]

with

\[
\mathcal{R}_{mono}
=
\frac{1}{|M|}
\sum_{j \in M}
\mathbb{E}\left[\max\left(0,\,-s_j \cdot \frac{\partial f_\theta(x)}{\partial x_j}\right)^2\right]
\]

where:
- \(f_\theta(x)\) is the model logit
- \(s_j \in \{+1,-1\}\) is prior direction for factor \(j\)
- \(M\) is the set of constrained factors

For linear models (LR/BLR), the derivative term reduces to coefficient-sign regularization, implemented via a smooth softplus penalty in coefficient space.

## 3. Why This Is Appropriate Here

- It encodes accepted geomorphological directionality while still allowing data-driven learning.
- It acts as structural regularization for small-sample training stability.
- It improves reviewer interpretability by linking model behavior to physically meaningful priors.
- It is architecture-agnostic: applied consistently to BNN, Deep Ensemble, BLR, and LR.

## 4. Prior Directions Used

Positive-risk direction (`+1`):
- `Elevation`, `Slope`, `Precipitation`, `TRI`, `Drainage_Density`

Negative-risk direction (`-1`):
- `Distance_to_River`, `Distance_to_Road`, `NDVI`

These priors and empirical agreement checks are exported per run to:
- `results/data/knowledge_priors_review.csv`

## 5. Reproducibility Artifacts

Each run now exports:
- `results/data/dataset_characteristics.json`
- `results/data/dataset_missingness.csv`
- `results/data/dataset_effect_sizes.csv`
- `results/data/dataset_target_correlations.csv`
- `results/data/knowledge_priors_review.csv`

This makes the loss choice explicitly traceable to measured dataset properties.
