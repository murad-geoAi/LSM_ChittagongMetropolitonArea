import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import sys

# Load pipeline methods
sys.path.append('__pycache__')
from bayesian_lsm_pipeline import load_dataset, build_model_builders, build_pipeline, RANDOM_SEED, TEST_SIZE

plt.rcParams.update({'axes.labelsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11})

# 1. VIF Tolerance Curve
print("Generating VIF curve...")
vif_df = pd.read_csv('results/data/vif_selected_features.csv')
fig, ax1 = plt.subplots(figsize=(12, 6.5))
x = np.arange(len(vif_df))
ax1.plot(x, vif_df["VIF"], color="#d55e00", marker="o", linewidth=2.0, label="VIF")
ax1.axhline(5.0, color="#d55e00", linestyle="--", linewidth=1.2, label="VIF threshold (5)")
ax1.set_ylabel("Variance Inflation Factor (VIF)", color="#d55e00", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#d55e00", labelsize=11)
ax1.set_ylim(bottom=0.0)

ax2 = ax1.twinx()
ax2.plot(x, vif_df["TOL"], color="#0072b2", marker="s", linewidth=2.0, label="Tolerance (1/VIF)")
ax2.axhline(0.2, color="#0072b2", linestyle="--", linewidth=1.2, label="Tolerance threshold (0.20)")
ax2.set_ylabel("Tolerance", color="#0072b2", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#0072b2", labelsize=11)
ax2.set_ylim(0.0, 1.05)

ax1.set_xticks(x)
ax1.set_xticklabels(vif_df["Feature"], rotation=45, ha="right", fontsize=11)
ax1.set_xlabel("Conditioning Factors", fontsize=11)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=11)
fig.tight_layout()
fig.savefig('results/figures/vif_tolerance_curve.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 2. CV Summary Comparison
print("Generating CV summary...")
cv_df = pd.read_csv("results/data/cv_metrics_summary.csv")
sp_df = pd.read_csv("results/data/spatial_cv_metrics.csv")
df = pd.merge(cv_df, sp_df, on=["Model", "Model_Role"])
preferred_order = ["BLR", "GPC", "LogReg_L2", "RandomForest"]
df["Model_Cat"] = pd.Categorical(df["Model"], categories=preferred_order, ordered=True)
df = df.sort_values("Model_Cat").reset_index(drop=True)
models = df["Model"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width / 2, df["roc_auc_mean"], width, yerr=df["roc_auc_std"], label="Repeated CV", capsize=5, color="#1f77b4")
axes[0].bar(x + width / 2, df["Spatial_ROC_AUC_Mean"], width, yerr=df["Spatial_ROC_AUC_Std"], label="Spatial block CV", capsize=5, color="#ff7f0e")
axes[0].set_ylabel("ROC-AUC", fontsize=11)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=11)
axes[0].set_ylim(0.5, 1.0)
axes[0].legend(fontsize=11)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)

axes[1].bar(x - width / 2, df["mcc_mean"], width, yerr=df["mcc_std"], label="Repeated CV", capsize=5, color="#1f77b4")
axes[1].bar(x + width / 2, df["Spatial_MCC_Mean"], width, yerr=df["Spatial_MCC_Std"], label="Spatial block CV", capsize=5, color="#ff7f0e")
axes[1].set_ylabel("MCC", fontsize=11)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=11)
axes[1].set_ylim(0, 1.0)
axes[1].legend(fontsize=11)
axes[1].grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig('results/figures/cv_summary_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 3. Models for ROC, PR, SHAP
print("Loading data for models...")
with open("results/data/model_report.json", "r") as f:
    report = json.load(f)
selected_features = report["selected_features"]
df_all, X, y, coords, features = load_dataset('results/data/LSM_Data.csv')
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

model_builders = build_model_builders(seed=RANDOM_SEED)
fitted_models = {}
for name, builder in model_builders.items():
    print(f"Training {name}...")
    model = build_pipeline(builder())
    model.fit(X_train, y_train)
    fitted_models[name] = model

colors = {"BLR": "#f28e2b", "GPC": "#4e79a7", "LogReg_L2": "#76b7b2", "RandomForest": "#59a14f"}

# ROC
print("Generating ROC curve...")
plt.figure(figsize=(10, 8))
for name, model in fitted_models.items():
    probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-9, 1.0 - 1e-9)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, color=colors.get(name, "black"), linewidth=2.0, label=f"{name} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/roc_curve.png', dpi=300)
plt.close()

# PR
print("Generating PR curve...")
plt.figure(figsize=(10, 8))
for name, model in fitted_models.items():
    probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-9, 1.0 - 1e-9)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    plt.plot(recall, precision, color=colors.get(name, "black"), linewidth=2.0, label=f"{name} (AUC = {pr_auc:.3f})")
plt.xlabel("Recall", fontsize=11)
plt.ylabel("Precision", fontsize=11)
plt.legend(loc="lower left", fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/pr_curve.png', dpi=300)
plt.close()

# SHAP
print("Generating SHAP plots...")
final_model = fitted_models["GPC"]
predict_fn = lambda x: final_model.predict_proba(pd.DataFrame(x, columns=selected_features))[:, 1]
background = shap.sample(X_train, min(30, len(X_train)), random_state=RANDOM_SEED)
explainer = shap.KernelExplainer(predict_fn, background, link="logit")

X_shap = X_train.sample(min(200, len(X_train)), random_state=RANDOM_SEED)
shap_values = explainer.shap_values(X_shap, nsamples=150)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, show=False)
ax = plt.gca()
ax.tick_params(labelsize=11)
plt.xlabel("SHAP value (impact on model output)", fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Waterfall
test_probs = final_model.predict_proba(X_test)[:, 1]
candidate_idx = np.where(y_test == 1)[0]
idx = int(candidate_idx[np.argmax(test_probs[candidate_idx])])
x_instance = X_test.iloc[[idx]]

shap_values_inst = explainer.shap_values(x_instance, nsamples=150)
if isinstance(shap_values_inst, list):
    val = np.asarray(shap_values_inst[1][0], dtype=float)
    base = float(np.asarray(explainer.expected_value)[1])
elif isinstance(shap_values_inst, np.ndarray) and shap_values_inst.ndim == 3:
    val = np.asarray(shap_values_inst[0, :, 1], dtype=float)
    base = float(np.asarray(explainer.expected_value)[1])
else:
    val = np.asarray(shap_values_inst[0], dtype=float)
    base = float(np.asarray(explainer.expected_value))

explanation = shap.Explanation(values=val, base_values=base, data=x_instance.iloc[0].to_numpy(dtype=float), feature_names=selected_features)

shap.plots.waterfall(explanation, show=False, max_display=14)
ax = plt.gca()
for text in ax.texts:
    text.set_color('black')
    text.set_fontsize(11)
ax.tick_params(labelsize=11)
plt.xlabel("SHAP value", fontsize=11)
plt.tight_layout()
plt.savefig('results/figures/shap_waterfall_example.png', dpi=300, bbox_inches='tight')
plt.close()

# Final step: copy to journal figures
print("Copying to journal figures...")
os.makedirs('journal figures', exist_ok=True)
figs = ['vif_tolerance_curve.png', 'roc_curve.png', 'pr_curve.png', 'shap_summary.png', 'cv_summary_comparison.png', 'shap_waterfall_example.png']
for f in figs:
    shutil.copy(f'results/figures/{f}', f'journal figures/{f}')

print("Done.")
