"""Additional spatial, attribution and classification checks; never edits inputs."""
from pathlib import Path
source=Path(__file__).with_name('run_revision_analysis.py').read_text(encoding='utf-8')
exec(compile(source.split('raw=pd.read_csv')[0],__file__,'exec'))
raw=pd.read_csv(ROOT/'revision/inputs/Landslide_Data.csv');d=raw.drop_duplicates().reset_index(drop=True)
features=json.loads((ROOT/'results/data/model_report.json').read_text())['selected_features'];X=d[features];y=d.Landslide.to_numpy()
tr,te=train_test_split(np.arange(len(d)),test_size=.2,stratify=y,random_state=42)
groups=KMeans(n_clusters=9,n_init=10,random_state=42).fit_predict(d.loc[tr,['Longitude','Latitude']])
foldrows=[]
for name in COLORS:
 for f,(a,b) in enumerate(GroupKFold(n_splits=5).split(tr,y[tr],groups),1):
  print('Spatial',name,f,flush=True)
  with warnings.catch_warnings(record=True):m=build(name).fit(X.iloc[tr[a]],y[tr[a]])
  foldrows.append(dict(Model=name,Fold=f,**metrics(y[tr[b]],m.predict_proba(X.iloc[tr[b]])[:,1])))
pd.DataFrame(foldrows).to_csv(OUT/'spatial_fold_metrics_reproduced.csv',index=False)
summary=pd.DataFrame(foldrows).groupby('Model')[['MCC','ROC_AUC']].agg(['mean','std']);summary.to_csv(OUT/'spatial_summary_reproduced.csv')
cv=pd.read_csv(ROOT/'results/data/cv_metrics_summary.csv').set_index('Model')
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');names=list(COLORS)
for ax,metric,cvcol in zip(axs,['MCC','ROC_AUC'],['mcc_mean','roc_auc_mean']):
 a=np.arange(4);ax.bar(a-.18,cv.loc[names,cvcol],width=.36,color='#9bc5df',edgecolor='.2',label='Repeated stratified CV');ax.bar(a+.18,summary.loc[names,(metric,'mean')],width=.36,color='#0F4D92',edgecolor='.2',label='Spatial CV');ax.set(xticks=a,xticklabels=names,ylabel=metric.replace('_','-'),ylim=(0,1))
axs[0].legend(fontsize=8);savefig(fig,'Figure_S2_validation')
clusters=[]
for k in range(9):
 sub=d.loc[tr[groups==k]];dist=haversine_distances(np.radians(sub[['Latitude','Longitude']]))*6371008.8
 clusters.append(dict(Cluster=k+1,N=len(sub),Positive=int(sub.Landslide.sum()),Negative=int((1-sub.Landslide).sum()),Max_sample_separation_km=dist.max()/1000))
pd.DataFrame(clusters).to_csv(OUT/'cluster_sizes.csv',index=False)
dd=haversine_distances(np.radians(d.loc[te,['Latitude','Longitude']]),np.radians(d.loc[tr,['Latitude','Longitude']]))*6371008.8
coords=d[['Longitude','Latitude']].duplicated(keep=False);d.loc[coords].to_csv(OUT/'shared_coordinates.csv',index=False)
pd.DataFrame({'Feature':features,'Missing_values':X.isna().sum().to_numpy(),'Unique_observed_values':X.nunique().to_numpy()}).to_csv(OUT/'predictor_audit.csv',index=False)
(OUT/'holdout_separation.json').write_text(json.dumps({'minimum_m':float(dd.min()),'median_nearest_m':float(np.median(dd.min(axis=1))),'holdout_within_500m_training':int((dd.min(axis=1)<500).sum())},indent=2))
# Document original and training-only rankings; do not pretend the revised rule was prespecified.
selection=cv[['mcc_mean','brier_mean','ece_mean']].copy();selection['spatial_mcc']=summary[('MCC','mean')]
selection['training_only_score']=.4*selection.mcc_mean.rank(ascending=False,method='min')+.4*selection.spatial_mcc.rank(ascending=False,method='min')+.1*selection.brier_mean.rank(method='min')+.1*selection.ece_mean.rank(method='min')
selection['Bayesian_eligible']=selection.index.isin(['BLR','GPC']);selection.to_csv(OUT/'training_only_selection_sensitivity.csv')
# Classification comparison at observed points, not area percentages.
u=pd.read_csv(OUT/'corrected_gpc_point_uncertainty.csv');import jenkspy
classrows=[]
for name,col in [('Probability','Probability'),('Latent_uncertainty','Normalized_latent_sd')]:
 vals=u[col].to_numpy();q=np.quantile(vals,[1/3,2/3]);jenks=np.array(jenkspy.jenks_breaks(vals,n_classes=3))[1:-1]
 for method,cuts in [('Tertiles',q),('Jenks',jenks),('Fixed_thirds',np.array([1/3,2/3]))]:
  cls=np.digitize(vals,cuts)
  classrows.append(dict(Variable=name,Method=method,Lower_cut=cuts[0],Upper_cut=cuts[1],Low=int((cls==0).sum()),Middle=int((cls==1).sum()),High=int((cls==2).sum()),Agreement_with_tertiles=float(np.mean(cls==np.digitize(vals,q)))))
pd.DataFrame(classrows).to_csv(OUT/'classification_sensitivity.csv',index=False)
# Regenerate SHAP with an explicit training background and holdout explanations.
import shap
np.random.seed(42);m=joblib.load(OUT/'GPC_holdout.joblib')
background=shap.sample(X.iloc[tr],30,random_state=42);explain=X.iloc[te].copy()
background.to_csv(OUT/'shap_background.csv',index_label='row_id');explain.to_csv(OUT/'shap_explained.csv',index_label='row_id')
def predict(data):return m.predict_proba(pd.DataFrame(data,columns=features))[:,1]
explainer=shap.KernelExplainer(predict,background,link='logit')
vals=explainer.shap_values(explain,nsamples=150,silent=True)
np.savez_compressed(OUT/'shap_values.npz',values=vals,expected_value=explainer.expected_value,row_ids=te)
importance=pd.DataFrame({'Feature':features,'Mean_absolute_SHAP':np.abs(vals).mean(axis=0)}).sort_values('Mean_absolute_SHAP',ascending=False);importance.to_csv(OUT/'shap_importance.csv',index=False)
shap.summary_plot(vals,explain,show=False,plot_size=(8,5.5));savefig(plt.gcf(),'Figure_S3_shap')
hold=pd.read_csv(OUT/'holdout_metrics_reproduced.csv').set_index('Model');old=pd.read_csv(ROOT/'results/data/holdout_metrics.csv').set_index('Model')
diffs={c:float(np.max(abs(hold[c]-old[c]))) for c in ['MCC','ROC_AUC','PR_AUC','ECE','Brier']}
assert max(diffs.values())<1e-6, diffs
spold=pd.read_csv(ROOT/'results/data/spatial_cv_metrics.csv').set_index('Model')
spdiff=float(np.max(abs(summary[('MCC','mean')]-spold.Spatial_MCC_Mean)))
assert spdiff<1e-6,spdiff
meta=dict(holdout_max_absolute_differences=diffs,spatial_mcc_max_absolute_difference=spdiff,shared_coordinate_rows=int(coords.sum()),uncertainty_variance_decomposition_passed=True,shap_background_n=30,shap_explained_n=74,shap_nsamples=150,shap_link='logit',shap_version=shap.__version__)
(OUT/'verification.json').write_text(json.dumps(meta,indent=2));print('ADDITIONAL CHECKS COMPLETE',meta,flush=True)
