"""Reproduce and audit the submitted analysis without modifying legacy files.

Run from repository root: python revision/run_revision_analysis.py
No original pipeline main() is called (it has cleanup operations).
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import ast, json, hashlib, platform, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score,average_precision_score,matthews_corrcoef,
    brier_score_loss,log_loss,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_curve,precision_recall_curve)
from sklearn.metrics.pairwise import haversine_distances
import sklearn, scipy, joblib
from threadpoolctl import threadpool_limits
threadpool_limits(1)
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'revision'/'analysis'; OUT.mkdir(exist_ok=True)
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
RANDOM_SEED=42
tree=ast.parse((ROOT/'__pycache__'/'bayesian_lsm_pipeline.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BayesianLogisticRegression')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<legacy BLR>','exec'))
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
COLORS={'BLR':'#767676','GPC':'#0F4D92','LogReg_L2':'#42949E','RandomForest':'#B64342'}
def savefig(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white'); plt.close(fig)
def build(name):
    estimators={
      'BLR':lambda:BayesianLogisticRegression(prior_precision=1.5,max_iter=500,tol=1e-6,n_posterior_samples=500,random_state=42),
      'GPC':lambda:GaussianProcessClassifier(kernel=1.0*Matern(length_scale=1.0,nu=1.5)+WhiteKernel(noise_level=1e-5),n_restarts_optimizer=5,random_state=42),
      'LogReg_L2':lambda:LogisticRegression(C=1.0,max_iter=3000,class_weight='balanced',random_state=42),
      'RandomForest':lambda:RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=1,random_state=42)}
    return Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('clf',estimators[name]())])
def ece(y,p,bins=10):
    ids=np.digitize(p,np.linspace(0,1,bins+1)[1:-1],right=True)
    return sum(np.mean(ids==b)*abs(np.mean(y[ids==b])-np.mean(p[ids==b])) for b in range(bins) if np.any(ids==b))
def metrics(y,p):
    pred=p>=.5; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(Accuracy=accuracy_score(y,pred),Precision=precision_score(y,pred,zero_division=0),Recall=recall_score(y,pred),F1=f1_score(y,pred),MCC=matthews_corrcoef(y,pred),ROC_AUC=roc_auc_score(y,p),PR_AUC=average_precision_score(y,p),Brier=brier_score_loss(y,p),LogLoss=log_loss(y,p),ECE=ece(y,p),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),FNR=fn/(fn+tp))
raw=pd.read_csv(ROOT/'revision'/'inputs'/'Landslide_Data.csv')
d=raw.drop_duplicates().reset_index(drop=True)
features=json.loads((ROOT/'results'/'data'/'model_report.json').read_text())['selected_features']
assert len(d)==369 and len(features)==14
assert np.allclose(d[features],pd.read_csv(ROOT/'results'/'data'/'LSM_Data.csv')[features],equal_nan=True)
X=d[features]; y=d.Landslide.to_numpy(); ids=np.arange(len(d))
tr,te=train_test_split(ids,test_size=.2,stratify=y,random_state=42)
groups=KMeans(n_clusters=9,n_init=10,random_state=42).fit_predict(d.loc[tr,['Longitude','Latitude']])
allgroups=KMeans(n_clusters=10,n_init=10,random_state=42).fit_predict(d[['Longitude','Latitude']])
folds=list(GroupKFold(n_splits=5).split(tr,y[tr],groups)); assignment=d[['Landslide','Longitude','Latitude']].copy()
assignment.insert(0,'row_id',ids); assignment['split']='training';assignment.loc[te,'split']='holdout'
assignment['training_cluster']=pd.Series(index=ids,dtype='Int64');assignment.loc[tr,'training_cluster']=groups+1
assignment['spatial_test_fold']=pd.Series(index=ids,dtype='Int64')
foldrows=[]
for f,(a,b) in enumerate(folds,1):
    assignment.loc[tr[b],'spatial_test_fold']=f
    foldrows.append(dict(Fold=f,Clusters=','.join(map(str,np.unique(groups[b])+1)),Train_n=len(a),Train_positive=int(y[tr[a]].sum()),Train_negative=int((1-y[tr[a]]).sum()),Test_n=len(b),Test_positive=int(y[tr[b]].sum()),Test_negative=int((1-y[tr[b]]).sum())))
assignment['all_sample_display_cluster']=allgroups+1
assignment.to_csv(OUT/'spatial_assignments.csv',index=False);pd.DataFrame(foldrows).to_csv(OUT/'spatial_fold_counts.csv',index=False)
neg=np.where(y==0)[0];pos=np.where(y==1)[0]
dist=haversine_distances(np.radians(d.loc[neg,['Latitude','Longitude']]),np.radians(d.loc[pos,['Latitude','Longitude']]))*6371008.8
nd=dist.min(axis=1)
pd.DataFrame({'row_id':neg,'slope_deg':d.loc[neg,'Slope'],'nearest_landslide_m':nd}).to_csv(OUT/'negative_sample_audit.csv',index=False)
audit=dict(raw_rows=len(raw),exact_duplicate_rows_removed=len(raw)-len(d),analysis_rows=len(d),positive_n=int(y.sum()),negative_n=int((1-y).sum()),negative_slope_at_least_10=int((d.loc[neg,'Slope']>=10).sum()),negative_within_500m=int((nd<500).sum()),minimum_negative_distance_m=float(nd.min()),negative_eligible_both=int(((nd>=500)&(d.loc[neg,'Slope']<10)).sum()),fractional_LULC=int((abs(d.LULC-np.round(d.LULC))>1e-6).sum()),fractional_SoilTexture=int((abs(d.SoilTexture-np.round(d.SoilTexture))>1e-6).sum()),training_n=len(tr),holdout_n=len(te),training_blocks=9)
old=pd.read_csv(ROOT/'results'/'data'/'LSM_Data.csv')
audit['legacy_uncertainty_max_difference_from_ambiguity']=float(np.max(abs(old.Uncertainty-(1-abs(old.Predicted_Prob-.5)*2))))
audit['source_sha256']=hashlib.sha256((ROOT/'revision'/'inputs'/'Landslide_Data.csv').read_bytes()).hexdigest()
(OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
print('DATA AUDIT',audit,flush=True)
fig,axs=plt.subplots(1,2,figsize=(11,5),layout='constrained')
for k in range(9):
    ix=tr[groups==k];axs[0].scatter(d.loc[ix,'Longitude'],d.loc[ix,'Latitude'],s=22,color=plt.cm.tab10(k),label=str(k+1))
axs[0].scatter(d.loc[te,'Longitude'],d.loc[te,'Latitude'],marker='x',c='black',s=26,label='Holdout')
axs[0].set_title('(a) Actual validation design: 295 training + 74 holdout')
for k in range(10):
    ix=ids[allgroups==k];axs[1].scatter(d.loc[ix,'Longitude'],d.loc[ix,'Latitude'],s=22,color=plt.cm.tab10(k),label=str(k+1))
axs[1].set_title('(b) Requested 10-cluster display: all 369 samples')
for ax in axs:
    ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)');ax.set_aspect(1/np.cos(np.deg2rad(d.Latitude.mean())));ax.ticklabel_format(useOffset=False);ax.legend(title='Cluster',fontsize=7,ncol=5,loc='upper center',bbox_to_anchor=(.5,-.17))
savefig(fig,'Figure_S1_spatial_clusters')
models={};probs={};rows=[];warnings_log=[]
for name in COLORS:
    print('Fitting holdout',name,flush=True)
    with warnings.catch_warnings(record=True) as w:
        m=build(name).fit(X.iloc[tr],y[tr]);warnings_log.extend(str(x.message) for x in w)
    models[name]=m;probs[name]=m.predict_proba(X.iloc[te])[:,1];rows.append(dict(Model=name,**metrics(y[te],probs[name])))
    joblib.dump(m,OUT/(name+'_holdout.joblib'))
hold=pd.DataFrame(rows);hold.to_csv(OUT/'holdout_metrics_reproduced.csv',index=False)
pred=assignment.loc[te].copy()
for n,p in probs.items():pred[n]=p
pred.to_csv(OUT/'holdout_predictions.csv',index=False)
print(hold.to_string(index=False),flush=True)
# Paired case bootstrap: identical sampled row IDs for every model and binning.
rng=np.random.default_rng(42); boot=rng.integers(0,len(te),(2000,len(te)))
bootrows=[];binrows=[];paired={}
for name,p in probs.items():
    for b in (5,10,15):
        vals=np.array([ece(y[te][ix],p[ix],b) for ix in boot]);low,hi=np.quantile(vals,[.025,.975])
        bootrows.append(dict(Model=name,Bins=b,ECE=ece(y[te],p,b),Bootstrap_median=np.median(vals),CI_low=low,CI_high=hi,Replicates=2000))
        if b==10:paired[name]=vals
    bi=np.digitize(p,np.linspace(0,1,11)[1:-1],right=True)
    for b in range(10):
        mask=bi==b;binrows.append(dict(Model=name,Bin=b+1,N=int(mask.sum()),Mean_probability=float(p[mask].mean()) if mask.any() else np.nan,Observed_fraction=float(y[te][mask].mean()) if mask.any() else np.nan))
pd.DataFrame(bootrows).to_csv(OUT/'ece_bootstrap_bins.csv',index=False)
pd.DataFrame(binrows).to_csv(OUT/'calibration_bins.csv',index=False)
diff=paired['GPC']-paired['RandomForest'];(OUT/'paired_ece_difference.json').write_text(json.dumps({'difference_GPC_minus_RF':ece(y[te],probs['GPC'])-ece(y[te],probs['RandomForest']),'CI_low':float(np.quantile(diff,.025)),'CI_high':float(np.quantile(diff,.975))},indent=2))
fig,axs=plt.subplots(2,2,figsize=(9,8),layout='constrained')
for ax,name in zip(axs.flat,COLORS):
    b=pd.DataFrame(binrows).query('Model==@name').dropna();ax.plot([0,1],[0,1],'--',color='.55');ax.plot(b.Mean_probability,b.Observed_fraction,'o-',color=COLORS[name]);
    for row in b.itertuples():ax.annotate(str(row.N),(row.Mean_probability,row.Observed_fraction),xytext=(3,5),textcoords='offset points',fontsize=8)
    ax.set(xlim=(-.02,1.02),ylim=(-.02,1.08),xlabel='Mean predicted probability',ylabel='Observed landslide fraction',title=f'{name} | ECE = {ece(y[te],probs[name]):.3f}')
savefig(fig,'Figure_3_reliability')
fig,axs=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
for name,p in probs.items():
    a,b,_=roc_curve(y[te],p);axs[0].plot(a,b,c=COLORS[name],label=f'{name} ({roc_auc_score(y[te],p):.3f})')
    pr,re,_=precision_recall_curve(y[te],p);axs[1].step(re,pr,where='post',c=COLORS[name],label=f'{name} ({average_precision_score(y[te],p):.3f})')
axs[0].plot([0,1],[0,1],'--',c='.6');axs[1].axhline(y[te].mean(),ls='--',c='.6')
axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='(a) ROC curves');axs[1].set(xlabel='Recall',ylabel='Precision',title='(b) Precision-recall curves')
for ax in axs:ax.legend(fontsize=8);ax.set(xlim=(0,1),ylim=(0,1.02))
savefig(fig,'Figure_2_discrimination')
# Sensitivity to retained negative locations: positive training points and holdout fixed.
sens=[];senspred=[];negtr=tr[y[tr]==0];postr=tr[y[tr]==1]
for seed in (7,21,84):
    keep=np.random.default_rng(seed).choice(negtr,size=int(np.floor(.8*len(negtr))),replace=False);sub=np.r_[postr,keep]
    for name in COLORS:
        print('Negative subset',seed,name,flush=True)
        with warnings.catch_warnings(record=True) as w:
            m=build(name).fit(X.iloc[sub],y[sub]);warnings_log.extend(str(x.message) for x in w)
        p=m.predict_proba(X.iloc[te])[:,1];allp=m.predict_proba(X)[:,1]
        sens.append(dict(Seed=seed,Model=name,Train_positive=len(postr),Train_negative=len(keep),**metrics(y[te],p),All_points_Spearman=float(spearmanr(models[name].predict_proba(X)[:,1],allp).statistic),All_points_MAE=float(np.mean(abs(models[name].predict_proba(X)[:,1]-allp)))))
        senspred.extend(dict(Seed=seed,Model=name,row_id=int(i),Probability=float(v)) for i,v in enumerate(allp))
pd.DataFrame(sens).to_csv(OUT/'negative_sampling_sensitivity.csv',index=False);pd.DataFrame(senspred).to_csv(OUT/'negative_sampling_point_predictions.csv',index=False)
# Paired coarse-variable ablations, with the same train/holdout split.
ablation=[]
for label,drop in [('Without_rainfall',['Precipitation']),('Without_soil',['SoilTexture']),('Without_both',['Precipitation','SoilTexture'])]:
    cols=[f for f in features if f not in drop]
    for name in COLORS:
        print('Ablation',label,name,flush=True)
        with warnings.catch_warnings(record=True) as w:
            m=build(name).fit(X.iloc[tr][cols],y[tr]);warnings_log.extend(str(x.message) for x in w)
        p=m.predict_proba(X.iloc[te][cols])[:,1]
        ablation.append(dict(Analysis=label,Model=name,**metrics(y[te],p)))
pd.DataFrame(ablation).to_csv(OUT/'coarse_predictor_ablation.csv',index=False)
# Full-data refit for point-level display only, without inventing a predictor raster.
print('Full-data GPC uncertainty',flush=True)
with warnings.catch_warnings(record=True) as w:
    full=build('GPC').fit(X,y);warnings_log.extend(str(x.message) for x in w)
joblib.dump(full,OUT/'GPC_full.joblib')
mu,var=full.named_steps['clf'].latent_mean_and_variance(full[:-1].transform(X));var=np.maximum(var,0)
nodes,weights=np.polynomial.hermite.hermgauss(64)
sig=expit(mu[:,None]+np.sqrt(2*var[:,None])*nodes);mean=sig@weights/np.sqrt(np.pi);second=(sig**2)@weights/np.sqrt(np.pi)
ep=np.maximum(second-mean**2,0);al=(sig*(1-sig))@weights/np.sqrt(np.pi)
assert np.allclose(ep+al,mean*(1-mean),atol=1e-12)
p=full.predict_proba(X)[:,1];std=np.sqrt(var);u=std/std.max()
out=d.copy();out['Probability']=p;out['Latent_mean']=mu;out['Latent_variance']=var;out['Normalized_latent_sd']=u;out['Quadrature_probability_mean']=mean;out['Probability_variance']=ep;out['Expected_Bernoulli_variance']=al
pb=np.quantile(p,[1/3,2/3]);ub=np.quantile(u,[1/3,2/3]);pc=np.digitize(p,pb);uc=np.digitize(u,ub)
out['Susceptibility_tertile']=pc+1;out['Uncertainty_tertile']=uc+1;out.to_csv(OUT/'corrected_gpc_point_uncertainty.csv',index=False)
(OUT/'uncertainty_metadata.json').write_text(json.dumps({'fitted_kernel':str(full.named_steps['clf'].kernel_),'probability_tertiles':pb.tolist(),'normalized_latent_sd_tertiles':ub.tolist(),'latent_sd_normalizer':float(std.max()),'max_quadrature_vs_sklearn_probability_difference':float(np.max(abs(mean-p))),'mapping_support':'369 observed sample locations; no continuous predictor rasters supplied'},indent=2))
fig,axs=plt.subplots(1,3,figsize=(12,4.6),layout='constrained')
for ax,vals,cmap,title in [(axs[0],p,'viridis','(a) Susceptibility score'),(axs[1],u,'magma','(b) Normalized latent standard deviation'),(axs[2],ep,'cividis','(c) Probability-scale posterior variance')]:
    sc=ax.scatter(d.Longitude,d.Latitude,c=vals,cmap=cmap,s=22,edgecolors='white',linewidths=.2);fig.colorbar(sc,ax=ax,shrink=.72);ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',title=title);ax.set_aspect(1/np.cos(np.deg2rad(d.Latitude.mean())));ax.ticklabel_format(useOffset=False)
savefig(fig,'Figure_4_corrected_uncertainty')
palette=np.array([['#e8e8e8','#ace4e4','#5ac8c8'],['#dfb0d6','#a5add3','#5698b9'],['#be64ac','#8c62aa','#3b4994']])
fig,axs=plt.subplots(1,2,figsize=(8.6,5),gridspec_kw={'width_ratios':[2,1]},layout='constrained')
axs[0].scatter(d.Longitude,d.Latitude,c=[palette[a,b] for a,b in zip(pc,uc)],s=30,edgecolors='.2',linewidths=.2);axs[0].set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',title='Susceptibility and latent uncertainty at sampled locations');axs[0].ticklabel_format(useOffset=False);axs[0].set_aspect(1/np.cos(np.deg2rad(d.Latitude.mean())))
from matplotlib.colors import to_rgb
axs[1].imshow(np.array([[to_rgb(c) for c in row] for row in palette]),origin='lower');axs[1].set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['Low','Middle','High'],yticklabels=['Low','Middle','High'],xlabel='Latent uncertainty tertile',ylabel='Susceptibility tertile');
for a in range(3):
 for b in range(3):axs[1].text(b,a,str(int(((pc==a)&(uc==b)).sum())),ha='center',va='center',color='white' if a==2 else 'black')
savefig(fig,'Figure_5_bivariate_points')
(OUT/'fit_warnings.txt').write_text('\n'.join(warnings_log),encoding='utf-8')
(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,'scipy':scipy.__version__},indent=2))
print('REVISION ANALYSIS COMPLETE',flush=True)
