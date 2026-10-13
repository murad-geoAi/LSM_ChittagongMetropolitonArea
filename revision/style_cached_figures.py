"""Journal typography/layout pass; only cached outputs are used, no model refits."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent;A=ROOT/'analysis';F=A/'figures'
plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,'axes.titlesize':8.5,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
colors={'BLR':'#767676','GPC':'#0F4D92','LogReg_L2':'#42949E','RandomForest':'#B64342'}
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=600,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def geo(ax,d):
 ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)');ax.set_aspect(1/np.cos(np.deg2rad(d.Latitude.mean())));ax.ticklabel_format(useOffset=False);ax.set_xticks([91.775,91.825])
d=pd.read_csv(A/'spatial_assignments.csv');fig,axs=plt.subplots(1,2,figsize=(6.3,4.5),layout='constrained')
for ax,col,n in [(axs[0],'training_cluster',9),(axs[1],'all_sample_display_cluster',10)]:
 for k in range(n):
  q=d[d[col]==k+1];ax.scatter(q.Longitude,q.Latitude,s=10,color=plt.cm.tab10(k),label=str(k+1))
 geo(ax,d);ax.legend(title='Cluster',ncol=3,fontsize=6.5,title_fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False)
q=d[d.split=='holdout'];axs[0].scatter(q.Longitude,q.Latitude,marker='x',s=13,c='black',lw=.8,label='Holdout');axs[0].legend(title='Cluster',ncol=3,fontsize=6.5,title_fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False)
axs[0].set_title('(a) Actual validation design\nNine clusters and random holdout');axs[1].set_title('(b) All 369 sampled locations\nSeparate ten-cluster illustration');save(fig,'Figure_S1_spatial_clusters')
pred=pd.read_csv(A/'holdout_predictions.csv');y=pred.Landslide.to_numpy();hold=pd.read_csv(A/'holdout_metrics_reproduced.csv').set_index('Model')
fig,axs=plt.subplots(1,2,figsize=(6.3,2.75),layout='constrained')
for n,c in colors.items():
 p=pred[n];x,z,_=roc_curve(y,p);axs[0].plot(x,z,c=c,lw=1,label=f'{n} ({roc_auc_score(y,p):.3f})')
 pr,re,_=precision_recall_curve(y,p);axs[1].step(re,pr,where='post',c=c,lw=1,label=f'{n} ({average_precision_score(y,p):.3f})')
axs[0].plot([0,1],[0,1],'--',c='.6',lw=.8);axs[1].axhline(y.mean(),ls='--',c='.6',lw=.8)
axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='(a) ROC');axs[1].set(xlabel='Recall',ylabel='Precision',title='(b) Precision–recall')
for ax in axs:ax.legend(fontsize=6.5,frameon=False);ax.set(xlim=(0,1),ylim=(0,1.02))
save(fig,'Figure_2_discrimination')
bins=pd.read_csv(A/'calibration_bins.csv');fig,axs=plt.subplots(2,2,figsize=(6.3,5),layout='constrained')
for ax,(n,c) in zip(axs.flat,colors.items()):
 b=bins[bins.Model==n].dropna();ax.plot([0,1],[0,1],'--',c='.6',lw=.8);ax.plot(b.Mean_probability,b.Observed_fraction,'o-',c=c,ms=3,lw=.8)
 for r in b.itertuples():ax.annotate(str(r.N),(r.Mean_probability,r.Observed_fraction),xytext=(2,4),textcoords='offset points',fontsize=6.5)
 ax.set(xlim=(-.02,1.02),ylim=(-.02,1.1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title=f'{n}  |  ECE = {hold.loc[n,"ECE"]:.3f}')
save(fig,'Figure_3_reliability')
u=pd.read_csv(A/'corrected_gpc_point_uncertainty.csv');fig,axs=plt.subplots(1,3,figsize=(6.3,4.2),layout='constrained')
for ax,col,cmap,title in zip(axs,['Probability','Normalized_latent_sd','Probability_variance'],['viridis','magma','cividis'],['(a) Susceptibility\nscore','(b) Normalized\nlatent SD','(c) Posterior variance\nof probability']):
 sc=ax.scatter(u.Longitude,u.Latitude,c=u[col],cmap=cmap,s=9);geo(ax,u);ax.set_title(title);fig.colorbar(sc,ax=ax,orientation='horizontal',shrink=.9,pad=.02,aspect=15)
save(fig,'Figure_4_corrected_uncertainty')
palette=np.array([['#e8e8e8','#ace4e4','#5ac8c8'],['#dfb0d6','#a5add3','#5698b9'],['#be64ac','#8c62aa','#3b4994']]);pc=u.Susceptibility_tertile.to_numpy()-1;uc=u.Uncertainty_tertile.to_numpy()-1
fig,axs=plt.subplots(1,2,figsize=(6.3,4),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
axs[0].scatter(u.Longitude,u.Latitude,c=[palette[a,b] for a,b in zip(pc,uc)],s=13,edgecolors='.2',linewidths=.2);geo(axs[0],u);axs[0].set_title('Bivariate classes at observed locations')
axs[1].imshow(np.array([[to_rgb(c) for c in r] for r in palette]),origin='lower');axs[1].set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['Low','Middle','High'],yticklabels=['Low','Middle','High'],xlabel='Latent uncertainty tertile',ylabel='Susceptibility tertile')
for a in range(3):
 for b in range(3):axs[1].text(b,a,str(int(((pc==a)&(uc==b)).sum())),ha='center',va='center',color='white' if a==2 else 'black')
save(fig,'Figure_5_bivariate_points')
cv=pd.read_csv(ROOT.parent/'results/data/cv_metrics_summary.csv').set_index('Model');sp=pd.read_csv(A/'spatial_fold_metrics_reproduced.csv').groupby('Model')[['MCC','ROC_AUC']].mean();names=list(colors)
fig,axs=plt.subplots(1,2,figsize=(6.3,2.8),layout='constrained')
for ax,metric,col in zip(axs,['MCC','ROC_AUC'],['mcc_mean','roc_auc_mean']):
 a=np.arange(4);ax.bar(a-.18,cv.loc[names,col],width=.36,color='#9bc5df',edgecolor='.2',lw=.5,label='Repeated stratified');ax.bar(a+.18,sp.loc[names,metric],width=.36,color='#0F4D92',edgecolor='.2',lw=.5,label='Spatial groups');ax.set(xticks=a,xticklabels=['BLR','GPC','LR-L2','RF'],ylabel=metric.replace('_','-'),ylim=(0,1));ax.tick_params(axis='x',labelsize=8)
axs[0].legend(fontsize=7,frameon=False);save(fig,'Figure_S2_validation')
import shap
arr=np.load(A/'shap_values.npz');explain=pd.read_csv(A/'shap_explained.csv').drop(columns='row_id')
shap.summary_plot(arr['values'],explain,show=False,plot_size=(6.3,4.8))
for ax in plt.gcf().axes:ax.tick_params(labelsize=8);ax.xaxis.label.set_size(8.5);ax.yaxis.label.set_size(8.5)
plt.gca().set_xlabel('SHAP contribution to positive-class log-odds',fontsize=8.5)
save(plt.gcf(),'Figure_S3_shap')
print('Reformatted cached figures without refitting models')
