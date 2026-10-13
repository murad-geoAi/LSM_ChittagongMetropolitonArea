"""Package review files and a minimal reproducible supplement without QA intermediates."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json,shutil
R=Path(__file__).resolve().parent;S=R/'submission';root=R.parent
# Explicitly archive the legacy BLR source and required summaries at their expected paths.
files=[root/'__pycache__/bayesian_lsm_pipeline.py']
files += [root/'results/data'/n for n in ['LSM_Data.csv','model_report.json','holdout_metrics.csv','cv_metrics_summary.csv','spatial_cv_metrics.csv']]
files += [R/n for n in ['run_revision_analysis.py','run_revision_checks.py','style_cached_figures.py','requirements-revision.txt']]
files += list((R/'inputs').glob('*.csv'))
files += [p for p in (R/'analysis').iterdir() if p.suffix in ['.csv','.json','.npz','.txt']]
intro='Reproducibility supplement\n\nFrom this extracted folder run:\npython revision/run_revision_analysis.py\npython revision/run_revision_checks.py\npython revision/style_cached_figures.py\n\nThe scripts never call the archived pipeline main() or its cleanup functions. The source table encoding is preserved for transparent numerical reproduction. The repeated-CV summary is reused; holdout and spatial models are refitted. New spatial maps show observed locations, not complete predictor-raster inference. The source metadata and categorical-code limitations described in the manuscript remain.\n'
with ZipFile(S/'Reproducibility_Supplement.zip','w',ZIP_DEFLATED) as z:
 z.writestr('README.txt',intro)
 for p in files:z.write(p,p.relative_to(root).as_posix())
shutil.copy2(R/'README.md',S/'START_HERE_AUTHOR_REVIEW_REQUIRED.md')
manifest=[]
for p in sorted(S.rglob('*')):
 if p.is_file() and p.name!='checksums.json':manifest.append({'file':p.relative_to(S).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(S/'checksums.json').write_text(json.dumps(manifest,indent=2))
dest=R/'JMS_Revision_Package_AUTHOR_REVIEW_REQUIRED.zip'
with ZipFile(dest,'w',ZIP_DEFLATED) as z:
 for p in sorted(S.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(S).as_posix())
with ZipFile(dest) as z:assert z.testzip() is None
print(dest,round(dest.stat().st_size/1024**2,2),'MiB')
