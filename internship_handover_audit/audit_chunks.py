from pathlib import Path
import pandas as pd,numpy as np,pyarrow.parquet as pq,json
r=Path('/home/siddiqun/Bureau/Internship/Data');out=Path('/home/siddiqun/Bureau/Conformal Library/internship_handover_audit')
ref=pd.read_parquet(r/'phage_host_scores.parquet')
cols=[ref.columns[i] for i in np.linspace(0,len(ref.columns)-1,5,dtype=int)]
results={}
for prefix in ['master','uncontaminated']:
 files=sorted((r/'processed_chunks').glob(prefix+'_chunk_*.parquet'))
 total=None;counts=None;rows=0
 for p in files:
  frame=pd.read_parquet(p,columns=cols)
  assert frame.index.name=='proteinID'
  access=frame.index.to_series().str.rsplit('_',n=1).str[0]
  sums=frame.groupby(access).sum();ns=frame.groupby(access).count()
  total=sums if total is None else total.add(sums,fill_value=0)
  counts=ns if counts is None else counts.add(ns,fill_value=0)
  rows+=len(frame)
 means=total/counts.replace(0,np.nan)
 common=means.index.intersection(ref.index)
 a=means.loc[common,cols].to_numpy();b=ref.loc[common,cols].to_numpy()
 nanmatch=np.isnan(a)==np.isnan(b);finite=np.isfinite(a)&np.isfinite(b)
 results[prefix]={'files':len(files),'protein_rows':rows,'aggregate_phages':len(means),'sampled_columns':cols,'common_phages':len(common),'sample_nan_mask_disagreements':int((~nanmatch).sum()),'sample_max_abs_difference':float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else None,'sample_allclose_1e-6':bool(np.allclose(a,b,rtol=1e-6,atol=1e-6,equal_nan=True))}
 print(prefix,results[prefix],flush=True)
(out/'chunk_comparison.json').write_text(json.dumps(results,indent=2))
