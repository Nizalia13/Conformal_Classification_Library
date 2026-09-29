from pathlib import Path
import pandas as pd,pyarrow.parquet as pq,numpy as np,json,hashlib
r=Path('/home/siddiqun/Bureau/Internship');o=Path('/home/siddiqun/Bureau/Conformal Library/internship_handover_audit')
out={}
p=r/'Data/phage_host_scores.parquet';f=pq.ParquetFile(p)
out['parquet']={'rows':f.metadata.num_rows,'columns':f.metadata.num_columns,'row_groups':f.num_row_groups}
df=pd.read_parquet(p)
out['parquet'].update(shape=list(df.shape),index_unique=bool(df.index.is_unique),index_name=df.index.name,all_nan_columns=int(df.isna().all().sum()),null_fraction=float(df.isna().to_numpy().mean()),minimum=float(df.min().min()),maximum=float(df.max().max()))
print('PARQUET',out['parquet'],flush=True)
a=pd.read_csv(r/'Codes/gram_type_predictions.csv').set_index('accession')
b=pd.read_csv(r/'Codes/Gram Type Classification/gram_type_predictions.csv').set_index('accession')
common=a.index.intersection(b.index)
diff=(a.loc[common]!=b.loc[common]).any(axis=1)
out['gram_exports']={'host_consumed_rows':len(a),'gram_folder_rows':len(b),'common_ids':len(common),'different_predictions':int(diff.sum()),'host_only_ids':len(a.index.difference(b.index)),'gram_only_ids':len(b.index.difference(a.index)), 'host_combinations':{str(k):int(v) for k,v in a.value_counts().items()},'gram_combinations':{str(k):int(v) for k,v in b.value_counts().items()}}
pd.concat({'host_consumed':a.loc[common[diff]],'gram_folder':b.loc[common[diff]]},axis=1).to_csv(o/'gram_export_disagreements.csv')
print('GRAM',out['gram_exports'],flush=True)
headers=pd.read_csv(r/'Data/metadata.csv',nrows=0).columns.tolist();out['metadata_columns']=headers
m=pd.read_csv(r/'Data/metadata.csv',usecols=['proteinID','accession','host','host_type','split'])
out['metadata']={'rows':len(m),'unique_proteins':int(m.proteinID.nunique()),'unique_phages':int(m.accession.nunique()),'missing_ids':int(m.proteinID.isna().sum()),'conflicting_phages':int(m.groupby('accession')[['host','host_type','split']].nunique().gt(1).any(axis=1).sum()),'accession_from_id_disagreements':int((m.proteinID.str.rsplit('_',n=1).str[0]!=m.accession).sum())}
ph=m.groupby('accession')[['host','host_type','split']].first()
out['metadata']['phages_by_split']={str(k):int(v) for k,v in ph.split.value_counts().items()}
out['parquet']['ids_missing_metadata']=int((~df.index.isin(ph.index)).sum())
out['metadata']['phages_missing_score_table']=int((~ph.index.isin(df.index)).sum())
print('METADATA',out['metadata'],flush=True)
scorehead=pd.read_csv(r/'Data/host_type_all_preds.csv',nrows=0).columns.tolist()
out['gram_score_columns']=scorehead
rowcount=0;validids=True
for chunk in pd.read_csv(r/'Data/host_type_all_preds.csv',usecols=['proteinID'],chunksize=100000):
 rowcount+=len(chunk);validids=validids and chunk.proteinID.isin(m.proteinID).all()
out['gram_scores']={'rows':rowcount,'columns':len(scorehead),'ids_in_metadata':bool(validids)}
sets={k:{p.name for p in (r/'Data'/f'random_pc_r_5_da_2_bal_s_{k}').glob('all_preds_*.pkl')} for k in range(5)}
union=set.union(*sets.values());out['fold_files']={'union_count':len(union),'folds':{str(k):{'count':len(s),'missing_from_union':sorted(union-s)} for k,s in sets.items()}}
(o/'data_checks.json').write_text(json.dumps(out,indent=2))
print('FOLDS',out['fold_files'],flush=True)
