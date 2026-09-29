from pathlib import Path
import hashlib,csv,json,shutil,ast
r=Path('/home/siddiqun/Bureau/Internship');o=Path('/home/siddiqun/Bureau/Conformal Library/internship_handover_audit')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def role(p):
 s=str(p.relative_to(r))
 if s in ['Data/metadata.csv','Data/phage_host_scores.parquet','Data/host_type_all_preds.csv','Codes/gram_type_predictions.csv']:return 'analysis_input'
 if s=='Data/all_random_pc_splits.pkl' or ('random_pc_r_5_da_2_bal_s_' in s and p.suffix=='.pkl'):return 'host_reconstruction_input'
 if 'processed_chunks/' in s:return 'intermediate_archive_separate_variants'
 if 'gram_type_predictions.csv' in s:return 'alternative_gram_export_do_not_substitute'
 if p.suffix=='.ipynb':return 'code_snapshot'
 return 'other_data_not_referenced_by_selected_pipeline'
files=[p for p in (r/'Data').rglob('*') if p.is_file() and not p.name.startswith('.~')]
notebooks=[p for p in (r/'Codes').rglob('*.ipynb') if '.venv' not in p.parts and '.ipynb_checkpoints' not in p.parts]+[r/'host_classification_library.ipynb']
files+=notebooks+[r/'Codes/gram_type_predictions.csv',r/'Codes/Gram Type Classification/gram_type_predictions.csv']
rows=[]
with (o/'file_manifest.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['relative_path','role','bytes','sha256']);writer.writeheader()
 for i,p in enumerate(sorted(files)):
  row={'relative_path':str(p.relative_to(r)),'role':role(p),'bytes':p.stat().st_size,'sha256':digest(p)}
  rows.append(row);writer.writerow(row);f.flush()
  if i%80==0 or p.stat().st_size>2e9:print(i,p.name,flush=True)
for p in notebooks:
 dest=o/'original_code'/p.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 n=json.loads(p.read_text());parts=[]
 for i,c in enumerate(n['cells']):
  if c['cell_type']=='code':parts.append(f'\n# ORIGINAL NOTEBOOK CELL {i} (zero-based)\n'+''.join(c['source']))
 dest=o/'code_text'/p.relative_to(r).with_suffix('.py');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('\n'.join(parts))
summary={}
for row in rows:
 g=summary.setdefault(row['role'],{'files':0,'bytes':0});g['files']+=1;g['bytes']+=row['bytes']
(o/'inventory_totals.json').write_text(json.dumps(summary,indent=2))
print('DONE',len(rows),'files',sum(row['bytes'] for row in rows),'bytes',flush=True)
