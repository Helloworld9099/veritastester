"""Independent saved-evidence audit: no training or checkpoint selection."""
from pathlib import Path
import json,sys,hashlib,math,unicodedata,tarfile,itertools
ROOT=Path('/workspace/scratch/2b30092347a7')
sys.path.append(str(ROOT/'veritastester-validation-deps'))
from sacrebleu.metrics import CHRF,BLEU
OUT=Path(__file__).resolve().parent
norm=lambda x:' '.join(unicodedata.normalize('NFKC',x).casefold().split())
def read(path):return list(map(json.loads,path.read_text().splitlines()))
report={}
for lang,p,evals in [('es',ROOT/'translation_training_20261116',['europarl_test','wmt_guard']),('fr',ROOT/'translation_fr_training_20261116',['tatoeba_test','diabla_guard'])]:
 splits={s:read(p/(s+'.jsonl'))for s in ['train','dev','test']};source=json.loads((p/'report.json').read_text());sel=json.loads((p/'selected.json').read_text());dev=json.loads((p/'dev_history.json').read_text());logs=json.loads((p/'training_log.json').read_text());check={'split_counts':{s:len(rows)for s,rows in splits.items()},'overlaps':{},'evaluation':{}}
 for a,b in itertools.combinations(splits,2):
  for key in ['source','reference','block']:
   overlap=set(norm(str(x[key]))for x in splits[a])&set(norm(str(x[key]))for x in splits[b]);check['overlaps'][f'{a}/{b}/{key}']=len(overlap);assert not overlap
 assert len(logs)==128 and [x['step']for x in logs]==list(range(1,129))
 assert all(math.isfinite(x['loss'])and math.isfinite(x['gradient_norm'])for x in logs)
 assert sel['step']==max(dev,key=lambda x:x['metrics']['chrf'])['step']==128
 check['selection_consistent_with_dev']=True
 check['selected_hash_matches_step128']=hashlib.sha256((p/'adapter_selected.safetensors').read_bytes()).digest()==hashlib.sha256((p/'adapter_step128.safetensors').read_bytes()).digest();assert check['selected_hash_matches_step128']
 manifest=json.loads((p/'model/download_manifest.json').read_text())
 check['base_hashes_match_manifest']=all(hashlib.sha256((p/'model'/x['file']).read_bytes()).hexdigest()==x['sha256']for x in manifest);assert check['base_hashes_match_manifest']
 for name in evals:
  rows=read(p/(name+'_outputs.jsonl'));r={'n':len(rows)}
  for arm in ['baseline','selected']:
   pred=[x[arm]for x in rows];refs=[x['reference']for x in rows]
   r[arm]={'chrf':CHRF().corpus_score(pred,[refs]).score,'bleu':BLEU().corpus_score(pred,[refs]).score}
   assert len(rows)==source[name][arm]['n'] and all(x.strip()for x in pred)
   for metric,value in r[arm].items():assert math.isclose(value,source[name][arm][metric],abs_tol=1e-10)
  check['evaluation'][name]=r
  if name in ['europarl_test','tatoeba_test']:
   assert [(x['source'],x['reference'])for x in rows]==[(x['source'],x['reference'])for x in splits['test']]
 if lang=='es':
  pairs=[(abs(a['line']-b['line']),sa,sb)for sa,sb in itertools.combinations(splits,2)for a in splits[sa]for b in splits[sb]]
  check['minimum_selected_cross_split_line_distance']=min(x[0]for x in pairs)
  new=json.loads((p/'wmt_guard_selection.json').read_text());old=json.loads((ROOT/'veritastester-independent-validation/independent_validation/wmt_selection.json').read_text())
  assert not set(new['indices'])&set(old['row_indices_zero_based']);check['wmt_previous_indices_overlap']=0
  rows=read(p/'wmt_guard_outputs.jsonl');alltexts={norm(r['source'])for rr in splits.values()for r in rr};assert not alltexts&{norm(x['source'])for x in rows};check['wmt_exact_source_overlap_pilot']=0
 else:
  guard=read(p/'diabla_guard.jsonl');smoke=read(ROOT/'veritastester-data-prep/private/diabla.fr-en.smoke.inputs.test.jsonl')
  assert not{x['group_id']for x in guard}&{x['group_id']for x in smoke}
  check['diabla_smoke_dialogue_overlap']=0;check['diabla_guard_dialogues']=len({x['group_id']for x in guard})
  orig=read(ROOT/'veritastester-data-prep/private/diabla.fr-en.inputs.test.jsonl');gold=read(ROOT/'veritastester-data-prep/private/diabla.fr-en.references.test.jsonl');protected={norm(x['utterance'])for x in orig}|{norm(x['reference_translation'])for x in gold}
  for s,rows in splits.items():
   assert not{x for r in rows for x in [norm(r['source']),norm(r['reference'])]}&protected
  check['pilot_exact_text_overlap_full_DiaBLa']=0
 report[lang]=check
archive=ROOT/'veritastester-neural-translation-pilots-20261116.tar.gz';manifest=json.loads((ROOT/'translation_training_20261116/DELIVERY_MANIFEST.json').read_text())
with tarfile.open(archive)as tf:
 assert all(not Path(n).is_absolute()and '..'not in Path(n).parts for n in tf.getnames())
 assert not any(n.endswith('.jsonl')for n in tf.getnames())
 for name,info in manifest.items():assert hashlib.sha256(tf.extractfile(name).read()).hexdigest()==info['sha256']
 report['archive']={'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'manifest_entries':len(manifest),'all_manifest_hashes_match':True,'no_raw_jsonl':True,'safe_paths':True}
(OUT/'saved_evidence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
