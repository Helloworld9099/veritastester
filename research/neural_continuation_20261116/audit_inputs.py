import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
read=lambda p:[json.loads(x)for x in p.read_text().splitlines()]
norm=lambda s:' '.join(s.casefold().split())
report={}
for lang,oldname in [('french','translation_fr_training_20261116'),('spanish','translation_training_20261116')]:
 p=ROOT/lang;old=BASE/oldname;d={s:read(p/(s+'.jsonl'))for s in['train','dev','test','guard']};oldrows=[r for s in['train','dev','test']for r in read(old/(s+'.jsonl'))]
 priorguard=read(old/('diabla_guard.jsonl'if lang=='french'else'wmt_guard_outputs.jsonl'))
 c={'counts':{s:len(rs)for s,rs in d.items()},'cross_current_partition_overlap':{},'overlap_with_original_fitting_and_validation':{},'overlap_with_previous_external_guard':{},'prepared_file_hashes_match_registration':{}}
 for a,b in[('train','dev'),('train','test'),('dev','test')]:
  for key in['source','reference','group']:
   c['cross_current_partition_overlap'][a+':'+b+':'+key]=len({norm(r[key])if key!='group'else r[key]for r in d[a]}&{norm(r[key])if key!='group'else r[key]for r in d[b]})
 for s,rs in d.items():
  for key in['source','reference']:
   c['overlap_with_original_fitting_and_validation'][s+':'+key]=len({norm(r[key])for r in rs}&{norm(r[key])for r in oldrows})
 for key in['source','reference']:
  c['overlap_with_previous_external_guard'][key]=len({norm(r[key])for r in d['guard']}&{norm(r[key])for r in priorguard})
 manifest=json.loads((p/'data_manifest.json').read_text())
 for name,sha in manifest['input_hashes'].items():c['prepared_file_hashes_match_registration'][name]=hashlib.sha256((p/name).read_bytes()).hexdigest()==sha
 protocol=json.loads((p/'protocol.json').read_text());c['previous_checkpoint_unchanged']=hashlib.sha256((old/'adapter_selected.safetensors').read_bytes()).hexdigest()==protocol['original_checkpoint_sha256']
 c['old_checkpoint_hashes']={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in old.glob('*.safetensors')}
 assert not any(c['cross_current_partition_overlap'].values());assert not any(c['overlap_with_original_fitting_and_validation'].values());assert all(c['prepared_file_hashes_match_registration'].values());assert c['previous_checkpoint_unchanged']
 report[lang]=c
(ROOT/'input_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
