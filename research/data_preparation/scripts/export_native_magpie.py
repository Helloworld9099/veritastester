"""Export existing strictly split MAGPIE data into BridgeCaption's native schema."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
outputs=[]
for split in ('train','dev','test'):
 rows=[]
 for line in (P/f'private/magpie.{split}.jsonl').read_text().splitlines():
  r=json.loads(line)
  rows.append({'schema':'bridgecaption_corpus_v1','dataset':'magpie','role':'idiomaticity',
   'case_id':r['case_id'],'group_id':r['group_id'],'split':split,'original_split':split,
   'language':'en','utterance':r['utterance'],'previous':' '.join(r['previous']),
   'situation':'','expression':r['expression'],'label_kind':'idiomaticity_not_exact_sense',
   'source_label':'idiomatic' if r['label']==1 else 'literal','reference_translation':'',
   'candidate_definition':'','provenance':r['provenance'],
   'rights_status':'cc-by-4.0-publisher-license-retained','source_sha256':r['source_sha256'],
   'source_filename':'MAGPIE_filtered_split_typebased.jsonl',
   'extra':{'document_id':r['document_id'],'target_offsets':r['target_offsets'],
            'surface_form':r['surface_form'],'target_alignment':r['target_alignment']}})
 q=P/f'private/magpie.normalized.{split}.jsonl'
 q.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows));q.chmod(0o600)
 outputs.append({'file':str(q.relative_to(P)),'count':len(rows),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
(P/'reports/native_adapter.json').write_text(json.dumps({'outputs':outputs,'truncation':False,'automatic_training':False},indent=2))
print('Exported native MAGPIE schema without truncation or altered targets.')
