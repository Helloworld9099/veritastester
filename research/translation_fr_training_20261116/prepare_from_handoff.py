import json,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;DATA=ROOT.parent/'veritastester-data-prep/private'
rng=random.Random(20261117);manifest={'language':'fr-en','seed':20261117,'source_handoff_sha256':hashlib.sha256((DATA.parent/'HANDOFF_MANIFEST.json').read_bytes()).hexdigest(),'warning':'Only pre-existing train/dev/test boundaries used; no DiaBLa training. Tatoeba attribution/redistribution limitations retained.','counts':{},'files':{}}
for split,n,file in [('train',1024,'tatoeba.fr-en.train.quick12000.jsonl'),('dev',128,'tatoeba.fr-en.dev.jsonl'),('test',256,'tatoeba.fr-en.test.jsonl')]:
 rows=[json.loads(x)for x in (DATA/file).read_text().splitlines()];rows=[r for r in rows if 3<=len(r['utterance'].split())<=32 and 3<=len(r['reference_translation'].split())<=36];chosen=rng.sample(rows,n)
 out=[dict(source=r['utterance'],reference=r['reference_translation'],block=r['group_id'],case_id=r['case_id'],attribution=r['attribution'],license=r['license'],provenance=r['provenance'])for r in chosen]
 (ROOT/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in out));manifest['counts'][split]=n;manifest['files'][file]=hashlib.sha256((DATA/file).read_bytes()).hexdigest()
inputs=[json.loads(x)for x in (DATA/'diabla.fr-en.inputs.test.jsonl').read_text().splitlines()]
smoke=[json.loads(x)for x in (DATA/'diabla.fr-en.smoke.inputs.test.jsonl').read_text().splitlines()];usedgroups={r['group_id']for r in smoke}
bygroup={}
for r in inputs:
 if r['group_id']not in usedgroups:bygroup.setdefault(r['group_id'],[]).append(r)
selectedgroups=rng.sample(sorted(g for g,rs in bygroup.items()if len(rs)>=2),64)
chosen=[r for g in selectedgroups for r in rng.sample(bygroup[g],2)]
refs={r['case_id']:r for r in map(json.loads,(DATA/'diabla.fr-en.references.test.jsonl').read_text().splitlines())}
( ROOT/'diabla_guard.jsonl').write_text(''.join(json.dumps(dict(source=r['utterance'],reference=refs[r['case_id']]['reference_translation'],case_id=r['case_id'],group_id=r['group_id'],license=r['license'],provenance=r['provenance']),ensure_ascii=False)+'\n'for r in chosen))
manifest['diabla_guard']={'rows':len(chosen),'dialogues':len(selectedgroups),'excluded_smoke_dialogues':len(usedgroups),'selection':'random64 untouched-from-smoke dialogues, random2 turns each; no history, role or references sent as source input','selected_case_ids':[r['case_id']for r in chosen]}
(ROOT/'data_manifest.json').write_text(json.dumps(manifest,indent=2));print(manifest['counts'],manifest['diabla_guard']['rows'])
