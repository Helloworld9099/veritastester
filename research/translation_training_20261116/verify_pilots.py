import sys,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'veritastester-localmt-deps'),str(ROOT.parent/'veritastester-validation-deps')]
from safetensors import safe_open
from sacrebleu.metrics import CHRF,BLEU
checks={}
for p,testnames in [(ROOT,['europarl_test','wmt_guard']),(ROOT.parent/'translation_fr_training_20261116',['tatoeba_test','diabla_guard'])]:
 c={};report=json.loads((p/'report.json').read_text());manifest=json.loads((p/'model/download_manifest.json').read_text())
 c['pinned_base_files_unchanged']=all(hashlib.sha256((p/'model'/r['file']).read_bytes()).hexdigest()==r['sha256']for r in manifest)
 data={s:[json.loads(x)for x in(p/(s+'.jsonl')).read_text().splitlines()]for s in ['train','dev','test']}
 norm=lambda s:' '.join(s.casefold().split())
 c['no_source_reference_or_group_overlap']=all(not{norm(r[key])if key!='block'else r[key]for r in data[a]}&{norm(r[key])if key!='block'else r[key]for r in data[b]}for a,b in [('train','dev'),('train','test'),('dev','test')]for key in ['source','reference','block'])
 logs=json.loads((p/'training_log.json').read_text());c['all_128_updates_completed']=len(logs)==128 and [r['step']for r in logs]==list(range(1,129)) and report['steps_completed']==128
 with safe_open(str(p/'adapter_selected.safetensors'),framework='pt')as f:
  c['finite_nonzero_adapter']=all(f.get_tensor(k).isfinite().all().item()for k in f.keys())and any(f.get_tensor(k).abs().sum().item()>0 for k in f.keys()if 'lora_B'in k)
 c['exact_initial_backbone_and_selected_replay']=report['initial_adapter_exact_backbone_logits']and report['selected_adapter_replay_exact']
 c['output_counts_and_metrics_reproduce']=True
 for n in testnames:
  rows=[json.loads(x)for x in(p/(n+'_outputs.jsonl')).read_text().splitlines()]
  for arm in ['baseline','selected']:
   pred=[r[arm]for r in rows];ref=[r['reference']for r in rows]
   assert len(rows)==report[n][arm]['n']and all(x.strip()for x in pred)
   assert math.isclose(CHRF().corpus_score(pred,[ref]).score,report[n][arm]['chrf'],abs_tol=1e-10)
   assert math.isclose(BLEU().corpus_score(pred,[ref]).score,report[n][arm]['bleu'],abs_tol=1e-10)
 assert all(c.values()),c
 checks[p.name]=c
old=json.loads((ROOT.parent/'veritastester-independent-validation/independent_validation/wmt_selection.json').read_text())
new=json.loads((ROOT/'wmt_guard_selection.json').read_text());assert not set(new['indices'])&set(old['row_indices_zero_based']);checks['wmt_old_guard_disjoint']=True
fr=ROOT.parent/'translation_fr_training_20261116';smoke=[json.loads(x)for x in(ROOT.parent/'veritastester-data-prep/private/diabla.fr-en.smoke.inputs.test.jsonl').read_text().splitlines()]
guard=[json.loads(x)for x in(fr/'diabla_guard.jsonl').read_text().splitlines()];assert not{r['group_id']for r in smoke}&{r['group_id']for r in guard};checks['diabla_smoke_dialogues_excluded']=True
assert json.loads((fr/'replay_smoke.json').read_text())['output']=="I don't want to cancel the reservation."
checks['fresh_process_french_reload_produces_translation']=True
(ROOT/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
