import sys,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path[:0]=[str(BASE/'translation_training_20261116/deps'),str(BASE/'veritastester-localmt-deps'),str(BASE/'veritastester-validation-deps')]
from safetensors.torch import load_file
from sacrebleu.metrics import CHRF,BLEU
read=lambda p:[json.loads(x)for x in p.read_text().splitlines()]
all_checks={}
for lang,oldname in[('french','translation_fr_training_20261116'),('spanish','translation_training_20261116')]:
 p=ROOT/lang;old=BASE/oldname;report=json.loads((p/'report.json').read_text());proto=report['protocol'];sel=json.loads((p/'selected.json').read_text());history=report['dev_history'];candidates=history if lang=='french'else[r for r in history if r['candidate']!='previous'];expected=max(candidates,key=lambda r:r['metrics']['chrf'])['candidate']
 checks={'all_planned_updates_completed':report['completed_updates']==proto['recipe']['planned_updates'],'selection_matches_registered_dev_rule':expected==report['selected']==sel['candidate'],'checkpoint_replay_exact':report['selected_replay_exact'],'old_selected_checkpoint_unchanged':hashlib.sha256((old/'adapter_selected.safetensors').read_bytes()).hexdigest()==proto['original_checkpoint_sha256']}
 checkpoints={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in old.glob('*.safetensors')};oldregistered=json.loads((ROOT/'input_audit.json').read_text())[lang]['old_checkpoint_hashes'];checks['all_original_checkpoint_hashes_preserved']=checkpoints==oldregistered
 manifest=json.loads((old/'model/download_manifest.json').read_text());checks['all_original_base_files_unchanged']=all(hashlib.sha256((old/'model'/r['file']).read_bytes()).hexdigest()==r['sha256']for r in manifest)
 a=load_file(str(p/'adapter_selected.safetensors'));checks['finite_nonzero_selected_weights']=all(v.isfinite().all().item()for v in a.values())and any(v.abs().sum().item()>0 for k,v in a.items()if'lora_B'in k)
 for name,source in[('new_source_test','test'),('new_external_guard','guard')]:
  rs=read(p/(name+'_outputs.jsonl'));orig=read(p/(source+'.jsonl'));assert len(rs)==len(orig)
  checks[name+'_alignment_and_counts']=all(r['source']==o['source']and r['reference']==o['reference']and r['group']==o['group']for r,o in zip(rs,orig))
  checks[name+'_freeze_precedes_outputs']=(p/'selected.json').stat().st_mtime<=(p/(name+'_outputs.jsonl')).stat().st_mtime
  for arm in['base','previous','selected']:
   preds=[r[arm]for r in rs];refs=[r['reference']for r in rs];m=report['evaluations'][name][arm]
   checks[name+'_'+arm+'_metric_replay']=all(pred.strip()for pred in preds)and math.isclose(CHRF().corpus_score(preds,[refs]).score,m['chrf'],abs_tol=1e-10)and math.isclose(BLEU().corpus_score(preds,[refs]).score,m['bleu'],abs_tol=1e-10)
 assert all(checks.values()),checks
 all_checks[lang]=checks
p=ROOT/'french';add=json.loads((p/'strict_guard_addendum.json').read_text());assert(p/'strict_guard_addendum.json').stat().st_mtime<(p/'new_external_guard_outputs.jsonl').stat().st_mtime;all_checks['strict_dialogue_sensitivity_registered_before_outputs']=True
(ROOT/'verification.json').write_text(json.dumps(all_checks,indent=2));print(json.dumps(all_checks,indent=2))
