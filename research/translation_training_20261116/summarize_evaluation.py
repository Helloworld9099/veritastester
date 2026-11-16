"""Post-freeze descriptive uncertainty; never selects or alters a checkpoint."""
import sys,json,hashlib,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'deps'),str(ROOT.parent/'veritastester-localmt-deps'),str(ROOT.parent/'veritastester-validation-deps')]
import numpy as np
from sacrebleu.metrics import CHRF

def process(root,files):
 report=json.loads((root/'report.json').read_text());out={}
 for name in files:
  rows=[json.loads(x)for x in(root/(name+'_outputs.jsonl')).read_text().splitlines()];metric=CHRF()
  delta=np.array([metric.sentence_score(r['selected'],[r['reference']]).score-metric.sentence_score(r['baseline'],[r['reference']]).score for r in rows])
  groups=[str(r.get('group_id',r.get('block',r.get('row',i))))for i,r in enumerate(rows)];unique=sorted(set(groups));indices={g:np.array([i for i,x in enumerate(groups)if x==g])for g in unique}
  rng=np.random.default_rng(20261116);boots=[]
  for _ in range(1000):
   ix=np.concatenate([indices[g]for g in rng.choice(unique,len(unique),replace=True)]);boots.append(delta[ix].mean())
  out[name]={'n':len(rows),'groups':len(unique),'mean_sentence_chrf_delta':float(delta.mean()),'group_bootstrap_95pct_mean_sentence_chrf_delta':np.quantile(boots,[.025,.975]).tolist(),'warning':'Sentence chrF mean/interval is a different aggregation from corpus chrF. Group means reflect available block/component/dialogue groups, not proven independent speakers. No tuning based on these results.','changed_outputs':sum(r['selected']!=r['baseline']for r in rows),'empty_baseline':sum(not r['baseline'].strip()for r in rows),'empty_selected':sum(not r['selected'].strip()for r in rows),'corpus_chrf_delta':report[name]['selected']['chrf']-report[name]['baseline']['chrf'],'corpus_bleu_delta':report[name]['selected']['bleu']-report[name]['baseline']['bleu']}
 (root/'uncertainty.json').write_text(json.dumps(out,indent=2));print(root.name,json.dumps(out,indent=2))
if __name__=='__main__':
 process(ROOT,['europarl_test','wmt_guard'])
 process(ROOT.parent/'translation_fr_training_20261116',['tatoeba_test','diabla_guard'])
