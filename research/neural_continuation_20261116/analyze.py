"""Post-selection paired corpus-chrF analysis; does not choose models."""
import sys,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path[:0]=[str(BASE/'translation_training_20261116/deps'),str(BASE/'veritastester-localmt-deps'),str(BASE/'veritastester-validation-deps')]
import numpy as np
from sacrebleu.metrics import CHRF,BLEU
read=lambda p:[json.loads(x)for x in p.read_text().splitlines()]
def analyze(rs,known=None):
 m=CHRF();refs=[r['reference']for r in rs];stats={};scores={}
 for arm in['base','previous','selected']:
  texts=[r[arm]for r in rs];assert all(x.strip()for x in texts)
  stats[arm]=np.asarray(m._extract_corpus_statistics(texts,[refs]),dtype=np.int64)
  c=m._compute_score_from_stats(stats[arm].sum(axis=0).tolist()).score
  public=CHRF().corpus_score(texts,[refs]).score;assert math.isclose(c,public,abs_tol=1e-10)
  b=BLEU().corpus_score(texts,[refs]).score
  if known:assert math.isclose(c,known[arm]['chrf'],abs_tol=1e-10)and math.isclose(b,known[arm]['bleu'],abs_tol=1e-10)
  scores[arm]=dict(chrf=c,bleu=b)
 groups=sorted({str(r['group'])for r in rs});ix={g:np.array([i for i,r in enumerate(rs)if str(r['group'])==g])for g in groups};rng=np.random.default_rng(202611162);draws=[]
 for _ in range(1000):draws.append(np.concatenate([ix[g]for g in rng.choice(groups,len(groups),replace=True)]))
 comparisons={}
 for a,b in[('selected','base'),('selected','previous'),('previous','base')]:
  diffs=[m._compute_score_from_stats(stats[a][ids].sum(axis=0).tolist()).score-m._compute_score_from_stats(stats[b][ids].sum(axis=0).tolist()).score for ids in draws]
  comparisons[a+'_minus_'+b]=dict(corpus_chrf_delta=scores[a]['chrf']-scores[b]['chrf'],corpus_bleu_delta=scores[a]['bleu']-scores[b]['bleu'],group_bootstrap_95pct_corpus_chrf_delta=np.quantile(diffs,[.025,.975]).tolist(),changed_outputs=sum(r[a]!=r[b]for r in rs))
 return dict(n=len(rs),groups=len(groups),scores=scores,comparisons=comparisons,warning='Conditional paired group bootstrap, 1000 resamples, descriptive intervals without multiple-comparison adjustment. Group identifiers do not prove independent authors or unknown pretraining. Pinned sacrebleu 2.6.0 private sufficient-statistic methods reproduce public corpus scores exactly before use.')
def main():
 reports={}
 for lang in['french','spanish']:
  p=ROOT/lang;r=json.loads((p/'report.json').read_text());result={}
  for name in['new_source_test','new_external_guard']:
   result[name]=analyze(read(p/(name+'_outputs.jsonl')),r['evaluations'][name])
  if lang=='french':
   add=json.loads((p/'strict_guard_addendum.json').read_text());assert hashlib.sha256((p/'guard.jsonl').read_bytes()).hexdigest()==add['guard_sha256'];keep=set(add['retained_case_ids']);rs=[x for x in read(p/'new_external_guard_outputs.jsonl')if x['case_id']in keep];assert len(rs)==add['retained_n'];result['strict_new_external_guard']=analyze(rs)
  (p/'analysis.json').write_text(json.dumps(result,indent=2));reports[lang]=result
 print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
