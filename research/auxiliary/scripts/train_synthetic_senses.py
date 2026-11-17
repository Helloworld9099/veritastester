"""Explicitly synthetic auxiliary-sense experiment; NEVER real-world evidence."""
import json
from pathlib import Path
from collections import defaultdict,Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
rows=[json.loads(x)for x in Path('data/synthetic_senses.jsonl').read_text().splitlines()];splits={s:[r for r in rows if r['split']==s]for s in ['train','dev','test']}
assert len({r['case_id']for r in rows})==len(rows)
g=defaultdict(set)
for r in rows:g[r['group_id']].add(r['split'])
assert all(len(s)==1 for s in g.values())
assert all('synthetic' in r['provenance'].lower()for r in rows)
report={'warning':'AI-authored synthetic train/dev/test, NOT independent human labels or real-world accuracy. Same eight expressions across splits. Not translation training.','counts':{s:len(rs)for s,rs in splits.items()},'arms':{}}
for view in ['current','context']:
 def text(r):return r['utterance']+(' '+r['previous']+' '+r['situation']if view=='context'else '')
 models={}
 for expr in sorted({r['expression']for r in rows}):
  train=[r for r in splits['train']if r['expression']==expr];v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True);x=v.fit_transform([text(r)for r in train]);m=LogisticRegression(C=1,max_iter=1000,random_state=2026).fit(x,[r['meaning']for r in train]);models[expr]=(v,m)
 def predict(rs):
  result=[]
  for r in rs:
   v,m=models[r['expression']];probs=m.predict_proba(v.transform([text(r)]))[0];i=probs.argmax();result.append((r['meaning'],m.classes_[i],float(probs[i])))
  return result
 dev=predict(splits['dev']);test=predict(splits['test'])
 # Select abstention to avoid incorrect suggestions on development only.
 thresholds=[.5,.55,.6,.7,.8,.9,1.0]
 threshold=max(thresholds,key=lambda t:(sum((1 if a==b else -3)for a,b,c in dev if c>=t),-t))
 selected=[(a,b,c)for a,b,c in test if c>=threshold]
 report['arms'][view]={'dev_accuracy':accuracy_score([a for a,b,c in dev],[b for a,b,c in dev]),'test_accuracy_no_abstention':accuracy_score([a for a,b,c in test],[b for a,b,c in test]),'threshold_selected_on_dev':threshold,'test_selected':len(selected),'test_selected_correct':sum(a==b for a,b,c in selected),'test_n':len(test),'test_predictions':[{'case_id':r['case_id'],'gold':a,'predicted':b,'confidence':c,'abstained':c<threshold}for r,(a,b,c)in zip(splits['test'],test)]}
 artifact={'format':'SYNTHETIC_RESEARCH_ONLY_NOT_LIVE','view':view,'threshold':threshold,'models':{e:{'vocabulary':v.vocabulary_,'idf':v.idf_.tolist(),'classes':m.classes_.tolist(),'coef':m.coef_.tolist(),'intercept':m.intercept_.tolist(),'ngram_range':[3,5],'analyzer':'char_wb','sublinear_tf':True}for e,(v,m)in models.items()}}
 Path('results/synthetic_'+view+'_weights.json').write_text(json.dumps(artifact))
Path('results/synthetic_senses_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
