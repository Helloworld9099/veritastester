"""Auxiliary real-data idiomaticity research. Not translation training."""
import csv,json,hashlib,time
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
P=Path('data/external/semeval'); out=Path('results/real_idioms');out.mkdir(exist_ok=True)
a=list(csv.DictReader(open(P/'train_zero_shot.csv'))); b=list(csv.DictReader(open(P/'dev.csv'))); gold={r['ID']:r['Label']for r in csv.DictReader(open(P/'dev_gold.csv'))}
for r in b:r['Label']=gold[r['ID']]
group=lambda r:r['Language']+':'+r['MWE'].casefold()
assert not set(map(group,a)) & set(map(group,b))
# Remove exact normalized evaluation targets from training, preserving official data files.
norm=lambda s:' '.join(s.casefold().split())
seen={norm(r['Target'])for r in b};n0=len(a);a=[r for r in a if norm(r['Target'])not in seen]
idx,jdx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=2026).split(a,groups=list(map(group,a))))
train=[a[i]for i in idx];val=[a[i]for i in jdx]
y=lambda r:np.array([int(x['Label'])for x in r])
def metric(yy,p):return {'accuracy':float(accuracy_score(yy,p)),'macro_f1':float(f1_score(yy,p,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(yy,p))}
def text(r,ctx):return r['Target']+(' [PREVIOUS] '+r['Previous'] if ctx else '')
report={'task':'idiomaticity; source label 0=idiomatic and 1=literal; NOT translation','source_repository':'H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity','source_license':'repository GPL-3.0; source retained separately; no redistribution/deployment','train_n':len(a),'internal_train_n':len(train),'internal_validation_n':len(val),'evaluation_n':len(b),'removed_exact_cross_set_targets':n0-len(a),'evaluation_warning':'Official dev set used as evaluation, NOT official test. Some examples were used by prior reported pilot: not a globally untouched benchmark. No test-based tuning in this run.','models':{}}
started=time.monotonic();preds={}
for ctx in [False,True]:
 name='previous_context'if ctx else 'current_only'; trials=[];best=None
 for analyzer,ngram in [('word',(1,2)),('char_wb',(3,5))]:
  v=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,min_df=2,max_features=50000,sublinear_tf=True)
  x=v.fit_transform([text(r,ctx)for r in train]); xv=v.transform([text(r,ctx)for r in val])
  for c in [.1,1,10]:
   m=LogisticRegression(C=c,max_iter=1000,class_weight='balanced',random_state=2026);m.fit(x,y(train)); probs=m.predict_proba(xv)[:,1]
   for t in [.35,.5,.65]:
    score=f1_score(y(val),probs>=t,average='macro');trials.append({'analyzer':analyzer,'C':c,'threshold':t,'validation_macro_f1':float(score)})
    if best is None or score>best[0]:best=(score,analyzer,ngram,c,t)
 _,analyzer,ngram,c,t=best
 v=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,min_df=2,max_features=50000,sublinear_tf=True);x=v.fit_transform([text(r,ctx)for r in a]);m=LogisticRegression(C=c,max_iter=1000,class_weight='balanced',random_state=2026).fit(x,y(a))
 probs=m.predict_proba(v.transform([text(r,ctx)for r in b]))[:,1];pred=(probs>=t).astype(int);preds[name]=pred
 report['models'][name]={'selected':{'analyzer':analyzer,'ngram_range':ngram,'C':c,'threshold':t},'trials':trials,'evaluation':metric(y(b),pred),'by_language':{l:metric(y([r for r in b if r['Language']==l]),pred[np.array([r['Language']==l for r in b])])for l in ['EN','PT']}}
 artifact={'format':'research_tfidf_binary_v1_NOT_LIVE','context':ctx,'analyzer':analyzer,'ngram_range':ngram,'sublinear_tf':True,'norm':'l2','vocabulary':{k:int(j) for k,j in v.vocabulary_.items()},'idf':v.idf_.tolist(),'coef':m.coef_.tolist(),'intercept':m.intercept_.tolist(),'threshold':t,'classes':m.classes_.tolist(),'warning':'Auxiliary idiomaticity only; not integrated or validated for live use.'}
 (out/(name+'_weights.json')).write_text(json.dumps(artifact))
 print(name,report['models'][name]['evaluation'],flush=True)
rng=np.random.default_rng(2026);groups=sorted(set(map(group,b)));ix={g:np.array([i for i,r in enumerate(b)if group(r)==g])for g in groups};deltas=[]
for _ in range(1000):
 ids=np.concatenate([ix[g]for g in rng.choice(groups,len(groups),replace=True)]);deltas.append(f1_score(y(b)[ids],preds['previous_context'][ids],average='macro')-f1_score(y(b)[ids],preds['current_only'][ids],average='macro'))
report['context_delta_macro_f1']=report['models']['previous_context']['evaluation']['macro_f1']-report['models']['current_only']['evaluation']['macro_f1'];report['group_bootstrap_95pct_delta']=np.quantile(deltas,[.025,.975]).tolist();report['evaluation_groups']=len(groups);report['elapsed_seconds']=time.monotonic()-started;report['source_hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in P.glob('*.csv')};report['majority_baseline']=metric(y(b),np.full(len(b),Counter(y(a)).most_common(1)[0][0]));(out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='models'},indent=2))
