"""Offline synthetic-topic experiment; never trains the translation model.
Tune only inside the original training split, evaluate original holdout once.
"""
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,f1_score,classification_report
from bridgecaption.training import holdout

rows=[(r['text'],r['topic']) for r in csv.DictReader(open('data/topic_train.csv'))]
train,test=holdout(rows)
x,y=zip(*train); xt,yt=zip(*test)
candidates=[]; started=time.monotonic()
for analyzer,ngram in [('word',(1,2)),('char_wb',(3,5))]:
 for c in [.1,1,10,100]:
  p=Pipeline([('v',TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,sublinear_tf=True)),('m',LogisticRegression(C=c,max_iter=2000,random_state=2026))])
  scores=cross_val_score(p,x,y,cv=StratifiedKFold(4,shuffle=True,random_state=2026),scoring='f1_macro')
  candidates.append((float(scores.mean()),analyzer,ngram,c,p,scores.tolist()))
best=max(candidates,key=lambda z:z[0]); p=best[4].fit(x,y); pred=p.predict(xt)
report={'warning':'Synthetic English topic hints only; NOT translation or real-world evaluation. Small 24-row holdout; no deployment.', 'source_sha256':hashlib.sha256(Path('data/topic_train.csv').read_bytes()).hexdigest(),'train_n':len(train),'test_n':len(test),'selection':'4-fold stratified CV within original 96 training examples only','candidates':[{'cv_macro_f1':a,'analyzer':b,'ngram':c,'C':d,'folds':f}for a,b,c,d,e,f in candidates],'selected':{'analyzer':best[1],'ngram':best[2],'C':best[3]},'test_accuracy':accuracy_score(yt,pred),'test_macro_f1':f1_score(yt,pred,average='macro'),'per_topic':classification_report(yt,pred,output_dict=True,zero_division=0),'seconds':time.monotonic()-started}
v=p['v']; m=p['m']; artifact={'format':'tfidf_logistic_topic_candidate_v1','trained_on_n':len(train),'analyzer':best[1],'ngram_range':best[2],'sublinear_tf':True,'norm':'l2','lowercase':True,'vocabulary':v.vocabulary_,'idf':v.idf_.tolist(),'classes':m.classes_.tolist(),'coef':m.coef_.tolist(),'intercept':m.intercept_.tolist(),'warning':report['warning']}
Path('results/topic_candidate_metrics.json').write_text(json.dumps(report,indent=2)); Path('results/topic_candidate_weights.json').write_text(json.dumps(artifact))
print(json.dumps(report,indent=2))
