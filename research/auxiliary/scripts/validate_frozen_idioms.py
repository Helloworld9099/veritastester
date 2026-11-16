"""No fitting or tuning: additional external examples + context perturbation."""
import csv,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
from predict_idiom_research import predict
p=Path('data/external/semeval');train=list(csv.DictReader(open(p/'train_zero_shot.csv')));new=list(csv.DictReader(open(p/'train_one_shot.csv')))
normalize=lambda x:' '.join(x.casefold().split())
traintexts={normalize(r['Target'])for r in train};new=[r for r in new if normalize(r['Target'])not in traintexts]
report={'warning':'Frozen models evaluated on additional one-shot training-source examples. Not official test; no further fitting, threshold changes or selection. Do not tune future models on these results.','new_evaluation_n':len(new),'models':{}}
def metrics(ys,p):return {'accuracy':float(accuracy_score(ys,p)),'macro_f1':float(f1_score(ys,p,average='macro')),'balanced_accuracy':float(balanced_accuracy_score(ys,p))}
for n in ['current_only','previous_context']:
 pred,probs=predict('results/real_idioms/'+n+'_weights.json',[r['Target']for r in new],[r['Previous']for r in new]);report['models'][n]=metrics([int(r['Label'])for r in new],pred)
report['group_overlap_with_training']=len({(r['Language'],r['MWE'])for r in new}&{(r['Language'],r['MWE'])for r in train})
# Descriptive perturbation test on already used 739-row dev, not new validation.
rows=list(csv.DictReader(open(p/'dev.csv')));gold={r['ID']:int(r['Label'])for r in csv.DictReader(open(p/'dev_gold.csv'))};ys=[gold[r['ID']]for r in rows];targets=[r['Target']for r in rows];prev=[r['Previous']for r in rows];rng=np.random.default_rng(2026);shuffled=prev.copy()
for lang in ['EN','PT']:
 ids=[i for i,r in enumerate(rows)if r['Language']==lang];order=rng.permutation(ids)
 for i,j in zip(ids,order):shuffled[i]=prev[j]
pert={}
for n,p in [('original',prev),('removed',['']*len(rows)),('shuffled_within_language',shuffled)]:
 pred,probs=predict('results/real_idioms/previous_context_weights.json',targets,p);pert[n]=metrics(ys,pred)
report['already_used_dev_context_perturbation']=pert
Path('results/real_idioms/frozen_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
