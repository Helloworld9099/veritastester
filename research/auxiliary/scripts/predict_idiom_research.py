"""Load the safe JSON research artifact; outputs source label 0/1, not translation."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def predict(path, targets, previous):
 m=json.loads(Path(path).read_text());assert m['format']=='research_tfidf_binary_v1_NOT_LIVE'
 v=TfidfVectorizer(analyzer=m['analyzer'],ngram_range=tuple(m['ngram_range']),sublinear_tf=m['sublinear_tf'],vocabulary=m['vocabulary']);v.fit(['initialization']);v.idf_=np.array(m['idf'])
 texts=[t+(' [PREVIOUS] '+p if m['context'] else '') for t,p in zip(targets,previous)]
 z=np.asarray(v.transform(texts)@np.array(m['coef']).T).ravel()+m['intercept'][0]
 probs=1/(1+np.exp(-np.clip(z,-700,700)));return (probs>=m['threshold']).astype(int),probs
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--text',required=True);p.add_argument('--previous',default='');a=p.parse_args();y,probs=predict(a.model,[a.text],[a.previous]);print(json.dumps({'label':int(y[0]),'literal_probability_uncalibrated':float(probs[0]),'warning':'Research idiomaticity detector, not translation or production confidence.'}))
