import urllib.request,json,hashlib,tarfile,io,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URL='https://www.statmt.org/europarl/v7/es-en.tgz'
p=ROOT/'es-en.tgz'
if not p.exists():
 with urllib.request.urlopen(URL,timeout=120)as r,p.open('wb')as o:
  while b:=r.read(1024*1024):o.write(b)
print('Downloaded',p.stat().st_size,flush=True)
with tarfile.open(p,'r:gz')as tar:
 names=tar.getnames();print(names,flush=True)
 src=tar.extractfile('europarl-v7.es-en.es').read().decode().splitlines()
 ref=tar.extractfile('europarl-v7.es-en.en').read().decode().splitlines()
assert len(src)==len(ref)
# Contiguous 256-line blocks prevent immediate adjacent utterances crossing splits.
# This is block grouping, NOT verified document/speaker grouping; preserve this limitation.
rng=random.Random(20261116);blocks=sorted(set(i//256 for i in range(len(src))));rng.shuffle(blocks)
N=len(blocks);bins={'train':set(blocks[:int(.8*N)]),'dev':set(blocks[int(.8*N):int(.9*N)]),'test':set(blocks[int(.9*N):])}
limits={'train':1024,'dev':128,'test':256};seen=set();eligible={k:[]for k in bins}
for i,(s,r)in enumerate(zip(src,ref)):
 if not(4<=len(s.split())<=32 and 4<=len(r.split())<=36):continue
 if not .4<=len(s)/max(1,len(r))<=2.5:continue
 key=' '.join(s.casefold().split())
 if key in seen:continue
 seen.add(key)
 split=next(k for k,v in bins.items()if i//256 in v)
 eligible[split].append({'line':i,'block':i//256,'source':s,'reference':r,'split':split,'provenance':'Europarl v7 human parliamentary parallel corpus; not synthetic; document/speaker identity unavailable in aligned text'})
chosen={}
for k,rs in eligible.items():
 chosen[k]=rng.sample(rs,limits[k]);(ROOT/(k+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in chosen[k]))
assert all(not{r['block']for r in chosen[a]}&{r['block']for r in chosen[b]}for a,b in [('train','dev'),('train','test'),('dev','test')])
manifest={'dataset':'Europarl v7 Spanish-English','url':URL,'archive_bytes':p.stat().st_size,'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_rows':len(src),'seed':20261116,'selection':'unique normalized source, 4-32/4-36 words, length ratio .4-2.5; grouped by contiguous 256-line blocks, not verified document/speaker disjoint','counts':{k:len(v)for k,v in chosen.items()},'eligible_counts':{k:len(v)for k,v in eligible.items()},'copyright_source':'https://www.statmt.org/europarl/','copyright_note':'Corpus author states no known copyright restrictions and asks research users to cite Koehn (2005). Internal research only; no raw corpus redistribution or commercial clearance claimed.','attribution':'Philipp Koehn. 2005. Europarl: A Parallel Corpus for Statistical Machine Translation. MT Summit. Proceedings of the European Parliament.','pretraining_warning':'OPUS base may already have seen Europarl. Split is held out from our fine-tuning only, not guaranteed unknown to pretraining.','file_hashes':{k:hashlib.sha256((ROOT/(k+'.jsonl')).read_bytes()).hexdigest()for k in chosen}}
(ROOT/'data_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
