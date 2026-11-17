import json,random,hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;DATA=BASE/'veritastester-data-prep/private';norm=lambda s:' '.join(s.casefold().split())
def rows(p):return[json.loads(x)for x in p.read_text().splitlines()]
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def lines(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n'for r in rs))
seed=202611162
# French: preserve all source train/dev/test boundaries, exclude previously used components.
p=ROOT/'french';old=BASE/'translation_fr_training_20261116';rng=random.Random(seed)
for split,n,f in [('train',4096,'tatoeba.fr-en.train.jsonl'),('dev',256,'tatoeba.fr-en.dev.jsonl'),('test',512,'tatoeba.fr-en.test.jsonl')]:
 used={r['block']for r in rows(old/(split+'.jsonl'))}
 available=[r for r in rows(DATA/f)if r['group_id']not in used and 3<=len(r['utterance'].split())<=32 and 3<=len(r['reference_translation'].split())<=36]
 chosen=rng.sample(available,n)
 out=[dict(source=r['utterance'],reference=r['reference_translation'],group=r['group_id'],case_id=r['case_id'],provenance=r['provenance'],license=r['license'],attribution=r['attribution'])for r in chosen]
 lines(p/(split+'.jsonl'),out)
used={r['group_id']for r in rows(old/'diabla_guard.jsonl')}|{r['group_id']for r in rows(DATA/'diabla.fr-en.smoke.inputs.test.jsonl')}
available=[r for r in rows(DATA/'diabla.fr-en.inputs.test.jsonl')if r['group_id']not in used]
bygroup={}
for r in available:bygroup.setdefault(r['group_id'],[]).append(r)
refs={r['case_id']:r['reference_translation']for r in rows(DATA/'diabla.fr-en.references.test.jsonl')}
selected=[r for g,rs in sorted(bygroup.items())for r in rng.sample(rs,min(4,len(rs)))]
out=[dict(source=r['utterance'],reference=refs[r['case_id']],group=r['group_id'],case_id=r['case_id'],provenance=r['provenance'],license=r['license'])for r in selected]
lines(p/'guard.jsonl',out)
write(p/'data_manifest.json',{'seed':seed,'data':'Tatoeba human French-English training; DiaBLa evaluation only','sample_counts':{'train':4096,'dev':256,'test':512,'guard':len(out)},'new_external_dialogue_groups':len(bygroup),'previously_consumed_dialogue_groups_excluded':len(used),'warning':'All prior evaluation sets consumed, not reused here. Original component partitions retained. Base pretraining contamination unknown. New Tatoeba train excludes original fitting components; warm-start still legitimately contains their learned adapter. Tatoeba author/template independence not guaranteed.','attribution_handoff':str(DATA.parent/'HANDOFF_MANIFEST.json'),'no_context_input':True,'input_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in p.glob('*.jsonl')}})
print('FRENCH',len(bygroup),'new dialogue groups,',len(out),'guard turns',flush=True)
# Spanish: reproduce original contiguous-block partition; disjoint from ALL prior sampled blocks.
p=ROOT/'spanish';old=BASE/'translation_training_20261116';rng=random.Random(seed)
with tarfile.open(old/'es-en.tgz','r:gz')as tar:
 src=tar.extractfile('europarl-v7.es-en.es').read().decode().splitlines();ref=tar.extractfile('europarl-v7.es-en.en').read().decode().splitlines()
blocks=list(range((len(src)+255)//256));random.Random(20261116).shuffle(blocks);N=len(blocks)
bins={'train':set(blocks[:int(.8*N)]),'dev':set(blocks[int(.8*N):int(.9*N)]),'test':set(blocks[int(.9*N):])}
usedblocks={r['block']for s in ['train','dev','test']for r in rows(old/(s+'.jsonl'))};seen_src=set();seen_ref=set();pool={s:[]for s in bins}
for i,(s,r)in enumerate(zip(src,ref)):
 if i//256 in usedblocks or not(4<=len(s.split())<=32 and 4<=len(r.split())<=36)or not .4<=len(s)/max(1,len(r))<=2.5:continue
 if norm(s)in seen_src or norm(r)in seen_ref:continue
 seen_src.add(norm(s));seen_ref.add(norm(r));split=next(k for k,b in bins.items()if i//256 in b)
 pool[split].append(dict(source=s,reference=r,group=i//256,line=i,provenance='Europarl v7, Koehn 2005; European Parliament'))
for split,n in [('train',2048),('dev',128),('test',256)]:lines(p/(split+'.jsonl'),rng.sample(pool[split],n))
# WMT: reserve new rows excluding all project-consumed 200 +128 cases; source duplicates excluded.
iv=BASE/'veritastester-independent-validation/independent_validation';selection=json.loads((iv/'wmt_selection.json').read_text());prior=json.loads((old/'wmt_guard_selection.json').read_text());used=set(selection['row_indices_zero_based'])|set(prior['indices'])
srcfile=next(f for f in(iv/'sacrebleu_cache').rglob('*')if f.is_file()and hashlib.sha256(f.read_bytes()).hexdigest()==selection['source_sha256']);reffile=next(f for f in srcfile.parent.iterdir()if f.is_file()and hashlib.sha256(f.read_bytes()).hexdigest()==selection['reference_sha256']);ss=srcfile.read_text().splitlines();rr=reffile.read_text().splitlines()
exclude={norm(r['source'])for split in['train','dev','test']for r in rows(p/(split+'.jsonl'))};exclude|={norm(ss[i])for i in used}
ids=sorted(rng.sample([i for i in range(len(ss))if i not in used and norm(ss[i])not in exclude],256));out=[dict(source=ss[i],reference=rr[i],group=i,row=i,provenance='WMT13 Spanish-English source human news references via sacrebleu')for i in ids];lines(p/'guard.jsonl',out)
write(p/'data_manifest.json',{'seed':seed,'data':'Europarl human Spanish-English research corpus; WMT evaluation only','sample_counts':{'train':2048,'dev':128,'test':256,'guard':256},'previously_consumed_block_groups_excluded':len(usedblocks),'previous_wmt_rows_excluded':len(used),'guard_rows':ids,'warning':'Original contiguous block boundaries, not verified document/speaker boundaries. Old inspected evaluations not reused. Base pretrained OPUS may have seen evaluation material.','attribution':'Philipp Koehn 2005, Europarl; European Parliament; WMT13 via sacrebleu','input_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in p.glob('*.jsonl')}})
print('SPANISH prepared',flush=True)
