"""Offline, reproducible corpus preparation. Never trains or activates models."""
from __future__ import annotations
import collections, hashlib, json, re, unicodedata, zipfile, os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'raw'; OUT=ROOT/'private'; REPORT=ROOT/'reports'
for directory in (RAW,OUT,REPORT):
    directory.mkdir(exist_ok=True)
    os.chmod(directory,0o700)

def norm(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(name, rows):
    path=OUT/name
    with path.open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    os.chmod(path,0o600)
    return {'file':str(path.relative_to(ROOT)),'count':len(rows),'sha256':sha(path)}

def prepare_diabla():
    data=json.loads((RAW/'diabla_all_dialogues.json').read_text())
    srcsha=sha(RAW/'diabla_all_dialogues.json')
    inputs={'fr-en':[],'en-fr':[]}; refs={'fr-en':[],'en-fr':[]}
    protected=set(); rejected=collections.Counter()
    group_manifest=[]
    for did,dialogue in sorted(data.items()):
        gid='diabla:'+digest(did)[:24]
        history={'french':[], 'english':[]}
        scenario=dialogue.get('scenario',[])
        scene=scenario[0][0] if scenario and scenario[0] else ''
        roles={dialogue[k]['lang']:dialogue[k].get('role',[''])[0]
               for k in ('user1','user2')}
        group_manifest.append({'group_id':gid,'source_dialogue_id':did,
                               'role':'evaluation_only','train_allowed':False})
        for tid,u in sorted(dialogue['utterances'].items(),key=lambda item:int(item[0])):
            lang=u.get('language'); text=u.get('original_text','').strip()
            ref=u.get('reference_translation','').strip()
            if text:protected.add(norm(text))
            if ref:protected.add(norm(ref))
            if lang not in history:
                rejected['unknown_language']+=1;continue
            other='english' if lang=='french' else 'french'
            pair='fr-en' if lang=='french' else 'en-fr'
            if text and ref:
                cid=gid+':turn:'+str(tid)
                source_lang,target_lang=pair.split('-')
                item={'case_id':cid,'group_id':gid,'dataset':'DiaBLa',
                      'source_dialogue_id':did,'source_turn_id':str(tid),
                      'split':'test','role':'evaluation_only','train_allowed':False,
                      'language':source_lang,'target_language':target_lang,
                      'utterance':text,'previous':history[lang][-3:],
                      'situation':' '.join([scene,roles.get(lang,'')]).strip(),
                      'history_policy':'earlier turns as visible to this speaker; other-language turns use original MT, never gold references',
                      'provenance':'Bawden et al. 2021; github.com/rbawden/DiaBLa-dataset',
                      'source_sha256':srcsha,'license':'CC-BY-SA-4.0','synthetic':False}
                inputs[pair].append(item)
                refs[pair].append({'case_id':cid,'group_id':gid,
                      'reference_translation':ref,'reference_type':'source-published-human',
                      'source_sha256':srcsha,'license':'CC-BY-SA-4.0'})
            else:rejected['missing_original_or_reference']+=1
            # Update only after creating the current request. Never look ahead.
            if text:history[lang].append(text)
            visible=u.get('postprocessed_text') or u.get('translated_text') or ''
            if visible.strip():history[other].append(visible.strip())
    files=[]
    for pair in inputs:
        assert {x['case_id'] for x in inputs[pair]}=={x['case_id'] for x in refs[pair]}
        assert all('reference_translation' not in x for x in inputs[pair])
        files.extend([dump(f'diabla.{pair}.inputs.test.jsonl',inputs[pair]),
                      dump(f'diabla.{pair}.references.test.jsonl',refs[pair])])
        # A small run spans many dialogues, deterministic and label-blind.
        small=[]
        for gid in sorted({x['group_id'] for x in inputs[pair]},key=digest)[:40]:
            group=[x for x in inputs[pair] if x['group_id']==gid and x['previous']]
            if group:small.append(group[min(4,len(group)-1)])
        files.append(dump(f'diabla.{pair}.smoke.inputs.test.jsonl',small))
    files.append(dump('diabla.groups.test.jsonl',group_manifest))
    return {'counts':{k:len(v) for k,v in inputs.items()},'dialogues':len(data),
            'rejected':dict(rejected),'protected_strings':len(protected),'files':files},protected

class DSU:
    def __init__(self):self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        if self.parent[x]!=x:self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def prepare_tatoeba(protected):
    with zipfile.ZipFile(RAW/'fra-eng.zip') as archive:
        about=archive.read('_about.txt').decode('utf-8')
        (RAW/'tatoeba_mirror_about.txt').write_text(about)
        lines=archive.read('fra.txt').decode('utf-8-sig').splitlines()
    dsu=DSU(); pairs=[]; seen=set(); rejected=collections.Counter()
    for n,line in enumerate(lines,1):
        cols=line.split('\t')
        if len(cols)<2:rejected['invalid_columns']+=1;continue
        en,fr=cols[:2]; en=en.strip();fr=fr.strip()
        if not en or not fr:rejected['empty_text']+=1;continue
        key=(norm(fr),norm(en))
        if key in seen:rejected['duplicate_pair']+=1;continue
        seen.add(key);a='fr:'+digest(key[0]);b='en:'+digest(key[1]);dsu.union(a,b)
        pairs.append((n,fr,en,a,b))
    blocked={dsu.find(a) for _,fr,en,a,b in pairs if norm(fr) in protected or norm(en) in protected}
    splits={'train':[],'dev':[],'test':[]}; source_sha=sha(RAW/'fra-eng.zip')
    for n,fr,en,a,b in pairs:
        component=dsu.find(a)
        if component in blocked:rejected['DiaBLa_overlap_connected_component']+=1;continue
        h=digest('split-v1:'+component);v=int(h[:8],16)%100
        split='train' if v<80 else 'dev' if v<90 else 'test'
        splits[split].append({'case_id':'tatoeba-fra-eng:row:'+str(n),
            'group_id':'tatoeba-component:'+digest(component)[:24],
            'dataset':'Tatoeba-ManyThings-TensorFlow-2019-04-07',
            'split':split,'language':'fr','target_language':'en',
            'utterance':fr,'reference_translation':en,'previous':[],
            'situation':'','synthetic':False,'label_kind':'parallel_translation_not_memory_utility',
            'provenance':'Tatoeba contributors; curated by ManyThings.org/anki; TensorFlow mirror',
            'source_row':n,'source_sha256':source_sha,'license':'CC-BY-2.0',
            'attribution':'tatoeba.org and www.manythings.org/anki; see raw/tatoeba_mirror_about.txt',
            'rights_note':'Local research only in this preparation; old mirror lacks per-sentence contributor IDs. Review attribution before redistributing text or artifacts.'})
    sets={s:{r['group_id'] for r in rows} for s,rows in splits.items()}
    assert not (sets['train']&sets['dev'] or sets['train']&sets['test'] or sets['dev']&sets['test'])
    for rows in splits.values():
        assert not any(norm(r['utterance']) in protected or norm(r['reference_translation']) in protected for r in rows)
    files=[]
    for split,rows in splits.items():files.append(dump(f'tatoeba.fr-en.{split}.jsonl',rows))
    small=sorted(splits['train'],key=lambda r:digest(r['case_id']))[:12000]
    files.append(dump('tatoeba.fr-en.train.quick12000.jsonl',small))
    return {'raw_rows':len(lines),'accepted':{k:len(v) for k,v in splits.items()},
            'groups':{k:len(v) for k,v in sets.items()},'rejected':dict(rejected),
            'cross_split_component_conflicts':0,'exact_overlap_with_DiaBLa_after_filter':0,
            'known_limit':'Sentence-connected components are not independent authors or semantic template families.',
            'local_research_allowed_by_publisher_notice':True,
            'publication_rights_fully_resolved':False,'files':files}

def prepare_magpie():
    raw=[json.loads(line) for line in (RAW/'MAGPIE_filtered_split_typebased.jsonl').read_text().splitlines()]
    splitmap={'training':'train','development':'dev','test':'test'}
    rejected=collections.Counter(); parsed=[]; source_sha=sha(RAW/'MAGPIE_filtered_split_typebased.jsonl')
    for r in raw:
        expression=r['idiom'].strip();contexts=r['context']
        # Source uses two preceding + target + two following sentences.
        # Use supplied character offsets, rather than matching the dictionary
        # idiom: inflection, inserted words and passive forms are legitimate.
        if len(contexts)!=5:
            rejected['invalid_five_sentence_window']+=1;continue
        i=2; offsets=r.get('offsets',[])
        if not offsets or not all(0 <= a < b <= len(contexts[i]) and contexts[i][a:b].strip() for a,b in offsets):
            rejected['invalid_source_target_offsets']+=1;continue
        if r.get('label') not in ('i','l'):rejected['invalid_label']+=1;continue
        parsed.append({'case_id':'magpie:'+str(r['id']),'dataset':'MAGPIE-typebased',
            'group_id':'magpie-idiom:'+digest(norm(expression))[:20],
            'document_id':r['document_id'],'source_sentence_id':r['sentence_no'],
            'split':splitmap[r['split']],'expression':expression,'language':'en',
            'utterance':contexts[i],'previous':contexts[max(0,i-2):i],
            'source_label':r['label'],'label':1 if r['label']=='i' else 0,
            'label_kind':'idiomaticity_only_1_is_idiomatic_0_is_literal',
            'confidence':r['confidence'],'license':'CC-BY-4.0',
            'target_offsets':r['offsets'],'surface_form':' '.join(contexts[i][a:b] for a,b in r['offsets']),
            'target_alignment':'source middle sentence, validated supplied character offsets',
            'provenance':'Haagsma, Bos and Nissim, 2020, MAGPIE Corpus; BNC-derived',
            'source_sha256':source_sha,'synthetic':False,
            'warning':'Not a translation target, exact sense label, or memory usefulness judgment.'})
    # Test priority, then dev: remove whole lower-priority source documents.
    chosen={}; reserved_docs=set(); reserved_text=set(); reserved_groups=set()
    for split in ('test','dev','train'):
        kept=[]
        for r in parsed:
            if r['split']!=split:continue
            if r['document_id'] in reserved_docs:
                rejected[split+'_document_overlap_with_higher_priority_split']+=1;continue
            texts={norm(r['utterance']),*(norm(s) for s in r['previous'])}
            if texts & reserved_text:
                rejected[split+'_context_text_overlap_with_higher_priority_split']+=1;continue
            if r['group_id'] in reserved_groups:
                rejected[split+'_idiom_overlap_with_higher_priority_split']+=1;continue
            kept.append(r)
        chosen[split]=kept
        reserved_docs|={r['document_id'] for r in kept}
        reserved_groups|={r['group_id'] for r in kept}
        for r in kept:reserved_text|={norm(r['utterance']),*(norm(s) for s in r['previous'])}
    files=[dump('magpie.'+s+'.jsonl',rows) for s,rows in chosen.items()]
    return {'raw_rows':len(raw),'accepted':{s:len(rows) for s,rows in chosen.items()},
            'label_counts':{s:dict(collections.Counter(r['source_label'] for r in rows)) for s,rows in chosen.items()},
            'groups':{s:len({r['group_id'] for r in rows}) for s,rows in chosen.items()},
            'rejected':dict(rejected),'cross_split_document_or_idiom_conflicts':0,
            'label_convention_warning':'This bundle uses label=1 for idiomatic; SemEval uses source Label=0 for idiomatic. Never merge raw numeric labels.',
            'files':files}

SYNTHETIC=[
 ('bank','The bank is nearby.','The river flooded the bank yesterday.','The teller works at the bank.','We are walking along a river.','We are looking for a place to deposit money.','river edge','financial institution'),
 ('crane','Look at the crane.','The crane has a long beak.','The crane lifts steel beams.','We are watching birds at a wetland.','We are at a construction site.','bird','lifting machine'),
 ('seal','The seal is damaged.','The wax seal on the envelope cracked.','The injured seal needs a veterinarian.','We are examining a closed envelope.','We are treating a marine animal.','closure or stamp','marine mammal'),
 ('pitch','That pitch was high.','The singer struggled with that note.','The pitcher threw above the strike zone.','We are rehearsing a song.','We are watching baseball.','musical frequency','baseball throw'),
 ('charge','They questioned the charge.','The invoice lists a delivery fee.','The prosecutor filed an accusation.','We are reviewing an invoice.','We are discussing a court case.','fee','criminal accusation'),
 ('light','It is light.','The empty suitcase weighs very little.','The room is bright after sunrise.','We are weighing luggage.','We are describing illumination.','low weight','bright'),
 ('cold','She is cold.','She forgot her coat in the snow.','She answered without warmth or sympathy.','We are outside in winter.','We are discussing an unfriendly response.','low body temperature','emotionally distant'),
 ('cool','That is cool.','The soup has cooled enough to eat.','Everyone admired the clever new design.','We are checking food temperature.','We are reacting positively to a design.','low temperature','impressive'),
 ('sick','He is sick.','He has a fever and needs to rest.','His skateboarding tricks impressed everyone.','We are talking about an illness.','We are praising a skateboarder using slang.','ill','exceptionally skilled'),
 ('fire','That is fire.','Flames are spreading through the shed.','The audience loved the new track.','We are reporting a burning building.','We are praising music using slang.','combustion','excellent'),
 ('dead','It is dead.','The battery has no remaining charge.','The plant has died and cannot recover.','We are checking a rechargeable battery.','We are examining a dead plant.','out of electrical power','no longer alive'),
 ('break','We need a break.','Everyone is tired after hours of work.','The code should exit this loop.','We are planning a rest from work.','We are discussing control flow in code.','rest interval','loop termination'),
 ('draw','It was a draw.','Both teams finished with equal scores.','A name was selected randomly for the prize.','We are discussing the result of a match.','We are discussing a lottery selection.','tied result','random selection'),
 ('fine','That is fine.','The fabric has extremely thin threads.','The proposal is acceptable to the group.','We are inspecting the texture of fabric.','We are agreeing to a proposal.','delicate texture','acceptable'),
 ('right','Take the right one.','The left path is blocked.','The marked answer is correct.','We are choosing a direction.','We are checking answers on a worksheet.','right-hand direction','correct choice'),
]
def prepare_synthetic():
    # These labels are authored fixture relations, not human-rated transfer value.
    files=[]; bysplit=collections.defaultdict(list)
    for i,(term,current,m1,m2,s1,s2,g1,g2) in enumerate(SYNTHETIC):
        split='train' if i<9 else 'dev' if i<12 else 'test'
        for sense,(memory,other,scene,meaning) in enumerate(((m1,m2,s1,g1),(m2,m1,s2,g2))):
            for relation,candidate in [('supportive',memory),('conflicting',other),('unrelated','The meeting begins at noon.')]:
                bysplit[split].append({'case_id':f'synthetic:{term}:{sense}:{relation}',
                   'group_id':'synthetic-family:'+term,'split':split,'language':'en',
                   'expression':term,'utterance':current,'previous':[],
                   'candidate_memory':candidate,'situation':scene,
                   'authored_meaning':meaning,'synthetic_relation':relation,
                   'synthetic':True,'human_reviewed':False,'real_translation_reference':False,
                   'role':'development_fixture_only','training_allowed':'synthetic experimentation only; keep separate from real labels',
                   'provenance':'Original AI-authored fixture for this task, 2026-11-16',
                   'warning':'Not a measured helpful/neutral/harmful judgment. Do not map directly into human_judgement or claim real accuracy.'})
    for split,rows in bysplit.items():files.append(dump(f'synthetic.context.{split}.jsonl',rows))
    return {'count':sum(map(len,bysplit.values())),'counts':{s:len(r) for s,r in bysplit.items()},
            'expression_family_disjoint':True,'human_reviewed':False,'files':files}

if __name__=='__main__':
    d,protected=prepare_diabla()
    report={'created_utc':'2026-11-16','training_executed':False,'published':False,
            'diabla':d,'tatoeba':prepare_tatoeba(protected),
            'magpie':prepare_magpie(),'synthetic':prepare_synthetic()}
    (REPORT/'data_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({k:({a:b for a,b in v.items() if a!='files'} if isinstance(v,dict) else v) for k,v in report.items()},indent=2))
