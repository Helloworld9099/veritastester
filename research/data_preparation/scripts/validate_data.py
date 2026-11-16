"""Independent structural/leakage checks; no model quality claims."""
import collections,hashlib,json,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from prepare_data import norm
P=Path(__file__).resolve().parents[1];D=P/'private'
def read(name):return [json.loads(x) for x in (D/name).read_text().splitlines()]
class DataChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.t={s:read(f'tatoeba.fr-en.{s}.jsonl') for s in ('train','dev','test')}
  cls.m={s:read(f'magpie.{s}.jsonl') for s in ('train','dev','test')}
  cls.di={p:read(f'diabla.{p}.inputs.test.jsonl') for p in ('fr-en','en-fr')}
  cls.dr={p:read(f'diabla.{p}.references.test.jsonl') for p in ('fr-en','en-fr')}
 def assert_disjoint(self,groups):
  for i,x in enumerate(groups):
   for y in groups[i+1:]:self.assertFalse(x&y)
 def test_tatoeba_groups_and_both_text_sides_are_disjoint(self):
  for key in ('group_id','utterance','reference_translation'):
   self.assert_disjoint([{norm(r[key]) for r in rows} for rows in self.t.values()])
 def test_no_tatoeba_text_is_in_diabla(self):
  protected={norm(r['utterance']) for rows in self.di.values() for r in rows}
  protected|={norm(r['reference_translation']) for rows in self.dr.values() for r in rows}
  for rows in self.t.values():
   self.assertFalse(protected & {norm(r[k]) for r in rows for k in ('utterance','reference_translation')})
 def test_magpie_documents_expressions_and_contexts_are_disjoint(self):
  for k in ('document_id','group_id'):
   self.assert_disjoint([{r[k] for r in rows} for rows in self.m.values()])
  self.assert_disjoint([{norm(t) for r in rows for t in [r['utterance'],*r['previous']]} for rows in self.m.values()])
 def test_magpie_target_and_labels_match_source(self):
  raw={str(r['id']):r for r in map(json.loads,(P/'raw/MAGPIE_filtered_split_typebased.jsonl').read_text().splitlines())}
  for rows in self.m.values():
   for r in rows:
    original=raw[r['case_id'].split(':')[1]]
    self.assertEqual(r['utterance'],original['context'][2])
    self.assertEqual(r['previous'],original['context'][:2])
    self.assertEqual(r['label'],int(original['label']=='i'))
    self.assertEqual(r['source_label'],original['label'])
    self.assertEqual(r['target_offsets'],original['offsets'])
 def test_diabla_input_never_contains_gold_or_future_history(self):
  raw=json.loads((P/'raw/diabla_all_dialogues.json').read_text())
  for pair,rows in self.di.items():
   for r in rows:
    self.assertNotIn('reference_translation',r)
    self.assertFalse(r['train_allowed']);self.assertEqual(r['split'],'test')
    u=raw[r['source_dialogue_id']]['utterances'];idx=int(r['source_turn_id'])
    lang='french' if pair=='fr-en' else 'english'
    actual=[]
    for tid,t in sorted(u.items(),key=lambda x:int(x[0])):
     if int(tid)>=idx:break
     v=t['original_text'] if t['language']==lang else (t.get('postprocessed_text') or t.get('translated_text') or '')
     if v.strip():actual.append(v.strip())
    self.assertEqual(r['previous'],actual[-3:])
 def test_diabla_reference_join_is_one_to_one(self):
  for pair,inputs in self.di.items():
   a=[r['case_id'] for r in inputs];b=[r['case_id'] for r in self.dr[pair]]
   self.assertEqual(len(a),len(set(a)));self.assertEqual(a,b)
 def test_synthetic_labels_are_not_human_ground_truth(self):
  groups=[]
  for split in ('train','dev','test'):
   rows=read(f'synthetic.context.{split}.jsonl');groups.append({r['group_id'] for r in rows})
   for r in rows:
    self.assertTrue(r['synthetic']);self.assertFalse(r['human_reviewed'])
    self.assertNotIn('human_judgement',r);self.assertNotIn('reference_translation',r)
  self.assert_disjoint(groups)
 def test_manifest_checksums(self):
  audit=json.loads((P/'reports/data_audit.json').read_text())
  for section in ('diabla','tatoeba','magpie','synthetic'):
   for item in audit[section]['files']:
    f=P/item['file'];self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),item['sha256'])
    self.assertEqual(sum(1 for _ in f.open()),item['count'])
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(DataChecks)
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
         'passed':result.wasSuccessful(),'scope':'schema, source alignment, no future/reference leakage, split isolation and hashes; not model quality'}
 (P/'reports/validation.json').write_text(json.dumps(report,indent=2))
 raise SystemExit(0 if result.wasSuccessful() else 1)
