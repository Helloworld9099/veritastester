"""Research integrity checks. Never invokes the app, translator, or training."""
import collections, copy, json, unittest
from pathlib import Path
import prepare as prep

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'

class Integrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = {s: list(prep.rows(ART / f'{s}.inputs.jsonl')) for s in ('train', 'dev')}
        cls.refs = {s: list(prep.rows(ART / f'{s}.references.jsonl')) for s in ('train', 'dev')}
        cls.docs = prep.load_source()
        cls.raw = {r['source_row']: (d, r) for d, rs in cls.docs.items() for r in rs}

    def test_01_originals_and_pinned_sources_unchanged(self):
        self.assertEqual(prep.input_snapshot(), json.loads((ART / 'original_inputs_snapshot.json').read_text()))
        manifest = json.loads((ART / 'manifest.json').read_text())
        for name, item in manifest['artifacts'].items():
            self.assertEqual(prep.sha(ART / name), item['sha256'], name)

    def test_02_bounded_counts_and_conversation_disjointness(self):
        sets = {}
        for split, expected in [('train', 2048), ('dev', 256)]:
            rs = self.inputs[split]; self.assertEqual(len(rs), expected)
            self.assertEqual(len({r['case_id'] for r in rs}), len(rs))
            count = collections.Counter(r['group_id'] for r in rs)
            self.assertLessEqual(max(count.values()), prep.MAX_PER_GROUP)
            sets[split] = set(count)
        self.assertFalse(sets['train'] & sets['dev'])
        self.assertEqual((len(sets['train']), len(sets['dev'])), (240, 24))

    def test_03_exact_text_disjoint_from_dev_and_existing_holdouts(self):
        forbidden, _ = prep.protected_text(); strings = {}
        for split in ('train', 'dev'):
            strings[split] = {prep.norm(t) for r in self.inputs[split]
                for t in [r['utterance'], *(h['text'] for h in r['previous'])]}
            strings[split] |= {prep.norm(r['reference_translation']) for r in self.refs[split]}
            self.assertFalse(strings[split] & forbidden)
        self.assertFalse(strings['train'] & strings['dev'])

    def test_04_source_published_translation_join_and_alignment(self):
        for split in ('train', 'dev'):
            refs = {r['case_id']: r for r in self.refs[split]}
            self.assertEqual(set(refs), {r['case_id'] for r in self.inputs[split]})
            for r in self.inputs[split]:
                d, raw = self.raw[r['source_row']]
                self.assertEqual(r['utterance'], raw['source'])
                self.assertEqual(refs[r['case_id']]['reference_translation'], raw['reference'])
                self.assertEqual(r['group_id'], 'wmt24-dialogue:' + prep.digest(d)[:24])
                self.assertEqual(refs[r['case_id']]['group_id'], r['group_id'])
                self.assertEqual(r['original_split'], 'train')
                self.assertFalse(r['synthetic'])
                self.assertEqual(r['train_allowed'], split == 'train')

    def test_05_history_is_strictly_prior_original_source(self):
        for rs in self.inputs.values():
            for r in rs:
                current_doc, raw = self.raw[r['source_row']]
                self.assertTrue(1 <= len(r['previous']) <= prep.MAX_HISTORY)
                self.assertEqual([h['turn_index'] for h in r['previous']], sorted(h['turn_index'] for h in r['previous']))
                for h in r['previous']:
                    doc, prior = self.raw[h['source_row']]
                    self.assertEqual(doc, current_doc)
                    self.assertLess(h['turn_index'], raw['turn_index'])
                    self.assertLess(h['source_row'], raw['source_row'])
                    self.assertEqual(h['text'], prior['source'])
                    self.assertEqual(h['language'], prior['source_language'])
                    self.assertEqual(h['speaker'], prior['sender'])

    def test_06_model_request_allowlist_and_gold_injection(self):
        row = copy.deepcopy(self.inputs['train'][0]); row['reference_translation'] = 'GOLD_SENTINEL'
        row['authored_meaning'] = 'MEANING_SENTINEL'; row['future'] = 'FUTURE_SENTINEL'; row['human_judgement'] = 'helpful'
        for arm in ('no_context', 'all_history', 'selected_history', 'selected_history_plus_scene'):
            request = prep.request_payload(row, arm, [0])
            self.assertEqual(set(request), {'language', 'target_language', 'utterance', 'previous', 'situation'})
            self.assertNotIn('SENTINEL', json.dumps(request))
        self.assertEqual(prep.request_payload(row, 'no_context')['previous'], [])
        with self.assertRaises(ValueError): prep.request_payload(row, 'selected_history', [len(row['previous'])])

    def test_07_no_fresh_dialba_and_both_directions_reserved(self):
        registry = json.loads((ART / 'holdout_registry.json').read_text())
        self.assertEqual(registry, prep.holdout_registry())
        sets = [set(registry[k]) for k in ('smoke_groups', 'consumed_french_pilot_guard_groups', 'consumed_continued_french_guard_groups')]
        self.assertEqual([len(s) for s in sets], [40, 64, 40])
        self.assertEqual(len(set.union(*sets)), 144)
        self.assertEqual(registry['unscored_group_count'], 0)
        self.assertEqual(registry['unscored_candidate_groups'], [])
        self.assertFalse(registry['train_allowed'])

    def test_08_review_candidates_remain_unlabelled_train_only(self):
        rs = list(prep.rows(ART / 'memory_review_queue.UNLABELLED.jsonl'))
        groups = {r['group_id'] for r in self.inputs['train']}
        self.assertEqual(len(rs), 64)
        for r in rs:
            self.assertEqual(r['split'], 'train')
            self.assertIn(r['group_id'], groups); self.assertIn(r['candidate_group_id'], groups)
            self.assertIsNone(r['human_judgement'])
            self.assertEqual(r['situation'], ''); self.assertEqual(r['reference_translation'], '')
            self.assertEqual(r['label_status'], 'UNLABELLED_NOT_TRAINING_DATA')

    def test_09_rights_and_no_official_evaluation_acquisition(self):
        manifest = json.loads((ROOT / 'raw/download_manifest.json').read_text())
        self.assertFalse(manifest['official_valid_test_acquired'])
        self.assertEqual({r['file'] for r in manifest['files']}, {'wmt24_README.md', 'wmt24_LICENSE.md', 'wmt24_train_en-fr.csv'})
        for rs in self.inputs.values():
            for r in rs:
                self.assertEqual(r['license'], 'CC-BY-NC-4.0')
                self.assertTrue(r['research_only']); self.assertFalse(r['commercial_use_allowed'])
                self.assertTrue(r['attribution']); self.assertTrue(r['source_revision'])

if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(Integrity))
    report = {'checks_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'passed': result.wasSuccessful(), 'scope': 'data integrity, source/reference alignment, prior-only context, split/text isolation, rights and consumed holdout accounting; NOT translation quality'}
    destination = ROOT / 'reports'; destination.mkdir(exist_ok=True, mode=0o700)
    prep.write_json(destination / 'validation.json', report)
    raise SystemExit(not result.wasSuccessful())
