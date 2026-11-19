"""Offline guards for genuinely new Stage8 Tatoeba groups and pinned prior models."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_stage8 import (ROOT,WARM_REL,WARM_SHA,TATOEBA_TRAIN,
                          MASSIVE_REPLAY,TEST_SIZE,UPDATES,SEED,
                          sha_pairs)
import train_stage7 as stage7


def row(index, group=None):
    return {
        "case_id": "row-" + str(index),
        "group_id": group or "connected-component-" + str(index),
        "utterance": f"Je voudrais ecouter cette chanson avec les amis numero {index}",
        "reference_translation": f"I want to listen to this song with friends number {index}",
    }


class TestStageEight(unittest.TestCase):
    def test_warm_adapter_is_genuinely_stage7_selected(self):
        expected = ROOT / WARM_REL
        self.assertTrue(expected.is_file())
        self.assertEqual(hashlib.sha256(expected.read_bytes()).hexdigest(), WARM_SHA)
        self.assertEqual((UPDATES, TATOEBA_TRAIN, MASSIVE_REPLAY, TEST_SIZE),
                         (512, 6144, 2048, 256))

    def test_historical_old_train_and_test_component_exclusion(self):
        many = [row(i) for i in range(140)]
        prior = [row(0), row(4), row(7)]
        prior_test = [row(21), row(28)]
        used_components = {r["group_id"] for r in prior+prior_test}
        old_quick = [row(60), row(62)]
        first = stage7.choose_train_and_test(
            many, old_quick+prior+prior_test, set(),
            train_n=24, test_n=15)
        chosen = first["train"]+first["reserved_test"]
        source_lookup={r["utterance"]:r["group_id"] for r in many}
        self.assertFalse({source_lookup[src] for src,_ in chosen} & used_components)
        self.assertFalse({source_lookup[src] for src,_ in chosen} &
                         {r["group_id"] for r in old_quick})
        self.assertEqual(first,stage7.choose_train_and_test(
            many,old_quick+prior+prior_test,set(),train_n=24,test_n=15))

    def test_multi_translation_component_cannot_cross_frozen_sets(self):
        many=[row(i) for i in range(135)]
        for i in (2,5,9):
            many[i]["group_id"]=many[1]["group_id"]
        selected=stage7.choose_train_and_test(many,[],set(),
                                               train_n=25,test_n=22)
        source_lookup={r["utterance"]:r["group_id"] for r in many}
        first_groups={source_lookup[src] for src,_ in selected["train"]}
        second_groups={source_lookup[src] for src,_ in selected["reserved_test"]}
        self.assertFalse(first_groups & second_groups)

    def test_protect_final_test_source_strings_without_reuse(self):
        many=[row(i) for i in range(85)]
        protected={stage7.norm(many[3]["utterance"])}
        with self.assertRaisesRegex(RuntimeError,"DiaBLa final test"):
            stage7.choose_train_and_test(many,[],protected,train_n=12,test_n=12)

    def test_pinned_research_data_manifest_is_deterministic(self):
        self.assertEqual(SEED,202611188)
        self.assertEqual(len(sha_pairs([("Bonjour","Hello")])),64)


if __name__=="__main__":
    unittest.main()
