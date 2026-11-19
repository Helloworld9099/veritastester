"""Pure-Python training-dataset firewall tests; never download on import."""
import hashlib
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_stage7 import norm,choose_train_and_test,sha_pairs,WARM_REL,WARM_SHA,ROOT

def row(i,group=None):
    return {
        "case_id":"row:"+str(i),
        "group_id":group or "group:"+str(i),
        "utterance":f"Je veux parler avec mes amis numero {i}",
        "reference_translation":f"I want to talk with my friends number {i}",
    }

class TestStageSeven(unittest.TestCase):
    def test_test_does_not_share_component_with_train_or_old_quick(self):
        rows=[row(i) for i in range(80)]
        # A linked source/target should never move across splits.
        rows[2]["group_id"]=rows[0]["group_id"]
        old=[rows[0]]
        cohorts=choose_train_and_test(rows,old,set(),train_n=25,test_n=20)
        all_pairs=cohorts["train"]+cohorts["reserved_test"]
        self.assertEqual((len(cohorts["train"]),len(cohorts["reserved_test"])),(25,20))
        self.assertNotIn((rows[0]["utterance"],rows[0]["reference_translation"]),all_pairs)
        self.assertNotIn((rows[2]["utterance"],rows[2]["reference_translation"]),all_pairs)
        self.assertEqual(cohorts,choose_train_and_test(rows,old,set(),
                                                       train_n=25,test_n=20))
        self.assertNotEqual(cohorts["train_manifest"]["group_sha256"],
                            cohorts["reserved_test_manifest"]["group_sha256"])

    def test_linked_component_never_repeated_or_half_selected(self):
        rows=[row(i) for i in range(50)]
        for i in (2,5,7,9):
            rows[i]["group_id"]=rows[0]["group_id"]
        cohorts=choose_train_and_test(rows,[],set(),train_n=14,test_n=11)
        self.assertEqual(len(set(cohorts["train"])),14)
        self.assertEqual(len(set(cohorts["reserved_test"])),11)

    def test_external_diabla_is_protected(self):
        rows=[row(i) for i in range(60)]
        with self.assertRaisesRegex(RuntimeError,"DiaBLa final test"):
            choose_train_and_test(rows,[],{norm(rows[4]["utterance"])},
                                  train_n=10,test_n=10)

    def test_not_enough_disjoint_data_fails(self):
        with self.assertRaisesRegex(ValueError,"Not enough component-disjoint"):
            choose_train_and_test([row(i) for i in range(8)],[],set(),
                                  train_n=8,test_n=3)

    def test_saved_trained_warm_model_hash_verified(self):
        model=ROOT/WARM_REL
        self.assertTrue(model.is_file())
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(),WARM_SHA)
        self.assertEqual(len(sha_pairs([("Bonjour","Hello")])),64)

if __name__=="__main__":
    unittest.main()
