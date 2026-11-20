import tempfile
import unittest
from pathlib import Path

from research.es_en_v2.controller import (
    HumanDevResult,SUBGROUPS,decide_promotion,next_experiment,
    write_immutable_receipt,
)


def human(score=.95,critical=.004,rev=10,hashval="a"*64):
    return HumanDevResult(direction="es-en",reviews_n=rev*len(SUBGROUPS),
                          overall_pass=score,
                          by_subgroup={k:{"n":max(20,rev),"pass_rate":score} for k in SUBGROUPS},
                          critical_error_rate=critical,reviewer_agreement=.9,
                          dataset_sha256=hashval,blinded=True)


class TestController(unittest.TestCase):
    def test_no_human_labels_never_promote(self):
        result=decide_promotion(None,None,code_passed=True,candidate_p95=.2)
        self.assertFalse(result["promote"])

    def test_permitted_and_human_reviewed_only(self):
        previous=human(.93,rev=40)
        updated=human(.96,rev=40)
        approved=decide_promotion(previous,updated,parent_p95=.5,
                candidate_p95=.48,candidate_peak_rss_kib=800000,
                seed_variation_gain_floor=.01,code_passed=True)
        self.assertTrue(approved["promote"])
        self.assertFalse(approved["production_authorization_granted"])
        self.assertFalse(approved["final_human_holdout_evaluated"])

    def test_critical_regression_blocks_any_other_gain(self):
        previous=human(.93,critical=.005,rev=40)
        candidate=human(.97,critical=.006,rev=40)
        result=decide_promotion(previous,candidate,parent_p95=.5,
                candidate_p95=.4,candidate_peak_rss_kib=800000,
                seed_variation_gain_floor=.001,code_passed=True)
        self.assertFalse(result["promote"])
        self.assertIn("Critical error rate increased",result["reasons"])

    def test_missing_subgroup_or_dataset_mismatch_rejected(self):
        previous=human(.93,rev=40)
        candidate=human(.97,rev=40,hashval="b"*64)
        result=decide_promotion(previous,candidate,parent_p95=.5,
                candidate_p95=.4,candidate_peak_rss_kib=800000,
                seed_variation_gain_floor=.001,code_passed=True)
        self.assertFalse(result["promote"])

    def test_no_unreviewed_synthetic_slanga_training(self):
        plan=next_experiment({"contains_slang_needs_review":90},{"regional_slang":0})
        self.assertEqual(plan["target"],"regional_slang")
        self.assertFalse(plan["can_train"])
        self.assertTrue(plan["no_final_holdout_used"])

    def test_immutable_history_never_claims_final_95(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"log.json"
            obj={"parent_sha256":"a"*64,"checkpoint_sha256":"b"*64,
                 "human_final_95_verified":False}
            digest=write_immutable_receipt(obj,path)
            self.assertEqual(len(digest),64)
            with self.assertRaises(FileExistsError):
                write_immutable_receipt(obj,path)
            with self.assertRaises(ValueError):
                write_immutable_receipt({**obj,
                    "human_final_95_verified":True},Path(tmp)/"bad.json")


if __name__=="__main__":
    unittest.main()
