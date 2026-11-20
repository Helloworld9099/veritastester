import json
import unittest

from research.unified_models.feedback_priorities import (
    SUMMARY, PRIORITIES, derive_development_data_priorities
)


class TestReviewGuidesNewTrainingWithoutContaminatingIt(unittest.TestCase):
    def test_review_is_never_claimed_as_500_human_gold_training_examples(self):
        report=json.loads(SUMMARY.read_text())
        result=derive_development_data_priorities(report)
        self.assertEqual(result["status"],"review_informed_not_executed_data_priorities")
        self.assertEqual(result["new_verified_examples_added"],0)
        self.assertTrue(result["all_500_previous_examples_held_out_from_new_training"])
        self.assertTrue(result["no_final_holdout_scored"])
        self.assertEqual(set(result["language_plan"]),{"es-en","en-es"})
        self.assertFalse(any("accuracy" in row for group in result["language_plan"].values()
                             for row in group))

    def test_real_user_flag_direction_breakdown_and_priorities(self):
        result=derive_development_data_priorities(json.loads(SUMMARY.read_text()))
        es=result["language_plan"]["es-en"]
        en=result["language_plan"]["en-es"]
        self.assertTrue(any(x["category"]=="critical_meaning_or_role" and
                            x["reported_row_flags"]==14 for x in es))
        self.assertTrue(any(x["category"]=="critical_meaning_or_role" and
                            x["reported_row_flags"]==7 for x in en))
        self.assertTrue(any(x["category"]=="idiom_or_slang_interpretation" for x in es))
        self.assertTrue(any(x["category"]=="register_formality" for x in en))
        for route in (es,en):
            priority=[x["risk_index_not_accuracy"] for x in route]
            self.assertEqual(priority,sorted(priority,reverse=True))

    def test_never_accept_false_independent_95_or_new_training_from_dev(self):
        summary=json.loads(SUMMARY.read_text())
        for overwrite in ({"independently_evaluated_95_percent":True},
                          {"training_examples_added":100},
                          {"per_row_pass_fail_labels_collected":500}):
            with self.subTest(fake=overwrite),self.assertRaises(ValueError):
                derive_development_data_priorities({**summary,**overwrite})


if __name__=="__main__":
    unittest.main()
