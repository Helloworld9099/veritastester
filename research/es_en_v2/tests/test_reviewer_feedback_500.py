import json
import unittest

from research.es_en_v2.review_feedback_500 import (
    SPEC, load_review, build
)


class TestQualitativeFeedbackIsNotAnAccuracyScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(SPEC.read_text())
        cls.rows = load_review()
        cls.ledger, cls.summary = build(cls.rows, cls.spec)

    def test_all_500_real_bot_outputs_map_to_excel_rows(self):
        self.assertEqual(len(self.ledger), 500)
        self.assertEqual([row["Excel row"] for row in self.ledger],
                         list(range(2, 502)))
        self.assertEqual(len({row["Case ID"] for row in self.ledger}), 500)

    def test_hard_examples_are_tagged_without_fabricated_human_labels(self):
        for number, tag in [
            (311, "repetition_loop"), (314, "repetition_loop"),
            (416, "repetition_loop"), (318, "critical_meaning_or_role"),
            (457, "critical_meaning_or_role"), (436, "critical_meaning_or_role"),
            (176, "idiom_or_slang_interpretation"), (366, "idiom_or_slang_interpretation"),
        ]:
            row = self.ledger[number-2]
            self.assertIn(tag, row["Reported problem categories"])
            self.assertEqual(row["Formal pass/fail"], "")
            self.assertEqual(row["Adjudicated?"], "No")

    def test_original_review_identifiers_match_reviewer_claims(self):
        self.assertIn("Guisantes", self.ledger[311-2]["Original spoken text"])
        self.assertIn("yummy, yummy", self.ledger[311-2]["Actual model translation"].lower())
        self.assertIn("i heard", self.ledger[318-2]["Actual model translation"].lower())
        self.assertIn("diez de la tarde", self.ledger[457-2]["Actual model translation"].lower())
        self.assertIn("cien mil", self.ledger[436-2]["Actual model translation"].lower())

    def test_unmentioned_examples_not_counted_as_correct(self):
        self.assertEqual(self.summary["per_row_pass_fail_labels_collected"], 0)
        self.assertEqual(self.summary["input_model_responses"], 500)
        self.assertGreater(self.summary["unflagged_not_assumed_correct"], 0)
        self.assertFalse(self.summary["independently_evaluated_95_percent"])
        self.assertEqual(self.summary["training_examples_added"], 0)

    def test_semantically_ambiguous_report_is_not_autolabeled_fail(self):
        # The reviewer flagged row 360 as omitting a prior worried clause,
        # but that wording isn't in source row 360 of the pinned workbook.
        # It is legitimate feedback to reconcile, not an automatic label.
        row = self.ledger[358]
        self.assertEqual(row["Excel row"], 360)
        self.assertIn("flagged_for_adjudication", row["Review status"])
        self.assertEqual(row["Formal pass/fail"], "")


if __name__ == "__main__":
    unittest.main()
