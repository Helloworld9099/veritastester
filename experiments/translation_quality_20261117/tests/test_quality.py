import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_human import CATEGORIES, score, wilson_lower


def digest(x):
    return hashlib.sha256(str(x).encode()).hexdigest()


def cases(n=220, good=220):
    return [{
        "sample_id": str(i),
        "language": "spanish",
        "checkpoint_sha256": digest("model-v1"),
        "evaluation_manifest_sha256": digest("heldout-sample-v1"),
        "source_group_sha256": digest(i // 10),
        "category": CATEGORIES[i % len(CATEGORIES)],
        "split": "independent_test",
        "rater_a_adequate": i < good,
        "rater_b_adequate": i < good,
    } for i in range(n)]


class TestQualityGate(unittest.TestCase):
    def test_all_verified_adequate_satisfies_90_gate(self):
        report = score(cases())
        self.assertTrue(report["meets_90_percent_human_adequacy_gate"])
        self.assertGreater(report["wilson_lower_95"], 0.9)

    def test_exactly_90_observed_does_not_prove_90_adequacy(self):
        report = score(cases(good=198))
        self.assertAlmostEqual(report["observed_fraction"], 0.9)
        self.assertFalse(report["meets_90_percent_human_adequacy_gate"])

    def test_insufficient_sample_not_eligible(self):
        report = score(cases(100, 100))
        self.assertFalse(report["eligibility"]["at_least_200_cases"])
        self.assertFalse(report["meets_90_percent_human_adequacy_gate"])

    def test_disagreement_requires_adjudication(self):
        sample = cases(1)[0]
        sample["rater_b_adequate"] = False
        with self.assertRaisesRegex(ValueError, "adjudication"):
            score([sample])
        sample["adjudicated_adequate"] = False
        self.assertEqual(score([sample])["adequate"], 0)

    def test_no_mixed_models_or_duplicated_ids(self):
        rows = cases(2)
        rows[1]["checkpoint_sha256"] = digest("model-v2")
        with self.assertRaisesRegex(ValueError, "Do not pool"):
            score(rows)
        rows = cases(2)
        rows[1]["sample_id"] = rows[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            score(rows)

    def test_reject_ineligible_data(self):
        rows = cases(1)
        rows[0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "independent_test"):
            score(rows)
        with self.assertRaises(ValueError):
            wilson_lower(2, 1)


if __name__ == "__main__":
    unittest.main()
