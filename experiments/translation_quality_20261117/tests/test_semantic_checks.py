"""Unit tests for bilingual numerical preservation (not semantic adequacy)."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_checks import numeric_fidelity, parse_number


class TestNumbers(unittest.TestCase):
    def test_french_dangerous_decimal_error_caught(self):
        src = "69 décès (0,004 % de la population) liés à la COVID-19."
        faulty = "69 deaths (0.04% of the population) linked to COVID-19."
        correct = "69 deaths (0.004% of the population) linked to COVID-19."
        self.assertEqual(numeric_fidelity(src, faulty, "french")["status"], "numeric_error")
        self.assertEqual(numeric_fidelity(src, correct, "french")["status"], "numbers_preserved")

    def test_spanish_date_and_percentage(self):
        src = "El 31 de enero, 20 personas (2,5 %) habían sido examinadas."
        en = "On January 31, 20 people (2.5%) had been tested."
        wrong = "On January 31, 200 people (25%) had been tested."
        self.assertEqual(numeric_fidelity(src, en, "spanish")["status"], "numbers_preserved")
        self.assertEqual(numeric_fidelity(src, wrong, "spanish")["status"], "numeric_error")

    def test_english_thousand_grouping(self):
        src = "El año 2025 tuvo 2000 casos."
        en = "The year 2025 had 2,000 cases."
        self.assertEqual(numeric_fidelity(src, en, "spanish")["status"], "numbers_preserved")

    def test_ambiguous_number_does_not_claim_failure(self):
        src = "Tenemos 1,234 pacientes."
        en = "We have 1,234 patients."
        self.assertEqual(numeric_fidelity(src, en, "spanish")["status"], "uncertain")

    def test_absent_number_does_not_imply_semantic_accuracy(self):
        r = numeric_fidelity("Estoy enfermo.", "I am well.", "spanish")
        self.assertEqual(r["status"], "no_numeric_facts")
        self.assertIn("no semantic-equivalence", r["limitations"])

    def test_missing_or_added_number_is_error(self):
        src = "Necesitamos 4 botellas de agua."
        self.assertEqual(numeric_fidelity(src, "We need bottles of water.", "spanish")["status"],
                         "numeric_error")
        self.assertEqual(numeric_fidelity(src, "We need 4 or 5 bottles of water.", "spanish")["status"],
                         "numeric_error")

    def test_reject_invalid_language(self):
        with self.assertRaises(ValueError):
            numeric_fidelity("un", "one", "german")

    def test_helper_roundtrip(self):
        self.assertEqual(parse_number("0,004%", "fr"), ("0.004", True))
        self.assertEqual(parse_number("0.004%", "en"), ("0.004", True))


if __name__ == "__main__":
    unittest.main()
