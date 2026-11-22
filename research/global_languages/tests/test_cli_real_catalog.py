"""Real imported 8k-language catalog + pinned native tokenizer CLI smoke."""
import io
import json
import unittest
from contextlib import redirect_stdout
from research.global_languages.cli import main


def run_cli(args):
    output=io.StringIO()
    with redirect_stdout(output):
        main(args)
    return json.loads(output.getvalue())


class TestGenuineWorldCatalogCLI(unittest.TestCase):
    def test_catalog_and_audit_are_really_present_and_distinct(self):
        stats=run_cli(["--summary"])
        self.assertEqual(stats["documented_languages"],8623)
        self.assertEqual(stats["documented_dialects"],13773)
        self.assertEqual(stats["native_m2m100_language_codes"],100)
        self.assertEqual(stats["independently_human_validated_pairs"],0)

    def test_name_search_does_not_promise_translation_quality(self):
        data=run_cli(["--search","Yoruba","--limit","5"])
        self.assertTrue(data["results"])
        self.assertTrue(data["these_are_documented_not_translation_verified"])
        self.assertTrue(any("yoruba" in x["name"].lower() for x in data["results"]))

    def test_native_requested_dialect_is_not_human_verified(self):
        data=run_cli(["--source","es-MX","--target","fr"])
        self.assertTrue(data["can_run_native_model"])
        self.assertEqual(data["model_coverage"],"native_variant_unvalidated")
        self.assertFalse(data["production_approved"])

    def test_rare_documented_language_does_not_get_false_native_model(self):
        data=run_cli(["--source","glotto:aari1239","--target","en"])
        self.assertTrue(data["source_documented"])
        self.assertFalse(data["can_run_native_model"])
        self.assertEqual(data["model_coverage"],"catalog_only")


if __name__=="__main__":
    unittest.main()
