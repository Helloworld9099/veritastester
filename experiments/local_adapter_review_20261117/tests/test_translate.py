"""Checksum/provenance smoke checks without loading neural model weights."""
import sys
import json
import tempfile
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from translate import (MODELS, REPO, expected_book_sha, expected_saved_sha,
                       resolve_weights, validated_sources, validated_rows)


class TestLocalAdapterReview(unittest.TestCase):
    def test_pinned_books_adapters_are_preserved(self):
        expected = {
            "french": "8b2410003e8ead6732870d18963e03bfae45ceb3148ac442acb47917dee0f891",
            "spanish": "fabfaacddb980c47de5352ab6e13b45db326ac1d813906b11280f21a8f123554",
        }
        for language, digest in expected.items():
            self.assertEqual(expected_book_sha(language), digest)
            self.assertTrue(resolve_weights(REPO, language, "books").is_file())

    def test_all_variants_have_explicit_weights(self):
        for language in ("french", "spanish"):
            self.assertIsNone(resolve_weights(REPO, language, "base"))
            self.assertTrue(resolve_weights(REPO, language, "historical").is_file())
            self.assertTrue(resolve_weights(REPO, language, "books").is_file())

    def test_batch_validation_does_not_require_model_load(self):
        self.assertEqual(validated_sources("Bonjour"), (["Bonjour"], False))
        self.assertEqual(validated_sources(["Bonjour", "Salut"]),
                         (["Bonjour", "Salut"], True))
        for bad in ([], ["", "salut"], [7], ["x" * 5001], ["Bonjour"] * 257):
            with self.assertRaisesRegex(ValueError, "1-256"):
                validated_sources(bad)

    def test_private_heldout_records_require_source_group_and_category(self):
        example = {
            "sample_id": "speaker-01-utterance-001",
            "category": "idiom",
            "source_group_sha256": "0" * 64,
            "source_text": "J'ai le cafard",
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.jsonl"
            path.write_text(json.dumps(example) + "\n", encoding="utf-8")
            self.assertEqual(validated_rows(path), [example])
            path.write_text(json.dumps(example) + "\n" +
                            json.dumps(example) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                validated_rows(path)
            example["source_group_sha256"] = "not-a-sha"
            path.write_text(json.dumps(example) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validated_rows(path)

    def test_unknown_models_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            resolve_weights(REPO, "english", "books")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            resolve_weights(REPO, "french", "dev_untrusted")
        self.assertEqual(MODELS, ("base", "historical", "books", "massive",
                                  "qed", "globalvoices"))

    def test_new_saved_adapters_are_pinned_to_evidence(self):
        expected = {
            ("french", "massive"): "5c5094a3f8dfc9d783f6e8482ae8ed2fad7e632b81b31f20f54fa8f288538ad8",
            ("spanish", "massive"): "e60fbadae2037686f9693af4be2e7e59b128aa44f4be8c287f40b7f893b0a004",
            ("spanish", "qed"): "269b67cf8e49dc0da9fe13d47059c57533e61e5ef4db079ff12983e70ea62ab9",
            ("french", "globalvoices"): "670c226bbe16c010940b1def0127225c0a332b7ab1ca0360dd2694f8deaa78bc",
        }
        for (language, variant), sha in expected.items():
            with self.subTest(language=language, variant=variant):
                self.assertEqual(expected_saved_sha(language, variant), sha)
                self.assertTrue(resolve_weights(REPO, language, variant).is_file())

    def test_rejected_models_are_not_implicitly_activated(self):
        with self.assertRaisesRegex(ValueError, "no accepted"):
            resolve_weights(REPO, "french", "qed")
        with self.assertRaisesRegex(ValueError, "no accepted"):
            resolve_weights(REPO, "spanish", "globalvoices")


if __name__ == "__main__":
    unittest.main()
