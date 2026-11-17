"""Checksum/provenance smoke checks without loading neural model weights."""
import sys
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from translate import MODELS, REPO, expected_book_sha, resolve_weights


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

    def test_unknown_models_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            resolve_weights(REPO, "english", "books")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            resolve_weights(REPO, "french", "dev_untrusted")
        self.assertEqual(MODELS, ("base", "historical", "books"))


if __name__ == "__main__":
    unittest.main()
