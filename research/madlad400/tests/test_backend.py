"""Test multilingual research backend gates without downloading 3B weights."""
import tempfile
import unittest
from pathlib import Path

from research.madlad400.backend import MADLAD400ResearchBackend
from research.madlad400.tests.test_model_contract import FakeTokenizer
from research.global_languages.catalog import DocumentedCatalog


class TestMADLADResearchOnlyRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=DocumentedCatalog.load()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine=MADLAD400ResearchBackend(
            self.catalog,tokenizer=FakeTokenizer(),
            experiment=Path(self.tmp.name))

    def test_natively_known_but_unverified_text_routes(self):
        self.assertTrue(self.engine.supports_pair("en","es"))
        self.assertTrue(self.engine.supports_pair("fr","en"))
        self.assertTrue(self.engine.supports_pair("es","fr"))
        self.assertFalse(self.engine.supports_pair("fr","fr"))
        self.assertFalse(self.engine.supports_pair("es-MX","en"))
        self.assertFalse(self.engine.supports_pair("glotto:aari1239","en"))
        self.assertGreater(len(self.engine.native_codes()),400)
        status=self.engine.experimental_status()
        self.assertEqual(status["model_loads"],0)
        self.assertFalse(status["production_enabled"])
        self.assertFalse(status["independently_verified_meaning_accuracy"])

    def test_no_fake_trained_model_without_actual_model_weights(self):
        with self.assertRaisesRegex(RuntimeError,"training is not yet verified"):
            self.engine.translate_research("Hola","es","en")
        self.assertEqual(self.engine.model_loads,0)


if __name__=="__main__":
    unittest.main()
