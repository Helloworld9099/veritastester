import json
import tempfile
import unittest
from pathlib import Path

from research.unified_models.registry import ModelRegistry, ROOT, MANIFEST, safe_local_path


class TestUnifiedModelManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = ModelRegistry()

    def test_exact_three_directions_and_language_codes(self):
        expected = {
            "es-en": "Helsinki-NLP/opus-mt-es-en",
            "fr-en": "Helsinki-NLP/opus-mt-fr-en",
            "en-es": "Helsinki-NLP/opus-mt-en-es",
        }
        self.assertEqual({k:v.model for k,v in self.registry.entries.items()}, expected)
        self.assertEqual(self.registry.select("es-MX","en-US").model_id, "es-en")
        self.assertEqual(self.registry.select("fr-FR","en").model_id, "fr-en")
        self.assertEqual(self.registry.select("en-US","es-ES").model_id, "en-es")
        for source,target in (("auto","en"),("fr","es"),("es","fr"),("en","fr"),
                              ("zh","en"),("../es","en")):
            with self.subTest(route=(source,target)), self.assertRaises(ValueError):
                self.registry.select(source,target)

    def test_no_unsafe_nc_or_unverified_production_model(self):
        for spec in self.registry.entries.values():
            self.assertIn("research_only",spec.rights_review_status)
            self.assertFalse(spec.human_meaning_verified)
            self.assertFalse(spec.supports_context)
        active_sha = {entry.active_adapter["sha256"]
                      for entry in self.registry.entries.values()
                      if entry.active_adapter}
        restricted_sha = {archive["sha256"]
                          for archive in self.registry.archived_noncommercial}
        self.assertFalse(active_sha & restricted_sha)
        self.assertEqual(len(restricted_sha),2)

    def test_existing_both_french_and_spanish_are_unchanged_and_rollback_exists(self):
        evidence = self.registry.verify_sources()
        self.assertEqual(evidence["es-en"]["active_sha256"],
                         "3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71")
        self.assertEqual(evidence["fr-en"]["active_sha256"],
                         "fba2bd78958bf0bf8a4c14dc55e601e08663633718c4c9c17f91ef9b52dcfa1c")
        self.assertEqual(evidence["fr-en"]["rollback_sha256"],
                         "072f7f5d2d90b26991c5c62bf8ddb40f686f65e66be52717e7b121d507008ad9")
        self.assertEqual(evidence["es-en"]["rollback_sha256"],
                         "14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b")
        self.assertIsNone(evidence["en-es"]["active_sha256"])
        self.assertNotEqual(evidence["fr-en"]["tokenizer_hashes"]["source.spm"],
                            evidence["es-en"]["tokenizer_hashes"]["source.spm"])

    def test_model_registry_cannot_escape_repository_via_path(self):
        for evil in ("../private.key","/etc/passwd","foo\\bar","x/../../secret"):
            with self.subTest(evil=evil), self.assertRaises(ValueError):
                safe_local_path(ROOT,evil)

    def test_rejects_restricted_adapter_in_active_route(self):
        obj=json.loads(MANIFEST.read_text())
        obj["models"][0]["active_adapter"] = obj["archived_noncommercial"][1].copy()
        obj["models"][0]["active_adapter"].pop("direction",None)
        obj["models"][0]["active_adapter"].pop("reason",None)
        with tempfile.TemporaryDirectory() as tmp:
            altered=Path(tmp)/"unsafe.json"
            altered.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                ModelRegistry(altered, root=ROOT)

    def test_base_manifest_requires_immutable_revision(self):
        obj=json.loads(MANIFEST.read_text())
        obj["models"][1]["base_revision"]="main"
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"unsafe.json"
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                ModelRegistry(p,root=ROOT)


if __name__=="__main__":
    unittest.main()
