"""Test scale-out backend architecture without downloading unsupported models."""
import tempfile
import unittest
from pathlib import Path

from research.global_languages.backend_registry import (
    BackendIdentity,WorldModelBackends
)
from research.global_languages.catalog import DocumentedCatalog
from research.global_languages.tests.test_architecture import glotto_fixture,iana_fixture
from research.global_languages.sync_catalog import build_catalog


class StubBackend:
    def __init__(self,backend_id="m2m100_study"):
        self.identity=BackendIdentity(
            backend_id=backend_id,
            pretrained_model_id="facebook/m2m100_418M",
            pinned_revision="55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636",
            license="mit",modality="text",single_shared_base=True)
    def native_codes(self):
        return frozenset(("en","es","fr"))
    def supports_pair(self,source,target):
        return source in self.native_codes() and target in self.native_codes() and source!=target
    def translate_research(self,text,source,target):
        return source+":"+target+":"+text


class TestExpandableModelFamilies(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p=Path(self.temp.name)
        build_catalog(glotto_fixture(),iana_fixture(),out=p)
        self.world=WorldModelBackends(DocumentedCatalog.load(p))
        self.world.add(StubBackend())

    def test_only_single_available_native_research_backend_is_dispatched(self):
        status=self.world.inspect("en","es")
        self.assertEqual(status.eligible_backends,("m2m100_study",))
        self.assertFalse(status.production_approved)
        self.assertEqual(self.world.dispatch_research("en","es","Hello"),"en:es:Hello")
        self.assertEqual(self.world.stats()["new_production_languages_activated"],0)

    def test_future_new_documented_language_needs_real_model_not_alias(self):
        status=self.world.inspect("glotto:rare1234","en")
        self.assertTrue(status.missing_model)
        self.assertTrue(status.eligible_backends==())
        with self.assertRaises(ValueError):
            self.world.dispatch_research("glotto:rare1234","en","This cannot be faked")
        self.assertTrue(self.world.inspect("en","en").missing_model)

    def test_multiprovider_selection_requires_explicit_choice_and_authorization(self):
        self.world.add(StubBackend("second_approved_study"))
        status=self.world.inspect("en","es")
        self.assertEqual(len(status.eligible_backends),2)
        with self.assertRaises(ValueError):
            self.world.dispatch_research("en","es","Hello")
        self.assertEqual(self.world.dispatch_research(
            "en","es","Hello",backend_id="second_approved_study"),"en:es:Hello")
        with self.assertRaises(PermissionError):
            self.world.production_translate("en","es","Hello")

    def test_restricted_or_unpinned_new_model_cannot_register(self):
        with self.assertRaises(ValueError):
            BackendIdentity("bad","model","main","mit","text",True)
        with self.assertRaises(ValueError):
            BackendIdentity("bad","model","a"*40,"cc-by-nc-4.0","text",True)
        with self.assertRaises(ValueError):
            BackendIdentity("bad","model","a"*40,"mit","text",True,
                            verified_human_meaning=True)
        with self.assertRaises(ValueError):
            BackendIdentity("bad","model","a"*40,"mit","text",True,
                            released_to_production=True)


if __name__=="__main__":
    unittest.main()
