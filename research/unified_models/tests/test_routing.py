import unittest

from research.unified_models.registry import ModelRegistry
from research.unified_models.runtime import SharedMarianResearchTranslator, _Loaded
from bridgecaption.provider import TranslationRequest
from bridgecaption.providers import create_translator, PROVIDERS


class FakePinnedInference(SharedMarianResearchTranslator):
    """Exercising routing/cache/retry without downloading 3 x 300MB bases."""
    def __init__(self):
        super().__init__(registry=ModelRegistry())
        self.calls=[]

    def _create(self,spec,*,use_previous):
        if use_previous and spec.rollback_adapter is None:
            raise ValueError("No previous version for this route")
        self.load_count+=1
        return _Loaded(spec=spec, key=spec.cache_key(use_previous=use_previous),
                       model=object(),tokenizer=object())

    def _generate(self,loaded,text,*,override=None):
        self.calls.append((loaded.spec.model_id,text,override is not None))
        if text=="Me encanta comer guisantes":
            return ("Yummy! "*32) if override is None else "I love eating peas"
        return {"es-en":"Spanish input translated to English",
                "fr-en":"French input translated to English",
                "en-es":"English input translated to Spanish"}[loaded.spec.model_id]


class TestOneSharedRuntime(unittest.TestCase):
    def test_one_loader_auto_routes_by_lang_and_bounded_model_cache(self):
        engine=FakePinnedInference()
        es=TranslationRequest("Hola",source="es-MX",target="en")
        fr=TranslationRequest("Bonjour",source="fr-FR",target="en")
        en=TranslationRequest("Hello",source="en-US",target="es")
        self.assertIn("Spanish input",engine.translate(es))
        self.assertIn("Spanish input",engine.translate(es))
        self.assertEqual(engine.load_count,1)
        self.assertGreaterEqual(engine.cache_hits,1)
        self.assertIn("French input",engine.translate(fr))
        self.assertEqual(engine.loaded_model_count,1)
        self.assertEqual(engine.load_count,2)
        self.assertIn("English input",engine.translate(en))
        self.assertEqual(engine.loaded_model_count,1)
        self.assertEqual(engine.load_count,3)
        self.assertEqual(engine.last_run["direction"],"en-es")
        self.assertFalse(engine.last_run["human_95_accuracy_verified"])

    def test_rollback_is_same_language_and_not_noncommercial_model(self):
        engine=FakePinnedInference()
        es=TranslationRequest("Hola",source="es",target="en")
        fr=TranslationRequest("Bonjour",source="fr",target="en")
        engine.translate(es,use_previous=True)
        self.assertEqual(engine.last_run["version_sha256"],
                         "14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b")
        self.assertTrue(engine.last_run["rollback_selected"])
        engine.translate(fr,use_previous=True)
        self.assertEqual(engine.last_run["version_sha256"],
                         "072f7f5d2d90b26991c5c62bf8ddb40f686f65e66be52717e7b121d507008ad9")
        with self.assertRaises(ValueError):
            engine.translate(TranslationRequest("Hello",source="en",target="es"),use_previous=True)

    def test_does_not_send_previous_speech_to_untrained_local_model(self):
        engine=FakePinnedInference()
        with self.assertRaises(ValueError):
            engine.translate(TranslationRequest(
                "Está brutal",source="es",target="en",
                previous=["El concierto fue genial"],context_enabled=True))
        with self.assertRaises(ValueError):
            engine.translate(TranslationRequest(
                "Bonjour",source="fr",target="en",
                previous=["Sujet secret"],context_enabled=True))
        self.assertEqual(engine.load_count,0)
        self.assertFalse(engine.supports_context)

    def test_catches_actual_reviewed_loop_only_and_retries_once(self):
        engine=FakePinnedInference()
        result=engine.translate(TranslationRequest("Me encanta comer guisantes",
                source="es",target="en"))
        self.assertEqual(result,"I love eating peas")
        self.assertEqual(engine.repetition_retries,1)
        self.assertEqual(len(engine.calls),2)
        self.assertEqual(engine.calls[0][-1],False)
        self.assertEqual(engine.calls[1][-1],True)
        self.assertTrue(engine.last_run["generation_retry_for_loop"])
        engine.translate(TranslationRequest("Bonjour",source="fr",target="en"))
        self.assertEqual(len(engine.calls),3)

    def test_no_automatic_silent_routing_or_fake_merged_model(self):
        engine=FakePinnedInference()
        for src,tgt in (("fr","es"),("en","fr"),("auto","en")):
            with self.subTest(src=src,tgt=tgt),self.assertRaises(ValueError):
                engine.translate(TranslationRequest("bonjour",source=src,target=tgt))
        self.assertEqual(engine.load_count,0)
        self.assertEqual(len(engine.routes()),3)
        self.assertTrue(all(not x["model_ready_for_production"] for x in engine.routes()))

    def test_default_application_provider_unchanged_and_not_replaced_by_research(self):
        self.assertNotIn("local_marian",PROVIDERS)
        import os
        previous=os.environ.get("BRIDGE_TRANSLATOR")
        try:
            os.environ["BRIDGE_TRANSLATOR"]="gemini"
            from bridgecaption.provider import GeminiTranslator
            self.assertIsInstance(create_translator(),GeminiTranslator)
        finally:
            if previous is None: os.environ.pop("BRIDGE_TRANSLATOR",None)
            else: os.environ["BRIDGE_TRANSLATOR"]=previous


if __name__=="__main__":
    unittest.main()
