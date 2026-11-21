"""Mocked network-independent proof that all six requests share one model."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from research.multilingual_student.runtime import (
    SUPPORTED_DIRECTIONS, SingleMultilingualTextTranslator
)


class TestOneResidentModelRatherThanRoutingLegacyCheckpoint(unittest.TestCase):
    def test_all_six_language_pairs_use_same_loaded_model(self):
        engine=SingleMultilingualTextTranslator()
        net=object()
        tokenizer=object()
        load_calls=[]
        def shared_loader():
            if engine._loaded is None:
                engine._loaded=(net,tokenizer)
                engine._load_count+=1
            load_calls.append(engine._loaded)
            return engine._loaded
        engine._load_once=shared_loader
        results=[]
        with patch("research.multilingual_student.runtime.generate",
                   side_effect=lambda model,tok,text,src,tgt: f"{src}-{tgt}:{text}") as call:
            for pair in sorted(SUPPORTED_DIRECTIONS):
                src,tgt=pair.split("-")
                request=SimpleNamespace(source=src,target=tgt,text="hello",
                                        context_enabled=False,previous=())
                # A fake resident test cannot read a trained report yet, so
                # create a local, isolated receipt fixture only.
                from tempfile import TemporaryDirectory
                from pathlib import Path
                import json
                with TemporaryDirectory() as td:
                    folder=Path(td)
                    (folder/"report.json").write_text(json.dumps({
                        "checkpoint_sha256":"a"*64,
                        "status":"completed_actual_single_multilingual_student_training"}))
                    engine.experiment=folder
                    results.append(engine.translate(request))
        self.assertEqual(set(x.split(":")[0] for x in results),SUPPORTED_DIRECTIONS)
        self.assertEqual(engine.model_load_count,1)
        self.assertEqual(engine.loaded_model_count,1)
        self.assertEqual(len(load_calls),6)
        self.assertTrue(all(x==(net,tokenizer) for x in load_calls))
        self.assertEqual(call.call_count,6)
        self.assertFalse(engine.status()["production_activated"])
        self.assertFalse(engine.status()["speech_recognition_integrated"])

    def test_unknown_route_does_not_load_model(self):
        engine=SingleMultilingualTextTranslator()
        with self.assertRaises(ValueError):
            engine.translate(SimpleNamespace(source="zh",target="fr",
                                            text="hello",context_enabled=False,previous=()))
        self.assertEqual(engine.model_load_count,0)

    def test_untrained_context_refused_before_loading(self):
        engine=SingleMultilingualTextTranslator()
        with self.assertRaises(ValueError):
            engine.translate(SimpleNamespace(source="es",target="en",
                                            text="Estoy abajo",context_enabled=True,
                                            previous=["Quieres jugar?"]))
        self.assertEqual(engine.model_load_count,0)


if __name__=="__main__":
    unittest.main()
