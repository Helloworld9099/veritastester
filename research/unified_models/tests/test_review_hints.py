import json
import unittest

from bridgecaption.context import slang_hints
from bridgecaption.provider import TranslationRequest, prompt_for, SYSTEM_TRANSLATION_RULES


class TestReviewDrivenContextHints(unittest.TestCase):
    def test_ambiguous_regional_spanish_keeps_multiple_senses(self):
        hints=slang_hints("¿Cogemos la guagua?","es-ES")
        text=" ".join(hints).lower()
        self.assertIn("bus",text)
        self.assertIn("baby",text)
        self.assertIn("canary",text)
        self.assertEqual(slang_hints("¿Cogemos la guagua?","auto"),[])

    def test_reviewed_slang_errors_have_nonliteral_hints(self):
        examples=[
            ("Me quedé sin lana.","es-MX","money"),
            ("Me dejó en visto.","es","on read"),
            ("Se puso las pilas.","es","motivated"),
            ("Está de pelos.","es","awesome"),
            ("He ghosted me.","en","stopped communicating"),
            ("I'm cooked.","en","in trouble"),
            ("That song slaps.","en","excellent"),
            ("I'm down.","en","agree"),
            ("Spill the tea.","en","gossip"),
            ("Break a leg!","en","good luck"),
        ]
        for sentence,language,expected in examples:
            with self.subTest(text=sentence):
                self.assertIn(expected," ".join(slang_hints(sentence,language)).lower())

    def test_maybe_literal_meaning_remains_allowed(self):
        hints=" ".join(slang_hints("Está brutal", "es")).lower()
        self.assertIn("violence",hints)
        hints=" ".join(slang_hints("I'm down.", "en")).lower()
        self.assertIn("sadness",hints)

    def test_french_hints_unaffected(self):
        self.assertTrue(slang_hints("C'est ouf","fr-FR"))
        self.assertEqual(slang_hints("Bonjour","fr"),[])

    def test_existing_context_serialization_preserves_current_only_and_untrusted_data(self):
        r=TranslationRequest("¿Cogemos la guagua?",source="es-ES",target="en",
                             previous=["Estamos en Tenerife y necesitamos un autobús."],
                             context_enabled=True)
        encoded=prompt_for(r)
        parsed=json.loads(encoded.split("\n",1)[1])
        self.assertEqual(parsed["current_transcript"],r.text)
        self.assertEqual(parsed["optional_context"]["previous_utterances"],r.previous)
        self.assertTrue(parsed["optional_context"]["possible_slang_interpretations"])
        self.assertIn("Translate only the current utterance",SYSTEM_TRANSLATION_RULES)

    def test_disabled_context_does_not_embed_history_or_slang_guesses(self):
        r=TranslationRequest("Guagua",source="es",target="en",
                             previous=["private"],context_enabled=False)
        obj=json.loads(prompt_for(r).split("\n",1)[1])
        self.assertEqual(obj["optional_context"],{})


if __name__=="__main__":
    unittest.main()
