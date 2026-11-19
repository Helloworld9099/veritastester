"""No-network tests: preserve genuine conversation context and test isolation."""
from pathlib import Path
import hashlib
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_context import (
    ROOT,WARM_SHA,WARM_REL,TRAIN_N,UPDATES,DEV_N,
    SOURCE_FILES,parse_chat,add_context,select_group_aware
)


def row(doc,index,lang="fr",source=None,reference=None):
    fr=lang=="fr"
    return {
        "source_language":lang,
        "target_language":"en" if fr else "fr",
        "source":source or (
            "Pouvez vous expliquer cette facture numéro "+str(index) if fr
            else "I can help you with the invoice number "+str(index)),
        "reference":reference or (
            "Could you explain this invoice number "+str(index) if fr
            else "Je peux vous aider avec la facture numéro "+str(index)),
        "doc_id":doc, "client_id":"anonymous-"+doc,
        "sender":"customer" if fr else "agent",
        "tags":"['pronouns']" if index%3==0 else "[]"
    }


class TestContextTraining(unittest.TestCase):
    def test_prior_turns_are_real_prior_sources_not_reference_or_future(self):
        dialogue=[
            row("doc1",1,"fr",source="Je voudrais modifier la facture.",reference="I want to change the invoice."),
            row("doc1",2,"en",source="Which invoice are you referring to?",reference="De quelle facture parlez vous ?"),
            row("doc1",3,"fr",source="Celle de hier s'il vous plaît.",reference="The one from yesterday please."),
            row("doc1",4,"fr",source="Je vous remercie pour votre réponse.",reference="Thank you for your reply."),
        ]
        output=add_context(dialogue)
        self.assertEqual(len(output),3)
        self.assertEqual(output[0]["prior_turns"],0)
        self.assertEqual(output[1]["prior_turns"],2)
        self.assertTrue(output[1]["input_with_context"].startswith(
            "Celle de hier s'il vous plaît."))
        self.assertIn("Je voudrais modifier la facture.",output[1]["input_with_context"])
        self.assertIn("Which invoice are you referring to?",output[1]["input_with_context"])
        self.assertNotIn("I want to change the invoice.",output[1]["input_with_context"])
        self.assertNotIn("Je vous remercie",output[1]["input_with_context"])
        self.assertEqual(output[2]["prior_turns"],2)
        self.assertNotIn("Thank you for your reply.",output[2]["input_with_context"])

    def test_no_cross_conversation_history(self):
        items=[row("a",1),row("b",2),row("a",3),row("b",4)]
        converted=add_context(items)
        self.assertEqual(converted[0]["prior_turns"],0)
        self.assertEqual(converted[1]["prior_turns"],0)
        self.assertEqual(converted[2]["prior_turns"],1)
        self.assertEqual(converted[3]["prior_turns"],1)
        self.assertNotIn("number 2",converted[2]["input_with_context"])

    def test_development_selection_spans_independent_real_groups(self):
        rows=[row("chat-"+str(i//9),i) for i in range(400)]
        data=add_context(rows)
        selected=select_group_aware(data,180,42)
        self.assertEqual(len(selected),180)
        self.assertGreaterEqual(len({x["doc_id"] for x in selected}),40)
        self.assertEqual(selected,select_group_aware(data,180,42))
        self.assertEqual(len({x["source"] for x in selected}),180)

    def test_modified_training_data_schema_rejected(self):
        raw=("source_language,target_language,source\nfr,en,Bonjour\n").encode()
        with self.assertRaisesRegex(ValueError,"column schema"):
            parse_chat(raw)

    def test_warm_model_and_external_source_pins(self):
        self.assertEqual((TRAIN_N,UPDATES,DEV_N),(3840,240,256))
        self.assertEqual(len(SOURCE_FILES),2)
        for name,sha in SOURCE_FILES.values():
            self.assertTrue(name.endswith(".csv"))
            self.assertEqual(len(sha),40)
        warm=ROOT/WARM_REL
        self.assertTrue(warm.is_file())
        self.assertEqual(hashlib.sha256(warm.read_bytes()).hexdigest(),WARM_SHA)


if __name__=="__main__":
    unittest.main()
