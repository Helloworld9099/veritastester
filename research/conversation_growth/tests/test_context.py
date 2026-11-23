"""Context and slang remain review-only until true bilingual refs exist."""
import tempfile
import unittest
from pathlib import Path

from research.conversation_growth.context_gold import (
    VerifiedContextExample,history_only_source,group_split,load_verified_jsonl
)
from research.conversation_growth.mine_openassistant import analyze_rows


class TestRealDialogueEligibility(unittest.TestCase):
    @staticmethod
    def item(convo,turn,approved="independent_bilingual_verified"):
        return VerifiedContextExample(
            conversation_id=convo,turn_id=turn,
            source_language="es",target_language="en",
            current_text="¿Puedes recogerme esta tarde?",
            reference_translation="Can you pick me up this afternoon?",
            previous_utterances=("Estamos cerca de la estación.",),
            review_status=approved,
            dataset_license="CC-BY-4.0",
            dataset_source="test_reviewed_pairs",
        )

    def test_only_approved_refs_and_licensed_context_are_trainable(self):
        item=self.item("a","t0")
        self.assertEqual(item.review_status,"independent_bilingual_verified")
        self.assertIn("<current> ¿Puedes",history_only_source(item))
        self.assertNotIn("Can you pick",history_only_source(item))
        with self.assertRaises(ValueError):
            self.item("a","t1",approved="unverified_no_bilingual_target")
        with self.assertRaises(ValueError):
            VerifiedContextExample(
                conversation_id="a",turn_id="x",source_language="es",
                target_language="en",current_text="Saludos",
                reference_translation="Hello",previous_utterances=(),
                review_status="independent_bilingual_verified",
                dataset_license="CC-BY-NC-4.0",dataset_source="forbidden")

    def test_group_split_is_by_entire_conversation_not_individual_turn(self):
        examples=[self.item(str(i),str(j)) for i in range(12) for j in range(3)]
        train,dev,report=group_split(examples,seed=17)
        self.assertFalse({x.conversation_id for x in train}&
                         {x.conversation_id for x in dev})
        self.assertEqual(len(train)+len(dev),36)
        self.assertTrue(report["all_examples_from_bilingual_review"])
        self.assertFalse(report["final_holdout_used"])

    def test_actual_oasst_mining_does_not_forge_translation_targets(self):
        rows=[
            {"message_id":"a","message_tree_id":"conv1","parent_id":None,
             "lang":"en","role":"prompter","deleted":False,
             "text":"Are you joining the game later tonight?"},
            {"message_id":"b","message_tree_id":"conv1","parent_id":"a",
             "lang":"en","role":"assistant","deleted":False,
             "text":"I'm down! We can play after school."},
            {"message_id":"c","message_tree_id":"conv1","parent_id":"b",
             "lang":"en","role":"prompter","deleted":False,
             "text":"Actually, I meant around six o'clock."},
            {"message_id":"d","message_tree_id":"conv2","parent_id":None,
             "lang":"es","role":"prompter","deleted":False,
             "text":"¿Puedes venir al trabajo mañana?"},
            {"message_id":"e","message_tree_id":"conv2","parent_id":"d",
             "lang":"es","role":"assistant","deleted":False,
             "text":"No me dejó en visto, sino que respondió."},
        ]
        candidates,receipt=analyze_rows(rows)
        self.assertGreaterEqual(len(candidates),2)
        self.assertEqual(receipt["new_verified_parallel_training_examples"],0)
        self.assertFalse(receipt["bilingual_targets_available"])
        self.assertFalse(receipt["future_dialogue_in_context"])
        self.assertTrue(all(row["reference_translation"] is None
                            and not row["eligible_for_training"]
                            and row["review_status"]=="unverified_no_bilingual_target"
                            for row in candidates))
        self.assertTrue(all("Are you joining" not in row["current_text"] or
                            row["source_language"]=="en" for row in candidates))


if __name__=="__main__":
    unittest.main()
