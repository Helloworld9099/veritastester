import unittest

from research.es_en_v2.baby_dialogue import (
    parse_turns,parse_record,split_for_group,summarize_records,
)
from research.es_en_v2.dev_mining import issue_tags,mine_dev

SPANISH = "**Dad**: Mira ese perro grande.\n\n**Child**: ¿Dónde está el perro?"
ENGLISH = "**Dad**: Look at that big dog.\n\n**Child**: Where is the dog?"


def datum(group="group_1"):
    return {"custom_id":group, "content_en":ENGLISH, "content_es":SPANISH,
            "parent":"dad","context":"community",
            "content_intersent":"","content_intrasent":""}


class TestBilingualBabyDialogue(unittest.TestCase):
    def test_source_and_reference_turns_align_with_earlier_context_only(self):
        r=parse_record(datum(),direction="es-en")
        self.assertEqual(r["status"],"candidate_for_review")
        self.assertEqual(len(r["rows"]),2)
        self.assertEqual(r["rows"][0]["previous_source_turns"],[])
        self.assertEqual(r["rows"][1]["previous_source_turns"],
                         [{"speaker":"Dad","source":"Mira ese perro grande."}])
        self.assertEqual(r["rows"][1]["source"],"¿Dónde está el perro?")
        self.assertEqual(r["rows"][1]["reference_candidate"],"Where is the dog?")
        self.assertFalse(r["rows"][1]["eligible_for_training"])
        self.assertFalse(r["rows"][1]["semantics_independently_verified"])
        self.assertFalse(r["rows"][1]["permitted_as_human_gold"])

    def test_direction_reversal_uses_same_group(self):
        a=parse_record(datum(),direction="es-en")
        b=parse_record(datum(),direction="en-es")
        self.assertEqual(a["split"],b["split"])
        self.assertEqual(b["rows"][1]["source"],"Where is the dog?")
        self.assertEqual(b["rows"][1]["reference_candidate"],"¿Dónde está el perro?")

    def test_reject_potential_dialogue_misalignment(self):
        row=datum()
        row["content_es"]="**Dad**: Mira un perro."
        self.assertEqual(parse_record(row)["reason"],"turn_count_or_marker_mismatch")
        row=datum()
        row["content_es"]="**Child**: Mira ese perro grande.\n**Dad**: ¿Dónde está el perro?"
        self.assertEqual(parse_record(row)["reason"],"speaker_alignment")

    def test_group_split_deterministic_and_source_text_not_leaked_to_summary(self):
        example=datum("another-group")
        summary=summarize_records([example])
        self.assertEqual(summary["inspected_group_count"],1)
        self.assertEqual(summary["counts"]["candidate_turns"],2)
        self.assertEqual(summary["eligible_for_training_without_bilingual_quality_review"],0)
        self.assertNotIn("perro",str(summary))
        self.assertEqual(split_for_group("another-group"),split_for_group("another-group"))
        with self.assertRaises(ValueError):
            summarize_records([example,example])


class TestHardExampleMining(unittest.TestCase):
    def test_flags_number_negation_but_not_human_meaning_accuracy(self):
        flags=issue_tags("No eran 15", "It was not 15", "It was 50",
                         source_language="es",target_language="en")
        self.assertIn("number_disagreement",flags)
        self.assertIn("negation_disagreement",flags)
        self.assertNotIn("human_pass",flags)

    def test_source_context_is_clue_not_proof(self):
        tags=issue_tags("Está brutal", "It is amazing", "It is violent",
                        source_language="es",target_language="en",
                        prior_source=["Fue un concierto buenísimo"])
        self.assertIn("contains_slang_needs_review",tags)
        self.assertIn("context_sensitivity_needs_review",tags)

    def test_forbid_final_holdout_in_miner(self):
        with self.assertRaises(ValueError):
            mine_dev([{"case_id":"z","split":"final_holdout"}])

    def test_dev_outputs_only_rank_suspected_errors(self):
        rows=[{
            "case_id":"candidate-1", "source":"No eran 15",
            "reference":"It was not 15", "hypothesis":"It was 50",
            "split":"dev","source_language":"es","target_language":"en",
        }]
        result=mine_dev(rows)
        self.assertEqual(result["input_dev_n"],1)
        self.assertEqual(result["flagged_n"],1)
        self.assertTrue(result["no_final_holdout_accessed"])
        self.assertEqual(result["human_confirmed_translation_errors"],0)


if __name__=="__main__":
    unittest.main()
