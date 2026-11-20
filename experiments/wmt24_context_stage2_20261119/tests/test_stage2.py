"""No-network tests: training really uses fresh WMT24 turns, never publisher test."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_stage2 import REPO, WARM_REL, WARM_SHA, new_cohort
from train_context import add_context, select_group_aware


def row(doc, index):
    return {
        "doc_id": doc, "sender": "customer",
        "source_language": "fr", "target_language": "en",
        "source": f"Je voudrais savoir le numéro de facture du dossier {doc} entrée {index}",
        "reference": f"I would like to know the invoice number for file {doc} entry {index}",
        "client_id": "anonymous", "tags": "[]"
    }


class TestStageTwo(unittest.TestCase):
    def test_no_old_dev_or_training_text_reused(self):
        all_rows = [row(f"chat-{i // 15}", i) for i in range(240)]
        first = add_context(all_rows)
        old = select_group_aware(first, 100, 1)
        new = new_cohort(first, old, 70, seed=2)
        self.assertEqual(len(new), 70)
        old_src = {x["source"] for x in old}
        old_eng = {x["reference"] for x in old}
        self.assertFalse(old_src & {x["source"] for x in new})
        self.assertFalse(old_eng & {x["reference"] for x in new})
        self.assertEqual(new, new_cohort(first, old, 70, seed=2))

    def test_prevents_english_reference_overlap(self):
        all_rows = [row(f"chat-{i // 15}", i) for i in range(120)]
        prior = [{"source": "different phrase", "reference": all_rows[0]["reference"]}]
        new = new_cohort(add_context(all_rows), prior, 80, seed=11)
        self.assertNotIn(all_rows[0]["reference"], {x["reference"] for x in new})

    def test_new_sample_shortage_fails_before_optimizer(self):
        all_rows = add_context([row(f"group-{i // 10}", i) for i in range(50)])
        with self.assertRaisesRegex(ValueError, "distinct WMT24"):
            new_cohort(all_rows, all_rows[:45], 20, seed=3)

    def test_provenance_and_warm_checkpoint(self):
        file = REPO / WARM_REL
        self.assertTrue(file.is_file())
        self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), WARM_SHA)

    def test_context_is_real_source_history_not_future_or_reference(self):
        rows = [row("c", i) for i in range(4)]
        converted = add_context(rows)
        self.assertEqual(converted[0]["prior_turns"], 0)
        self.assertEqual(converted[2]["prior_turns"], 2)
        self.assertIn(rows[0]["source"], converted[2]["input_with_context"])
        self.assertNotIn(rows[3]["source"], converted[2]["input_with_context"])
        self.assertNotIn(rows[1]["reference"], converted[2]["input_with_context"])


if __name__ == "__main__":
    unittest.main()
