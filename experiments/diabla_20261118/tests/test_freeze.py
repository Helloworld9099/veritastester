"""No-network tests: DiaBLa test-only dialogue IDs, bilingual direction and schema."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freeze_external_holdout import freeze, group_from_id, validate_row


def item(group, idx, lang):
    return {
        "id": f"{group}_{idx}", "orig": "Bonjour mon ami",
        "mt": "Hello friend", "ref": "Hello, my friend",
        "utterance_meta": {"lang": lang, "eval_judgment": "medium"},
        "dialogue_history": [],
    }


class TestFrozenDialogues(unittest.TestCase):
    def test_grouping_preserves_all_turns(self):
        example = item("dialogue-abc_french_english_1", 3, "french")
        self.assertEqual(group_from_id(example), "dialogue-abc_french_english_1")
        self.assertEqual(validate_row(example),
                         ("dialogue-abc_french_english_1", "french"))

    def test_requires_original_and_reference(self):
        example = item("a", 0, "french")
        del example["ref"]
        with self.assertRaisesRegex(ValueError, "Missing original"):
            validate_row(example)

    def test_no_untagged_dialogue(self):
        example = item("a", 0, "french")
        example["utterance_meta"]["lang"] = "german"
        with self.assertRaisesRegex(ValueError, "speaker-language"):
            validate_row(example)

    def test_frozen_whole_real_groups(self):
        rows = []
        for i in range(145):
            for j in range(35):
                rows.append(item(f"real-dialogue-{i}", j,
                                 "french" if j % 2 else "english"))
        frozen = freeze(rows, "a" * 40)
        self.assertEqual(frozen["n_dialogue_groups"], 145)
        self.assertEqual(frozen["n_utterances"], len(rows))
        self.assertGreater(frozen["french_original_source_utterances"], 1000)
        self.assertTrue(frozen["all_utterances_test_only_no_train_or_dev"])
        self.assertEqual(frozen["frozen_group_ids_sha256"],
                         freeze(rows, "a"*40)["frozen_group_ids_sha256"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            freeze(rows + [rows[0]], "a"*40)

    def test_invalid_revision_fails(self):
        with self.assertRaisesRegex(ValueError, "immutable"):
            freeze([], "latest")


if __name__ == "__main__":
    unittest.main()
