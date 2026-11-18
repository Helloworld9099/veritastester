"""Pure-Python tests for author-split preservation and bilingual alignment."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_massive import select_pairs


def locales(n=25):
    en, foreign = {}, {}
    for split in ("train", "validation", "test"):
        en_rows, foreign_rows = [], []
        for idx in range(n):
            key = split + "." + str(idx)
            split_label = "dev" if split == "validation" else split
            en_rows.append({"id": key, "partition": split_label,
                            "intent": "alarm_set",
                            "utt": f"please wake me tomorrow in region {key}"})
            foreign_rows.append({"id": key, "partition": split_label,
                                 "intent": "alarm_set",
                                 "utt": f"por favor despertame manana en region {key}"})
        en[split] = en_rows
        foreign[split] = foreign_rows
    return en, foreign


class TestMassiveIntegrity(unittest.TestCase):
    def test_official_partitions_and_aligned_id(self):
        english, foreign = locales()
        selected = select_pairs(english, foreign, train_size=10, dev_size=8, test_size=8)
        self.assertEqual([len(selected[k]) for k in ("train", "dev", "test")],
                         [10, 8, 8])
        self.assertTrue(selected["original_splits_preserved"])
        for key in selected["ids_sha256"]:
            self.assertEqual(len(selected["ids_sha256"][key]), 64)
        self.assertTrue(all(t.startswith("please wake") for _, t in selected["test"]))
        self.assertEqual(selected,
                         select_pairs(english, foreign, train_size=10, dev_size=8, test_size=8))

    def test_fails_if_english_target_ids_differ(self):
        english, foreign = locales()
        foreign["train"][0]["id"] = "different"
        with self.assertRaisesRegex(ValueError, "Misaligned"):
            select_pairs(english, foreign, train_size=1, dev_size=1, test_size=1)

    def test_fails_if_intent_disagrees(self):
        english, foreign = locales()
        foreign["train"][0]["intent"] = "weather_query"
        with self.assertRaisesRegex(ValueError, "intent annotations disagree"):
            select_pairs(english, foreign, train_size=1, dev_size=1, test_size=1)

    def test_does_not_reuse_heldout_sentence_in_train(self):
        english, foreign = locales()
        # Duplicate across partitions is excluded from training rather than test.
        english["train"][0]["utt"] = english["test"][0]["utt"]
        foreign["train"][0]["utt"] = foreign["test"][0]["utt"]
        selected = select_pairs(english, foreign, train_size=20, dev_size=8, test_size=8)
        self.assertEqual(len(selected["train"]), 20)
        self.assertNotIn((foreign["train"][0]["utt"], english["train"][0]["utt"]),
                         selected["train"])

    def test_reject_duplicate_ids_in_original_partition(self):
        english, foreign = locales()
        english["train"].append(dict(english["train"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate sample ID"):
            select_pairs(english, foreign, train_size=1, dev_size=1, test_size=1)

    def test_reject_shortage(self):
        english, foreign = locales(2)
        with self.assertRaisesRegex(ValueError, "Not enough"):
            select_pairs(english, foreign, train_size=3, dev_size=1, test_size=1)


if __name__ == "__main__":
    unittest.main()
