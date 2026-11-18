"""Network-free tests: new source/reference pairs and frozen MASSIVE test exclusion."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_stage4 import (GV_SHA256, STAGE3_SHA256, _combine,
                          choose_new_globalvoices, cohort_digest, norm)


def rows(num=85):
    return [(f"J'aimerais prendre le train numero {i} vers le centre ville",
             f"I would like to take the number {i} train to the city centre")
            for i in range(num)]


class StageFourSafety(unittest.TestCase):
    def test_excludes_all_old_globalvoices_and_massive_test(self):
        texts = rows(80)
        used = texts[:12]
        massive_new_test = texts[12:16]
        fresh = choose_new_globalvoices(
            texts, used, massive_new_test,
            train_n=24, dev_n=10, test_n=8)
        self.assertEqual([len(fresh[k]) for k in ("train", "dev", "test")],
                         [24, 10, 8])
        forbidden = {norm(v) for s, _ in used + massive_new_test for v in [s]}
        for cohort in fresh.values():
            self.assertFalse(forbidden & {norm(s) for s, _ in cohort})
            self.assertFalse({norm(t) for _, t in used + massive_new_test} &
                             {norm(e) for _, e in cohort})

        all_fresh = [pair for cohort in fresh.values() for pair in cohort]
        self.assertEqual(len(set(all_fresh)), len(all_fresh))
        self.assertEqual(fresh, choose_new_globalvoices(
            texts, used, massive_new_test, train_n=24, dev_n=10, test_n=8))

    def test_fails_without_enough_truly_fresh_examples(self):
        with self.assertRaisesRegex(ValueError, "Insufficient genuinely new"):
            choose_new_globalvoices(rows(10), rows(9), [],
                                   train_n=2, dev_n=1, test_n=1)

    def test_english_reference_duplicate_alone_blocks_leakage(self):
        initial = rows(60)
        used = [("Different prior French source", initial[0][1])]
        future = choose_new_globalvoices(initial, used, [],
                                         train_n=15, dev_n=8, test_n=8)
        used_en = norm(used[0][1])
        for cohort in future.values():
            self.assertNotIn(used_en, {norm(en) for _, en in cohort})

    def test_combined_previous_cohorts_and_case_ids(self):
        def set_of(prefix):
            return {"selected_ids": {"train": [prefix+"tr"],
                                     "validation": [prefix+"de"], "test": [prefix+"te"]},
                    "train": [(prefix+"fr1", prefix+"en1")],
                    "dev": [(prefix+"fr2", prefix+"en2")],
                    "test": [(prefix+"fr3", prefix+"en3")]}
        group = _combine(set_of("1"), set_of("2"), set_of("3"))
        self.assertEqual(len(group["train"]), 3)
        self.assertEqual(group["selected_ids"]["test"], ["1te", "2te", "3te"])

    def test_immutable_prior_corpus_and_model_pins(self):
        for language in ("french", "spanish"):
            with self.subTest(language=language):
                self.assertEqual(len(GV_SHA256[language]), 64)
                path = (Path(__file__).resolve().parents[3] /
                        "massive_stage3_20261117" / "saved" /
                        language / "adapter_selected.safetensors")
                if path.exists():
                    import hashlib
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                     STAGE3_SHA256[language])
        self.assertEqual(len(cohort_digest(rows(2))), 64)


if __name__ == "__main__":
    unittest.main()
