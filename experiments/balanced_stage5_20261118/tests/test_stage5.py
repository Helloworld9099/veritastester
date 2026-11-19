"""Fast no-network regression tests for stage-five training/test separation."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_stage5 import (
    REPO, SEED, SOURCE_SHA, WARM_SHA, assert_no_overlap,
    combine_massive, normalize, pick_replay
)


class TestBalancedStageFive(unittest.TestCase):
    def rows(self, n=60):
        return [(f"quiero escuchar musica original numero {i}",
                 f"please play my original song number {i}") for i in range(n)]

    def test_replay_must_only_use_allowed_old_train(self):
        train = self.rows(80)
        previous_tests = train[:12]
        previous_dev = train[12:16]
        selection = pick_replay(train, previous_tests + previous_dev, count=25)
        self.assertEqual(len(selection), 25)
        self.assertFalse(set(selection) & set(previous_tests + previous_dev))
        self.assertTrue(set(selection) <= set(train))
        self.assertEqual(selection, pick_replay(train, previous_tests + previous_dev,
                                                 count=25))

    def test_also_reject_reuse_of_english_reference(self):
        old = self.rows(60)
        old_tests = [("new source text", old[0][1])]
        accepted = pick_replay(old, old_tests, count=12)
        self.assertNotIn(old[0], accepted)

    def test_cannot_replay_test_when_short_of_safe_training_rows(self):
        with self.assertRaisesRegex(ValueError, "Insufficient safe"):
            pick_replay(self.rows(5), self.rows(4), count=4)

    def test_cross_domain_test_training_overlap_fails_closed(self):
        test = self.rows(10)
        valid = self.rows(20)[10:20]
        train = self.rows(40)[20:40]
        assert_no_overlap(("frozen test", test),
                          ("validation", valid),
                          ("train", train))
        with self.assertRaisesRegex(RuntimeError, "collision"):
            assert_no_overlap(("test", test),
                              ("different surface same target",
                               [("different Spanish words", test[0][1])]))

    def test_mixed_replay_does_not_duplicate_identifiers(self):
        def round_(prefix):
            return {
                "selected_ids": {key: [prefix + key]
                                 for key in ("train","validation","test")},
                "train": [(prefix + "source train", prefix + "english train")],
                "dev": [(prefix + "source dev", prefix + "english dev")],
                "test": [(prefix + "source test", prefix + "english test")],
            }
        used = combine_massive(round_("a"), round_("b"), round_("c"))
        self.assertEqual(used["selected_ids"]["test"], ["atest","btest","ctest"])
        self.assertEqual(len(used["train"]), 3)
        self.assertEqual(len(used["test"]), 3)

    def test_both_historical_warm_weights_and_corpus_sha_are_pinned(self):
        for language in ("french", "spanish"):
            with self.subTest(language=language):
                path = (REPO / "experiments/massive_stage3_20261117/saved" /
                        language / "adapter_selected.safetensors")
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 WARM_SHA[language])
                self.assertEqual(len(SOURCE_SHA[language]),64)
                prev = (REPO / "experiments/globalvoices_stage4_20261118/saved" /
                        language / "stage4_evaluation.json")
                self.assertTrue(prev.is_file())


if __name__ == "__main__":
    unittest.main()
