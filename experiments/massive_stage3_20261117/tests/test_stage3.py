"""CPU-only/no-network stage-three research guardrails."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_stage3 import REPO, WARM_SHA256
import train_massive as massive
import train_stage2 as stage2


def fake_locales(n=42):
    english, target = {}, {}
    for split in ("train", "validation", "test"):
        left, right = [], []
        for idx in range(n):
            key = f"{split}-{idx}"
            part = "dev" if split == "validation" else split
            left.append({"id": key, "partition": part, "intent": "music_play",
                         "utt": f"play relaxing song number {key} again please"})
            right.append({"id": key, "partition": part, "intent": "music_play",
                          "utt": f"pon musica relajante numero {key} otra vez por favor"})
        english[split], target[split] = left, right
    return english, target


def combine(previous, next_run):
    return {
        "selected_ids": {k: previous["selected_ids"][k] + next_run["selected_ids"][k]
                         for k in ("train", "validation", "test")},
        "train": previous["train"] + next_run["train"],
        "dev": previous["dev"] + next_run["dev"],
        "test": previous["test"] + next_run["test"],
    }


class TestStageThreeSafety(unittest.TestCase):
    def test_three_rounds_never_replay_consumed_utterances(self):
        en, src = fake_locales()
        first = massive.select_pairs(en, src, train_size=10, dev_size=8,
                                     test_size=8, include_ids=True)
        second = stage2.next_cohorts(en, src, first, train_n=10, dev_n=8,
                                     test_n=8, seed=202611182, include_ids=True)
        third = stage2.next_cohorts(en, src, combine(first, second),
                                    train_n=10, dev_n=8, test_n=8,
                                    seed=202611183, include_ids=True)
        self.assertEqual([len(third[k]) for k in ("train", "dev", "test")], [10, 8, 8])
        old = {massive.canonical(s) for d in (first, second)
               for split in ("train", "dev", "test") for s, _ in d[split]}
        newest = {massive.canonical(s) for split in ("train", "dev", "test")
                  for s, _ in third[split]}
        self.assertFalse(old & newest)
        self.assertEqual(len(newest), 26)
        # All three rounds have separately named frozen official test IDs.
        self.assertEqual(len(set(first["selected_ids"]["test"] +
                                 second["selected_ids"]["test"] +
                                 third["selected_ids"]["test"])), 24)

    def test_second_stage_id_reconstruction_matches_report_hash_format(self):
        en, src = fake_locales()
        first = massive.select_pairs(en, src, train_size=10, dev_size=8,
                                     test_size=8, include_ids=True)
        second = stage2.next_cohorts(en, src, first, train_n=10, dev_n=8,
                                     test_n=8, seed=202611182, include_ids=True)
        for key in ("train", "validation", "test"):
            self.assertEqual(
                hashlib.sha256("\n".join(second["selected_ids"][key]).encode()).hexdigest(),
                second["new_id_sha256"][key])

    def test_third_stage_rejected_when_no_fresh_test_available(self):
        en, src = fake_locales(12)
        first = massive.select_pairs(en, src, train_size=5, dev_size=5,
                                     test_size=5, include_ids=True)
        second = stage2.next_cohorts(en, src, first, train_n=5, dev_n=5,
                                     test_n=5, seed=42, include_ids=True)
        with self.assertRaisesRegex(ValueError, "Not enough fresh test"):
            stage2.next_cohorts(en, src, combine(first, second),
                                train_n=3, dev_n=3, test_n=5, seed=43)

    def test_saved_stage_two_adapter_hashes_are_fixed(self):
        for language, expected in WARM_SHA256.items():
            with self.subTest(language=language):
                path = (REPO / "experiments/massive_stage2_20261117/saved" /
                        language / "adapter_selected.safetensors")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, expected)


if __name__ == "__main__":
    unittest.main()
