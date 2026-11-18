"""No-network tests for continuation from verified adapters and untouched cohorts."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_stage2 import next_cohorts, verified_previous_run
from train import verified_warm_adapter
from train_massive import select_pairs, canonical


def make_locales(n=48):
    en, translated = {}, {}
    for split in ("train", "validation", "test"):
        a, b = [], []
        for i in range(n):
            identifier = f"{split}-{i}"
            partition = "dev" if split == "validation" else split
            a.append({"id": identifier, "partition": partition,
                      "intent": "set_alarm",
                      "utt": f"set my alarm in location {identifier} please"})
            b.append({"id": identifier, "partition": partition,
                      "intent": "set_alarm",
                      "utt": f"pon mi alarma para lugar {identifier} por favor"})
        en[split] = a
        translated[split] = b
    return en, translated


class TestNewCohorts(unittest.TestCase):
    def test_new_cohorts_preserve_prior_test_and_source(self):
        en, translated = make_locales()
        old = select_pairs(en, translated, train_size=12, dev_size=7,
                           test_size=7, include_ids=True)
        second = next_cohorts(en, translated, old, train_n=12, dev_n=9, test_n=9)
        self.assertEqual((len(second["train"]), len(second["dev"]),
                          len(second["test"])), (12, 9, 9))
        original = {canonical(src) for name in ("train", "dev", "test")
                    for src, _ in old[name]}
        novel = {canonical(src) for name in ("train", "dev", "test")
                 for src, _ in second[name]}
        self.assertFalse(original & novel)
        self.assertEqual(len(novel), 30)
        self.assertTrue(second["no_previous_training_development_or_test_id_reused"])
        self.assertEqual(second, next_cohorts(en, translated, old,
                                              train_n=12, dev_n=9, test_n=9))
        self.assertEqual(set(second["new_id_sha256"]), {"train", "validation", "test"})

    def test_old_id_provenance_hash_matches_original_selection(self):
        en, translated = make_locales()
        old = select_pairs(en, translated, train_size=10, dev_size=5,
                           test_size=6, include_ids=True)
        for key, ids in old["selected_ids"].items():
            self.assertEqual(hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                             old["ids_sha256"][key])

    def test_cannot_train_on_old_test_even_when_data_short(self):
        en, translated = make_locales(9)
        old = select_pairs(en, translated, train_size=5, dev_size=5,
                           test_size=5, include_ids=True)
        with self.assertRaisesRegex(ValueError, "Not enough fresh test"):
            next_cohorts(en, translated, old, train_n=1, dev_n=1, test_n=5)

    def test_modifying_intent_is_not_silently_accepted(self):
        en, translated = make_locales()
        old = select_pairs(en, translated, train_size=5, dev_size=5,
                           test_size=5, include_ids=True)
        translated["train"][15]["intent"] = "wrong_intent"
        with self.assertRaisesRegex(ValueError, "intents changed"):
            next_cohorts(en, translated, old, train_n=8, dev_n=6, test_n=6)


class TestAdapterPins(unittest.TestCase):
    def test_reject_modified_or_external_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            candidate = root / "weights.safetensors"
            candidate.write_bytes(b"test weights")
            spec = {
                "adapter_rel": "legacy.safetensors",
                "warm_adapter_override": {
                    "path": "weights.safetensors",
                    "sha256": hashlib.sha256(b"test weights").hexdigest(),
                },
            }
            self.assertEqual(verified_warm_adapter(root, root, spec), candidate)
            candidate.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verified_warm_adapter(root, root, spec)
            spec["warm_adapter_override"]["path"] = "../outside.safetensors"
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                verified_warm_adapter(root, root, spec)

    def test_versioned_checkpoint_must_match_pinned_digest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "repo"
            directory = root / "experiments/massive_20261117/saved/french"
            directory.mkdir(parents=True)
            (directory / "adapter_selected.safetensors").write_bytes(b"incorrect")
            (directory / "report.json").write_text(
                '{"language":"french","selected":"step256",'
                '"selected_candidate_sha256":"5c5094a3f8dfc9d783f6e8482ae8ed2fad7e632b81b31f20f54fa8f288538ad8"}')
            (directory / "massive_evaluation.json").write_text(
                '{"language":"french","selected_on_dev":"step256",'
                '"revision":"ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",'
                '"source":"AmazonScience/massive","aligned_by_original_slurp_id":true}')
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verified_previous_run("french", root)


if __name__ == "__main__":
    unittest.main()
