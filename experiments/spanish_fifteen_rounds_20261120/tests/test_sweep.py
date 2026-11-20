"""Pure-Python tests for the 15 independently trained research rounds."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_sweep import CONFIGS, N_ROUNDS, UPDATES, TOTAL_N, DEV_N, quality_filter


class TestFifteenSpanishRounds(unittest.TestCase):
    def test_exactly_fifteen_unique_planned_experiments(self):
        self.assertEqual(N_ROUNDS, 15)
        self.assertEqual(set(CONFIGS), set(range(1, 16)))
        self.assertEqual(len(set(CONFIGS.values())), 15)

    def test_every_round_is_full_and_trainable(self):
        self.assertEqual(UPDATES, 240)
        self.assertEqual(TOTAL_N, UPDATES * 16)
        self.assertEqual(DEV_N, 128)
        for round_id, (lr, context, replay, bucket) in CONFIGS.items():
            with self.subTest(round_id=round_id):
                self.assertGreater(lr, 0)
                self.assertLess(lr, 1e-3)
                self.assertGreaterEqual(context, 0)
                self.assertLessEqual(context, 1)
                self.assertGreaterEqual(replay, 0)
                self.assertLessEqual(replay, TOTAL_N - 1000)
                self.assertIn(bucket, (0, 64, 128))

    def test_quality_filter_catches_changed_numbers_and_too_long(self):
        row = {"source": "Necesito tres boletos para mañana", 
               "reference": "I need three tickets for tomorrow"}
        self.assertTrue(quality_filter(row, strict=True))
        row["source"] = "Necesito 2 boletos para el lunes"
        row["reference"] = "I need 3 tickets for Monday"
        self.assertFalse(quality_filter(row, strict=False))
        row["source"] = "Hola muy bien"
        row["reference"] = "A fairly verbose and entirely unrelated English subtitle that is much longer than the Spanish"
        self.assertFalse(quality_filter(row, strict=True))

    def test_non_training_import_never_needs_hf_or_gpu(self):
        # Importing training scripts is side-effect free: downloads and model
        # initialization happen only inside the real Actions run.
        import train_sweep
        self.assertTrue(callable(train_sweep.build_data))
        self.assertEqual(train_sweep.CHECKPOINTS, (80, 160, 240))


if __name__ == "__main__":
    unittest.main()
