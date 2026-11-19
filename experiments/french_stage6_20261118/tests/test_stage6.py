"""No-network checks: stage-six historical data bounds and frozen model identity."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_stage6 import (
    ROOT, WARM_SHA, WARM_REL, MASSIVE_REPLAY, FRESH_NEWS,
    pre_stage5_news, SEED, TEST_N,
)
import train_globalvoices as voices
import train_stage4 as stage4
import train_stage5 as stage5


class TestStageSixNoLeakage(unittest.TestCase):
    def test_exact_historical_gv_slicing_preserves_prior_rounds(self):
        count1 = voices.TRAIN_COUNT + voices.DEV_COUNT + voices.TEST_COUNT
        count4 = stage4.TRAIN_COUNT + stage4.DEV_COUNT + stage4.GV_TEST_COUNT
        count5 = stage5.NEW_NEWS_COUNT + stage5.GV_DEV + stage5.TEST_COUNT
        rows = [(f"source-{i}",f"target-{i}") for i in range(count1+count4+count5)]
        past = pre_stage5_news(rows)
        self.assertEqual(len(past), count1+count4)
        self.assertEqual(past[-1], rows[count1+count4-1])
        self.assertNotIn(rows[-1],past)
        with self.assertRaisesRegex(RuntimeError,"Historical GlobalVoices stage-count"):
            pre_stage5_news(rows[:-1])

    def test_stage_six_selected_warm_adapter_is_persisted_and_pinned(self):
        model = ROOT / WARM_REL
        self.assertTrue(model.is_file())
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), WARM_SHA)
        self.assertEqual((MASSIVE_REPLAY, FRESH_NEWS, TEST_N), (1536,1536,256))

    def test_cross_domain_test_source_and_reference_duplicates_fail(self):
        old = [("Bonjour ca va", "Hello how are you")]
        with self.assertRaisesRegex(RuntimeError,"collision"):
            stage5.assert_no_overlap(
                ("frozen test",old),
                ("training", [("Je vais bien", "Hello how are you")]))
        with self.assertRaisesRegex(RuntimeError,"collision"):
            stage5.assert_no_overlap(
                ("frozen test",old),
                ("training", [("Bonjour ca va", "I feel great")]))
        stage5.assert_no_overlap(
            ("frozen test",old),("training",[("Salut toi","Hi friend")]))

    def test_source_cohort_seed_fixed_for_reproduction(self):
        self.assertEqual(SEED, 202611186)


if __name__ == "__main__":
    unittest.main()
