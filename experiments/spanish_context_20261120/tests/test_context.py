"""No-network regression checks for Spanish conversational LoRA research."""
from pathlib import Path
import hashlib
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_context import (
    ROOT, WARM_REL, WARM_SHA, SOURCE_PARTS, SEED,
    parse_srt, align_video, select_balanced, assert_video_and_text_separation,
    group_split, sha_pairs, git_blob_sha, TRAIN_DIALOGUE, TRAIN_REPLAY, UPDATES,
)


def sample(video, i):
    return {
        "video_id": video,
        "source": f"quisiera saber si la estación está abierta número {i}",
        "reference": f"i would like to know if the station is open number {i}",
        "input_with_context": f"quisiera saber si la estación está abierta número {i}",
        "prior_turns": 0,
        "cue_start": float(i),
    }


class TestSpanishContext(unittest.TestCase):
    def test_real_subtitle_timestamps_and_html_cleanup(self):
        es = ("1\n00:00:01,000 --> 00:00:03,000\n<b>¿Dónde está el hotel?</b>\n\n"
              "2\n00:00:03,100 --> 00:00:05,300\nEstá cerca del centro.\n")
        en = ("1\n00:00:01,050 --> 00:00:03,050\nWhere is the hotel?\n\n"
              "2\n00:00:03,150 --> 00:00:05,350\nIt is near downtown.\n")
        es_rows, en_rows = parse_srt(es), parse_srt(en)
        self.assertEqual(len(es_rows), 2)
        self.assertEqual(es_rows[0]["text"], "¿Dónde está el hotel?")
        rows = align_video("video1", es_rows, en_rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prior_turns"], 0)
        self.assertEqual(rows[1]["prior_turns"], 1)
        self.assertIn("¿Dónde está el hotel?", rows[1]["input_with_context"])
        self.assertNotIn("Where is the hotel?", rows[1]["input_with_context"])
        self.assertNotIn("It is near downtown.", rows[0]["input_with_context"])

    def test_different_video_history_is_separate(self):
        es = [{"start": 1., "end": 3., "text": "Quiero comprar un nuevo boleto"},
              {"start": 4., "end": 6., "text": "¿Cuánto cuesta el boleto?"}]
        en = [{"start": 1., "end": 3., "text": "I want to buy a new ticket"},
              {"start": 4., "end": 6., "text": "How much does the ticket cost?"}]
        x = align_video("a", es, en)
        y = align_video("b", es[1:], en[1:])
        self.assertEqual(x[1]["prior_turns"], 1)
        self.assertEqual(y[0]["prior_turns"], 0)

    def test_temporally_distant_or_unequal_subtitles_discarded(self):
        es = [{"start": 10., "end": 11., "text": "Tengo un gran problema con mi cuenta"}]
        en = [{"start": 15., "end": 17., "text": "I need help with this account issue"}]
        self.assertEqual(align_video("a", es, en), [])

    def test_video_disjoint_deterministic_selection(self):
        source = [sample(f"video-{i//4}", i) for i in range(120)]
        a = select_balanced(source, 40, split="test")
        b = select_balanced(source, 40, split="test")
        self.assertEqual(a, b)
        self.assertGreaterEqual(len({x["video_id"] for x in a}), 25)
        self.assertEqual(len(sha_pairs([(r["source"], r["reference"]) for r in a])), 64)

    def test_reject_cross_split_video_and_text_leakage(self):
        a, b, c = sample("video-a", 1), sample("video-b", 2), sample("video-a", 3)
        assert_video_and_text_separation(("test", [a]), ("dev", [b]))
        with self.assertRaisesRegex(RuntimeError, "Video"):
            assert_video_and_text_separation(("test", [a]), ("train", [c]))
        d = sample("other-video", 1)
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            assert_video_and_text_separation(("test", [a]), ("train", [d]))

    def test_pins_sealed_and_model_not_modified(self):
        self.assertEqual(UPDATES * 16, TRAIN_DIALOGUE + TRAIN_REPLAY)
        self.assertEqual(len(SOURCE_PARTS), 2)
        self.assertEqual(SEED, 2026112006)
        for path, sha, length in SOURCE_PARTS:
            self.assertTrue(path.endswith(".jsonl.gz"))
            self.assertEqual(len(sha), 40)
            self.assertGreater(length, 10000000)
        model = ROOT / WARM_REL
        self.assertTrue(model.is_file())
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), WARM_SHA)
        self.assertEqual(git_blob_sha(b"abc"),
                         hashlib.sha1(b"blob 3\0abc").hexdigest())

    def test_split_is_repeatable_for_same_video(self):
        videos = [f"video-test-{i}" for i in range(1500)]
        groups = [group_split(video) for video in videos]
        self.assertEqual(groups, [group_split(video) for video in videos])
        for split in ("train", "dev", "test"):
            self.assertGreater(groups.count(split), 70)


if __name__ == "__main__":
    unittest.main()
