"""Validate blinded human-review evidence and reject fabricated or missing ratings."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_reviews import digest, jsonl, prepare, assemble


def examples(count=220):
    categories = ("everyday", "idiom", "slang", "context", "technical")
    return [{"sample_id": f"sample-{i:04d}", "language": "spanish",
             "source_group_sha256": digest(f"conversation-{i // 10}"),
             "category": categories[i % 5],
             "previous_utterances": (["¿Qué dijiste antes?"] if i % 5 == 3 else []),
             "topic": "casual conversation",
             "source_text": f"Spanish source number {i}",
             "candidate_translation": f"English candidate number {i}"}
            for i in range(count)]


def mark(path, *, failures=set(), empty=set()):
    data = jsonl(path)
    for i, row in enumerate(data):
        if row["sample_id"] in failures:
            row["adequate"] = False
        elif row["sample_id"] in empty:
            pass
        else:
            row["adequate"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in data))


class TestHumanPacket(unittest.TestCase):
    def test_two_actual_ratings_trigger_gate_only_on_real_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "frozen"
            sig = prepare(examples(), "a" * 64, folder)
            manifest = json.loads((folder / "manifest.json").read_text())
            self.assertEqual(manifest["evaluation_manifest_sha256"], sig)
            a, b = folder / "rater_a.jsonl", folder / "rater_b.jsonl"
            with self.assertRaisesRegex(ValueError, "judgment"):
                assemble(folder / "manifest.json", a, b)
            mark(a)
            mark(b)
            scores = assemble(folder / "manifest.json", a, b)
            self.assertTrue(scores["meets_90_percent_human_adequacy_gate"])
            self.assertEqual(scores["n"], 220)
            self.assertEqual(scores["adequate"], 220)
            self.assertNotIn("source_text", scores)

    def test_observed_ninety_percent_not_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "packet"
            prepare(examples(200), "b" * 64, folder)
            a, b = folder / "rater_a.jsonl", folder / "rater_b.jsonl"
            failures = {f"sample-{i:04d}" for i in range(20)}
            mark(a, failures=failures)
            mark(b, failures=failures)
            result = assemble(folder / "manifest.json", a, b)
            self.assertEqual(result["observed_fraction"], 0.90)
            self.assertFalse(result["meets_90_percent_human_adequacy_gate"])

    def test_disagreement_requires_independent_arbitration(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "packet"
            prepare(examples(220), "c" * 64, folder)
            a, b = folder / "rater_a.jsonl", folder / "rater_b.jsonl"
            mark(a)
            mark(b, failures={"sample-0004"})
            with self.assertRaisesRegex(ValueError, "third reviewer"):
                assemble(folder / "manifest.json", a, b)
            case = next(x for x in jsonl(b) if x["sample_id"] == "sample-0004")
            case["adequate"] = False
            third = folder / "third.jsonl"
            third.write_text(json.dumps(case) + "\n")
            result = assemble(folder / "manifest.json", a, b, third)
            self.assertEqual(result["disagreements_adjudicated"], 1)
            self.assertEqual(result["adequate"], 219)

    def test_context_is_visible_to_bilingual_raters_and_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "packet"
            prepare(examples(220), "e" * 64, folder)
            rater_a = folder / "rater_a.jsonl"
            rows = jsonl(rater_a)
            contextual = next(row for row in rows if row["category"] == "context")
            self.assertEqual(contextual["previous_utterances"], ["¿Qué dijiste antes?"])
            self.assertEqual(contextual["topic"], "casual conversation")
            mark(rater_a)
            mark(folder / "rater_b.jsonl")
            modified = jsonl(rater_a)
            altered = next(row for row in modified if row["category"] == "context")
            altered["previous_utterances"] = ["¿Qué dijiste después?"]
            rater_a.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                       for row in modified))
            with self.assertRaisesRegex(ValueError, "Altered source, translation or dialogue context"):
                assemble(folder / "manifest.json", rater_a, folder / "rater_b.jsonl")

    def test_context_category_must_have_actual_prior_utterances(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = examples(220)
            rows[3]["previous_utterances"] = []
            with self.assertRaisesRegex(ValueError, "Context category requires"):
                prepare(rows, "a" * 64, Path(tmp) / "packet")
            rows[3]["previous_utterances"] = ["x" * 501]
            with self.assertRaisesRegex(ValueError, "0-4 prior utterances"):
                prepare(rows, "a" * 64, Path(tmp) / "packet")
            rows[3]["previous_utterances"] = "string is not a list"
            with self.assertRaisesRegex(ValueError, "0-4 prior utterances"):
                prepare(rows, "a" * 64, Path(tmp) / "packet")

    def test_tampered_manifest_or_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "packet"
            prepare(examples(220), "d" * 64, folder)
            a, b = folder / "rater_a.jsonl", folder / "rater_b.jsonl"
            mark(a)
            mark(b)
            payload = jsonl(a)
            payload[0]["case_hash"] = "e" * 64
            a.write_text("".join(json.dumps(row) + "\n" for row in payload))
            with self.assertRaisesRegex(ValueError, "altered"):
                assemble(folder / "manifest.json", a, b)
            frozen = json.loads((folder / "manifest.json").read_text())
            frozen["language"] = "french"
            (folder / "manifest.json").write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                assemble(folder / "manifest.json", a, b)


if __name__ == "__main__":
    unittest.main()
