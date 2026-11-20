import json
import os
import tempfile
import unittest
from pathlib import Path

from research.es_en_v2.quality import (
    Example, assess, audit, near_duplicate, review_near_duplicates
)
from research.es_en_v2.review_packet import (
    create_blank_review_packet, seal, validate_review, wilson_interval
)


def case(i, split="train", **updates):
    row = {
        "case_id": f"c{i}", "source": f"Necesito ayuda con el recibo numero {i}",
        "reference": f"I need help with receipt number {i}",
        "source_language": "es", "target_language": "en",
        "group_id": f"g{i}", "speaker_id": f"speaker{i}",
        "document_id": f"doc{i}", "split": split, "dataset": "test-licensed",
        "license_id": "CC-BY-4.0", "original_id": str(i),
        "rights_verified": True, "provenance_url": "https://example.org/dataset",
        "verified_parallel": True, "previous_source": (),
    }
    row.update(updates)
    return Example(**row)


class TestV2Quality(unittest.TestCase):
    def test_permissive_license_and_alignment_required(self):
        case(1).validate()
        for bad in ("CC-BY-NC-4.0", "research-only", "unknown"):
            with self.assertRaises(ValueError):
                case(1, license_id=bad).validate()
        with self.assertRaises(ValueError):
            case(1, rights_verified=False).validate()
        with self.assertRaises(ValueError):
            case(1, verified_parallel=False).validate()

    def test_real_human_holdout_cannot_be_synthetic(self):
        with self.assertRaises(ValueError):
            case(1, split="final_holdout", is_synthetic=True).validate()

    def test_reject_exact_cross_split_overlap(self):
        train = case(1)
        dev = case(2, split="dev", source=train.source)
        with self.assertRaisesRegex(ValueError, "exact text collision"):
            audit([train, dev])

    def test_reject_cross_split_related_groups(self):
        for key in ("group_id", "speaker_id", "document_id"):
            with self.subTest(key=key):
                train = case(1)
                dev = case(2, split="dev", **{key: getattr(train, key)})
                with self.assertRaisesRegex(ValueError, "leakage"):
                    audit([train, dev])

    def test_numbers_and_private_text_flagged_not_certified(self):
        obj = case(1, source="Mi correo es someone@example.com y tengo 25 dólares",
                   reference="My address is someone@example.com and I have 52 dollars")
        flags = assess(obj)
        self.assertIn("digit_mismatch_needs_bilingual_review", flags)
        self.assertIn("possible_personal_information_source", flags)

    def test_near_duplicate_needs_explicit_review(self):
        a = case(1, split="train", source="Necesito una factura para esta compra")
        b = case(2, split="dev", source="Necesito una factura para esta compra!")
        self.assertTrue(near_duplicate(a.source, b.source))
        self.assertEqual(review_near_duplicates([a, b]), [("c1", "c2")])
        with self.assertRaises(ValueError):
            review_near_duplicates([a, b, case(3)], max_pairs=2)

    def test_no_text_in_manifest_receipt(self):
        report = audit([case(1), case(2, split="dev")])
        self.assertEqual(report["rows"], {"train": 1, "dev": 1})
        self.assertNotIn("Necesito ayuda", json.dumps(report))


class TestReviewPacket(unittest.TestCase):
    def test_score_interval_is_human_only_math(self):
        lo, hi = wilson_interval(95, 100)
        self.assertLess(lo, .95)
        self.assertGreater(hi, .95)
        with self.assertRaises(ValueError):
            wilson_interval(101, 100)

    def test_seal_requires_private_path_and_is_non_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data/private"
            root.mkdir(parents=True)
            corpus = root / "unseen.jsonl"
            corpus.write_bytes(b'{"case_id":"hidden"}\n')
            manifest = root / "receipt.json"
            outcome = seal(corpus, manifest)
            self.assertEqual(outcome["trained_model_access_count"], 0)
            self.assertEqual(outcome["human_evaluation_status"], "not_started")
            self.assertNotIn("hidden", manifest.read_text())
            with self.assertRaises(FileExistsError):
                seal(corpus, manifest)

    def test_blind_human_packets_do_not_label(self):
        cases = [
            {"case_id": "alpha", "direction": "es-en", "category": "context",
             "group_id": "dialogue1", "source": "Está brutal",
             "previous_source": ["Me gustó mucho el concierto"],
             "frozen_system_outputs": {"baseline": "It's brutal", "trained": "It's amazing"}},
            {"case_id": "beta", "direction": "en-es", "category": "idiom",
             "group_id": "dialogue2", "source": "Break a leg",
             "frozen_system_outputs": {"baseline": "Rómpete una pierna", "trained": "Mucha suerte"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            packet = Path(tmp)/"packets"
            info = create_blank_review_packet(cases, ("baseline", "trained"), packet)
            self.assertEqual(info["human_labels_written"], 0)
            self.assertEqual(info["rows_per_reviewer"], 4)
            self.assertEqual((packet/"blind_rater_1.csv").read_bytes(),
                             (packet/"blind_rater_2.csv").read_bytes())
            self.assertNotIn("trained", (packet/"blind_rater_1.csv").read_text())
            self.assertIn("trained", (packet/"system_key_private.csv").read_text())
            self.assertEqual(os.stat(packet/"blind_rater_1.csv").st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                create_blank_review_packet(cases, ("baseline", "trained"), packet)

    def test_wrong_label_or_critical_pass_cannot_be_accepted(self):
        with self.assertRaises(ValueError):
            validate_review({"adequate": "pass", "critical_error": "yes", "error_types": "negation"})
        with self.assertRaises(ValueError):
            validate_review({"adequate": "maybe", "critical_error": "no", "error_types": ""})


if __name__ == "__main__":
    unittest.main()
