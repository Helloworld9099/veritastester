"""Synthetic-only checks for the offline per-direction CPU receipt gate."""

import copy
import unittest

from research.unified_eval.performance import compare_performance, input_mix_sha256


FIXTURE = "f" * 64
MIX = input_mix_sha256(["First   caption", "A second caption"] * 20)
BASE_PINS = {
    "base_revision": "a" * 40,
    "files": {"model.safetensors": "b" * 64, "vocab.json": "c" * 64},
    "adapter_sha256": "d" * 64,
    "rollback_sha256": "e" * 64,
}


def _receipt(side, *, latency=100, rss=1000):
    return {
        "schema_version": 1,
        "direction": "es->en",
        "checkpoint": copy.deepcopy(BASE_PINS),
        "fixture_sha256": FIXTURE,
        "input_mix_sha256": MIX,
        "host": {
            "fingerprint_sha256": "8" * 64,
            "cpu": "Example CPU",
            "logical_cpus": 8,
            "memory_bytes": 16 * 1024**3,
            "os": "Linux test fixture",
            "kernel": "example-release",
            "architecture": "x86_64",
            "cpu_affinity": [2, 3, 4, 5],
            "power_policy": "same recorded governor",
            "runtime_fingerprint_sha256": "9" * 64,
            "isolated": True,
        },
        "decode": {"num_beams": 4, "max_new_tokens": 128, "do_sample": False},
        "threads": {"intraop": 4, "interop": 1},
        "methodology": {
            "device": "cpu",
            "workload": "text-only-translation",
            "timer": "monotonic-ns",
            "rss": "peak-process-rss-bytes",
            "isolation": "fresh-process-per-arm-repetition",
            "pairing": "alternating-ab-ba",
            "warmup_requests": 5,
        },
        "warm_runs": [
            {
                "pair_id": f"pair-{i}",
                "position": 1 + ((i + side) % 2),
                "requests_ns": [latency] * 40,
                "peak_rss_bytes": rss,
                "pressure_ok": True,
            }
            for i in range(3)
        ],
    }


def _compare(a, b, **kwargs):
    options = {
        "baseline_pins": BASE_PINS,
        "candidate_pins": BASE_PINS,
        "change_kind": "infrastructure",
        "expected_direction": "es->en",
        "expected_fixture_sha256": FIXTURE,
        "expected_input_mix_sha256": MIX,
        "expected_warm_requests": 40,
    }
    options.update(kwargs)
    return compare_performance(a, b, **options)


class CPUPerformanceTests(unittest.TestCase):
    def test_input_mix_preserves_order_punctuation_whitespace_and_repeats(self):
        self.assertEqual(input_mix_sha256(["é", "hello"]), input_mix_sha256(["é", "hello"]))
        self.assertNotEqual(input_mix_sha256(["é", "hello"]), input_mix_sha256(["hello", "é"]))
        self.assertNotEqual(input_mix_sha256(["é", "hello"]), input_mix_sha256(["é", "hello", "hello"]))
        self.assertNotEqual(input_mix_sha256(["é", "hello"]), input_mix_sha256(["e\u0301", "hello"]))
        self.assertNotEqual(input_mix_sha256(["é", "hello"]), input_mix_sha256(["é", " hello"]))
        with self.assertRaises(ValueError):
            input_mix_sha256("not an ordered list of strings")

    def test_exactly_ten_percent_passes_and_one_unit_more_fails_either_metric(self):
        a = _receipt(0)
        actual = _compare(a, _receipt(1, latency=110, rss=1100))
        self.assertEqual(actual["status"], "pass")
        self.assertEqual(actual["metrics"]["warm_p95_ns"]["regression_fraction"], 0.1)
        self.assertEqual(actual["metrics"]["peak_process_rss_bytes"]["regression_fraction"], 0.1)
        one_slower = _compare(a, _receipt(1, latency=111, rss=1100))
        self.assertEqual(one_slower["status"], "fail")
        self.assertEqual(one_slower["metrics"]["peak_process_rss_bytes"]["status"], "pass")
        one_higher_rss = _compare(a, _receipt(1, latency=110, rss=1101))
        self.assertEqual(one_higher_rss["status"], "fail")
        self.assertEqual(one_higher_rss["metrics"]["warm_p95_ns"]["status"], "pass")

    def test_both_arms_cannot_pad_the_independently_pinned_request_count(self):
        before, after = _receipt(0), _receipt(1)
        for receipt in (before, after):
            for run in receipt["warm_runs"]:
                run["requests_ns"] += [100] * 21
        result = _compare(before, after)
        self.assertEqual(result["status"], "incomparable")
        self.assertIn("sample count", " ".join(result["reasons"]))

    def test_p95_uses_nearest_rank_and_rss_uses_max_seen_process(self):
        a, b = _receipt(0), _receipt(1)
        for r in b["warm_runs"]:
            r["requests_ns"] = [100] * 38 + [1100] * 2
        for i, r in enumerate(b["warm_runs"]):
            r["peak_rss_bytes"] += i * 10
        actual = _compare(a, b)
        self.assertEqual(actual["status"], "pass")
        self.assertEqual(actual["metrics"]["warm_p95_ns"]["candidate"], 100)
        self.assertEqual(actual["metrics"]["peak_process_rss_bytes"]["candidate"], 1020)
        for r in b["warm_runs"]:
            r["requests_ns"] = [100] * 37 + [111] * 3
        self.assertEqual(_compare(a, b)["metrics"]["warm_p95_ns"]["candidate"], 111)

    def test_mixed_or_noisy_repeat_evidence_is_not_a_pass_or_numeric_failure(self):
        a, b = _receipt(0), _receipt(1, latency=109)
        b["warm_runs"][0]["requests_ns"] = [111] * 40
        r = _compare(a, b)
        self.assertEqual(r["status"], "incomparable")
        self.assertIn("threshold", " ".join(r["metrics"]["warm_p95_ns"]["reasons"]))
        b["warm_runs"][1]["requests_ns"] = [170] * 40
        r = _compare(a, b)
        self.assertIn("noisy", " ".join(r["metrics"]["warm_p95_ns"]["reasons"]))

    def test_rejects_noisy_metadata_short_unqualified_or_wrongly_paired_runs(self):
        a = _receipt(0)
        mutations = (
            (lambda b: b["warm_runs"].pop(), "independent warm"),
            (lambda b: b["warm_runs"][0]["requests_ns"].pop(), "positive request timings"),
            (lambda b: b["warm_runs"][1]["requests_ns"].append(100), "sample count"),
            (lambda b: b["warm_runs"][1].update(pair_id="pair-0"), "pair identifier"),
            (lambda b: b["warm_runs"][1].update(pair_id="another-pair"), "pair IDs"),
            (lambda b: b["warm_runs"][1].update(position=a["warm_runs"][1]["position"]), "alternating"),
            (lambda b: b["warm_runs"][1].update(pressure_ok=False), "pressure"),
            (lambda b: b["warm_runs"][1].update(peak_rss_bytes=True), "RSS"),
        )
        for mutate, reason in mutations:
            with self.subTest(reason=reason):
                b = _receipt(1)
                mutate(b)
                result = _compare(a, b)
                self.assertEqual(result["status"], "incomparable")
                self.assertIn(reason, " ".join(result["reasons"]))
                self.assertFalse(result["metrics"])

    def test_host_decode_threads_expected_mix_and_methodology_must_match(self):
        a = _receipt(0)
        mutations = (
            (lambda b: b["host"].update(fingerprint_sha256="7" * 64), "host or runtime identity"),
            (lambda b: b["host"].update(runtime_fingerprint_sha256="7" * 64), "host or runtime identity"),
            (lambda b: b["host"].update(cpu_affinity=[0, 1, 2, 3]), "host or runtime identity"),
            (lambda b: b["threads"].update(intraop=2), "thread counts"),
            (lambda b: b["decode"].update(do_sample=0), "decode configuration"),
            (lambda b: b["methodology"].update(warmup_requests=7), "measurement methodology"),
            (lambda b: b.update(direction="fr->en"), "direction"),
            (lambda b: b.update(fixture_sha256="7" * 64), "expected frozen suite"),
            (lambda b: b.update(input_mix_sha256="7" * 64), "expected workload"),
        )
        for mutate, reason in mutations:
            with self.subTest(reason=reason):
                b = _receipt(1)
                mutate(b)
                result = _compare(a, b)
                self.assertEqual(result["status"], "incomparable")
                self.assertIn(reason, " ".join(result["reasons"]))

    def test_exact_pins_gate_infrastructure_but_model_change_may_use_independently_pinned_new_adapter(self):
        a, b = _receipt(0), _receipt(1)
        new = copy.deepcopy(BASE_PINS)
        new["adapter_sha256"] = "7" * 64
        b["checkpoint"] = copy.deepcopy(new)
        self.assertEqual(_compare(a, b)["status"], "incomparable")
        self.assertIn("identical", " ".join(_compare(a, b, candidate_pins=new)["reasons"]))
        accepted = _compare(a, b, candidate_pins=new, change_kind="model")
        self.assertEqual(accepted["status"], "pass")
        self.assertEqual(accepted["model_pin_changes"], ["adapter_sha256"])
        self.assertEqual(_compare(a, b, change_kind="model")["status"], "incomparable")
        b["checkpoint"]["files"]["vocab.json"] = "6" * 64
        self.assertEqual(_compare(a, b, candidate_pins=new, change_kind="model")["status"], "incomparable")

    def test_missing_performance_never_passes_cold_and_audio_cannot_stand_in_for_it(self):
        a, b = _receipt(0), _receipt(1)
        b["cold_runs"] = [{"load_ns": 42, "first_request_ns": 30, "peak_rss_bytes": 1500}]
        b["audio_runs"] = [{"end_to_end_ns": 1100, "fixture_sha256": "a" * 64}]
        r = _compare(a, b)
        self.assertEqual(r["status"], "pass")
        self.assertEqual(r["cold"]["baseline"]["status"], "not_measured")
        self.assertEqual(r["cold"]["candidate"]["model_load_median_ns"], 42)
        self.assertEqual(r["audio"]["candidate"]["audio_to_caption_p95_ns"], 1100)
        del b["warm_runs"]
        self.assertEqual(_compare(a, b)["status"], "incomparable")
        self.assertEqual(_compare(a, None)["status"], "incomparable")


if __name__ == "__main__":
    unittest.main()
