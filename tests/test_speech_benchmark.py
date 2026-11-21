"""Synthetic metadata tests for local benchmark plumbing; no audio or inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from research.unified_models.speech_benchmark import BenchmarkError, compare, validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _man(directions=("es->en", "cmn->en"), clips=4):
    return {
        "schema_version": 1, "kind": "bridgecaption_speech_benchmark_manifest",
        "scope": "development", "sealed": False,
        "rights": {"status": "reviewed_for_development_benchmark", "access": "public_licensed",
                   "review_sha256": _hash("synthetic rights pointer")},
        "audio": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
        "profile": {"host_sha256": _hash("host"), "method_sha256": _hash("method"),
                    "cpu_threads": 2, "device": "cpu",
                    "latency_basis": "source_end_to_stable_translation",
                    "memory_basis": "peak_local_process_tree_rss_bytes"},
        "models": {
            "incumbent": {"kind": "unchanged_app_pipeline", "identity_sha256": _hash("incumbent")},
            "candidate": {"kind": "one_shared_streaming_speech_checkpoint",
                          "identity_sha256": _hash("candidate")},
        },
        "budget": {"cpu_seconds": 30, "wall_seconds": 60, "peak_rss_bytes": 100_000_000,
                   "repetitions": 3},
        "directions": {
            name: {"clips": clips, "pcm_sha256": _hash(name+"pcm"),
                   "order_sha256": _hash(name+"order"), "boundaries_sha256": _hash(name+"cuts"),
                   "references_sha256": _hash(name+"refs"),
                   "asr_metric": "cer" if name == "cmn->en" else "wer"}
            for name in directions
        },
    }


def _arm(manifest, arm, plan_sha=None):
    receipt = {
        "schema_version": 1, "kind": "bridgecaption_speech_benchmark_receipt",
        "scope": "development", "sealed": False, "arm": arm,
        "manifest_sha256": plan_sha or _hash("plan"),
        "identity_sha256": manifest["models"][arm]["identity_sha256"],
        "profile": deepcopy(manifest["profile"]),
        "resources": {"cpu_seconds": 9.0, "wall_seconds": 12.0, "peak_rss_bytes": 50_000_000},
        "directions": {},
    }
    first = arm == "incumbent"
    for name, paired in manifest["directions"].items():
        metric = paired["asr_metric"]
        binding = {key: value for key, value in paired.items() if key != "asr_metric"}
        receipt["directions"][name] = {
            **binding,
            "asr": {"metric": metric, "edits": 10, "reference_units": 100},
            "translation": {"metric": "chrf2", "chrf": 72.125,
                            "scorer_sha256": _hash("chrf-settings")},
            "timing": [
                {"repetition_id": f"trial-{i}", "position": 1 + int((i % 2 == 0) != first),
                 "stable_ms": [300] * paired["clips"], "first_caption_ms": None}
                for i in range(3)
            ],
        }
    return receipt


def _compare(plan, before, after):
    return compare(plan, manifest_sha256=_hash("plan"), incumbent=before, candidate=after)


def test_comparable_development_receipts_are_not_a_model_choice():
    plan = _man()
    out = _compare(plan, _arm(plan, "incumbent"), _arm(plan, "candidate"))
    assert out["status"] == "reported_within_bounds"
    assert out["human_adequacy"] == "not_evaluated"
    assert not out["model_selected"] and not out["promotion_eligible"]
    assert out["directions"]["cmn->en"]["asr"]["metric"] == "cer"
    assert out["directions"]["es->en"]["timing"]["timed_clips_per_arm"] == 12


def test_a_stronger_direction_does_not_cancel_another_regression():
    plan = _man()
    before, after = _arm(plan, "incumbent"), _arm(plan, "candidate")
    after["directions"]["es->en"]["translation"]["chrf"] = 92
    after["directions"]["cmn->en"]["translation"]["chrf"] = 69
    out = _compare(plan, before, after)
    assert out["status"] == "fail"
    assert out["directions"]["es->en"]["translation"]["status"] == "reported_within_bounds"
    assert out["directions"]["cmn->en"]["translation"]["status"] == "fail"


@pytest.mark.parametrize("field", ["pcm_sha256", "order_sha256", "boundaries_sha256", "references_sha256"])
def test_changed_pairing_hash_is_rejected(field):
    plan = _man()
    after = _arm(plan, "candidate")
    after["directions"]["es->en"][field] = _hash("changed")
    with pytest.raises(BenchmarkError, match="receipt_does_not_use_paired_audio"):
        _compare(plan, _arm(plan, "incumbent"), after)


def test_incomparable_profile_and_candidate_identity_are_rejected():
    man = _man()
    before, after = _arm(man, "incumbent"), _arm(man, "candidate")
    trial = deepcopy(after)
    trial["profile"]["host_sha256"] = _hash("different host")
    with pytest.raises(BenchmarkError, match="receipt_manifest_model_or_profile_mismatch"):
        _compare(man, before, trial)
    trial = deepcopy(after)
    trial["identity_sha256"] = man["models"]["incumbent"]["identity_sha256"]
    with pytest.raises(BenchmarkError, match="receipt_manifest_model_or_profile_mismatch"):
        _compare(man, before, trial)


def test_tiny_or_unmatched_warm_blocks_do_not_verify():
    small = _man(clips=1)
    out = _compare(small, _arm(small, "incumbent"), _arm(small, "candidate"))
    assert out["status"] == "not_verified"
    assert out["directions"]["es->en"]["timing"]["reason"] == "too_few_paired_timing_observations"
    plan = _man()
    before = _arm(plan, "incumbent")
    after = _arm(plan, "candidate")
    after["directions"]["es->en"]["timing"][0]["position"] = 1
    out = _compare(plan, before, after)
    assert out["status"] == "not_verified"
    assert out["directions"]["es->en"]["timing"]["reason"] == "matching_alternating_ab_ba_trials_required"


def test_missing_direction_or_changed_reference_count_is_not_verified():
    plan = _man()
    after = _arm(plan, "candidate")
    del after["directions"]["cmn->en"]
    after["directions"]["es->en"]["asr"]["reference_units"] = 200
    out = _compare(plan, _arm(plan, "incumbent"), after)
    assert out["status"] == "not_verified"
    assert out["directions"]["es->en"]["asr"]["reason"] == "comparable_asr_counts_required"
    assert out["directions"]["cmn->en"]["translation"]["status"] == "not_verified"


def test_cpu_wall_and_peak_process_tree_memory_caps_are_not_overridden():
    plan = _man()
    before, after = _arm(plan, "incumbent"), _arm(plan, "candidate")
    after["resources"].update(cpu_seconds=22, wall_seconds=55, peak_rss_bytes=101_000_000)
    out = _compare(plan, before, after)
    assert out["status"] == "fail"
    assert set(out["resources"]["reasons"]) == {"combined_run_budget_exceeded", "peak_rss_regressed"}
    before["resources"] = None
    out = _compare(plan, before, after)
    assert out["status"] == "fail" and "available_arm_alone_exceeded_combined_budget" in out["resources"]["reasons"]


def test_no_holdout_unreviewed_rights_or_raw_audio_fields():
    plan = _man()
    for update in ({"sealed": True}, {"scope": "holdout"},
                   {"rights": {"status": "not_reviewed", "access": "public_licensed",
                               "review_sha256": _hash("pointer")}},
                   {"audio_path": "secret-local-recording.wav"}):
        with pytest.raises(BenchmarkError):
            validate_manifest({**plan, **update})


def test_real_cli_reports_digest_bound_metadata_and_redacts_tamper(tmp_path):
    plan = _man()
    def write(name, data):
        path = tmp_path / name
        text = json.dumps(data, sort_keys=True) + "\n"
        path.write_text(text, encoding="utf-8")
        return path, _hash(text)

    p, digest = write("secret-plan-name.json", plan)
    a, a_digest = write("secret-existing-name.json", _arm(plan, "incumbent", digest))
    b, b_digest = write("secret-one-candidate.json", _arm(plan, "candidate", digest))
    cmd = [sys.executable, "-m", "research.unified_models.speech_benchmark"]
    required = ["--manifest", str(p), "--manifest-sha256", digest]
    extra = ["--incumbent", str(a), "--incumbent-sha256", a_digest,
             "--candidate", str(b), "--candidate-sha256", b_digest]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    def run(command):
        result = subprocess.run(command, cwd=ROOT, env=env, timeout=10, text=True,
                                capture_output=True, check=False)
        assert str(tmp_path) not in (result.stdout + result.stderr)
        assert "secret-" not in (result.stdout + result.stderr)
        return result

    result = run([*cmd, "preflight", *required])
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "protocol_only_not_measured"
    result = run([*cmd, "compare", *required, *extra])
    assert result.returncode == 0 and json.loads(result.stdout)["status"] == "reported_within_bounds"
    b.write_bytes(b.read_bytes() + b" ")
    result = run([*cmd, "compare", *required, *extra])
    assert result.returncode == 2 and json.loads(result.stderr)["status"] == "rejected"

@pytest.mark.parametrize(("candidate_times", "status", "reason"), [
    ([327, 333, 327], "not_verified", "paired_repeats_disagree_at_latency_threshold"),
    ([250, 400, 300], "not_verified", "per_arm_timing_repeats_too_noisy"),
    ([340, 340, 340], "fail", None),
])
def test_noisy_or_threshold_straddling_timings_never_pass(candidate_times, status, reason):
    plan = _man()
    a, b = _arm(plan, "incumbent"), _arm(plan, "candidate")
    for trial, value in zip(b["directions"]["es->en"]["timing"], candidate_times):
        trial["stable_ms"] = [value] * 4
    result = _compare(plan, a, b)
    timing = result["directions"]["es->en"]["timing"]
    assert timing["status"] == result["status"] == status
    if reason:
        assert reason in timing["reasons"]


def test_boolean_thread_count_is_not_equal_to_integer_profile():
    plan = _man()
    plan["profile"]["cpu_threads"] = 1
    b = _arm(plan, "candidate")
    b["profile"]["cpu_threads"] = True
    with pytest.raises(BenchmarkError, match="receipt_manifest_model_or_profile_mismatch"):
        _compare(plan, _arm(plan, "incumbent"), b)
