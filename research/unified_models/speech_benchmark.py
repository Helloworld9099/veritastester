"""Bounded, offline comparison of redacted speech benchmark receipts.

No audio, checkpoints, model runtime, or subprocess is opened. A SHA pin binds
bytes supplied by an independent operator; it is not proof that an experiment,
consent review, or measurement actually happened. This is development evidence,
not a selection, final evaluation, or deployment decision.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from decimal import Decimal
import json
import math
from pathlib import Path
import re
from statistics import median
import sys

from research.workflow_contract import ContractError, is_sha256, read_pinned_json

_MANIFEST = "bridgecaption_speech_benchmark_manifest"
_RECEIPT = "bridgecaption_speech_benchmark_receipt"
_DIR = re.compile(r"([a-z]{2,3})->([a-z]{2,3})\Z")
_REPETITION = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}\Z")
_BIND = {"clips", "pcm_sha256", "order_sha256", "boundaries_sha256", "references_sha256"}
_AUDIO = {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1}
_RESOURCE_FIELDS = {"cpu_seconds", "wall_seconds", "peak_rss_bytes"}
_MAX_COMBINED_CPU_SECONDS = 7200
_MAX_COMBINED_WALL_SECONDS = 3600
_MAX_RSS_BYTES = 16 * 1024 ** 3
_LATENCY_REGRESSION = Decimal("0.10")
_ASR_REGRESSION_PP = Decimal("0.5")
_CHRF_REGRESSION = Decimal("0.5")
_MIN_TIMED_CLIPS_PER_ARM = 12
_MAX_REPEAT_RANGE = Decimal("0.10")
_MAX_PAIRED_RATIO_RANGE = Decimal("0.10")


class BenchmarkError(ValueError):
    """Policy failure with a fixed, redacted code."""


def _require(condition, code):
    if not condition:
        raise BenchmarkError(code)


def _shape(value, fields, code):
    _require(isinstance(value, Mapping) and set(value) == fields, code)


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _number(value, minimum, maximum):
    return type(value) in (int, float) and minimum <= value <= maximum and math.isfinite(value)


def _direction(name):
    match = _DIR.fullmatch(name) if isinstance(name, str) else None
    return bool(match and match[1] != match[2])


def _dev(record, kind, fields):
    _shape(record, {"schema_version", "kind", "scope", "sealed"} | fields,
           "invalid_speech_evidence_fields")
    _require(type(record["schema_version"]) is int and record["schema_version"] == 1
             and record["kind"] == kind, "unsupported_speech_evidence_schema")
    _require(record["scope"] == "development" and record["sealed"] is False,
             "speech_benchmark_is_open_dev_only")


def _bind(record):
    _require(_integer(record["clips"], 1, 64), "direction_clip_count_out_of_bounds")
    _require(all(is_sha256(record[k]) for k in _BIND - {"clips"}), "paired_audio_digest_required")


def validate_manifest(record):
    """Validate the explicit CPU-only protocol, without reading its referenced assets."""
    _dev(record, _MANIFEST, {"rights", "audio", "profile", "models", "budget", "directions"})
    _require(record["audio"] == _AUDIO
             and all(type(record["audio"][k]) is int for k in ("sample_rate_hz", "channels")),
             "unsupported_pcm_contract")
    rights = record["rights"]
    _shape(rights, {"status", "access", "review_sha256"}, "invalid_rights_metadata")
    _require(rights["status"] == "reviewed_for_development_benchmark"
             and rights["access"] in ("public_licensed", "local_consent_authorized")
             and is_sha256(rights["review_sha256"]), "reviewed_audio_rights_required")
    profile = record["profile"]
    _shape(profile, {"host_sha256", "method_sha256", "cpu_threads", "device",
                     "latency_basis", "memory_basis"}, "invalid_comparison_profile")
    _require(is_sha256(profile["host_sha256"]) and is_sha256(profile["method_sha256"])
             and _integer(profile["cpu_threads"], 1, 32) and profile["device"] == "cpu"
             and profile["latency_basis"] == "source_end_to_stable_translation"
             and profile["memory_basis"] == "peak_local_process_tree_rss_bytes",
             "unsupported_cpu_comparison_profile")
    models = record["models"]
    _shape(models, {"incumbent", "candidate"}, "exactly_two_arms_required")
    for name, style in (("incumbent", "unchanged_app_pipeline"),
                        ("candidate", "one_shared_streaming_speech_checkpoint")):
        identity = models[name]
        _shape(identity, {"kind", "identity_sha256"}, "invalid_arm_identity")
        _require(identity["kind"] == style and is_sha256(identity["identity_sha256"]),
                 "unchanged_incumbent_and_one_candidate_required")
    _require(models["incumbent"]["identity_sha256"] != models["candidate"]["identity_sha256"],
             "candidate_and_incumbent_identities_must_differ")
    budget = record["budget"]
    _shape(budget, _RESOURCE_FIELDS | {"repetitions"}, "explicit_budget_required")
    _require(_integer(budget["cpu_seconds"], 1, _MAX_COMBINED_CPU_SECONDS)
             and _integer(budget["wall_seconds"], 1, _MAX_COMBINED_WALL_SECONDS)
             and _integer(budget["peak_rss_bytes"], 1, _MAX_RSS_BYTES)
             and _integer(budget["repetitions"], 3, 5), "budget_out_of_bounds")
    dirs = record["directions"]
    _require(isinstance(dirs, Mapping) and 1 <= len(dirs) <= 8, "explicit_direction_set_required")
    for name, paired in dirs.items():
        _require(_direction(name), "invalid_direction")
        _shape(paired, _BIND | {"asr_metric"}, "invalid_paired_audio_binding")
        _bind(paired)
        _require(paired["asr_metric"] in ("wer", "cer"), "explicit_direction_asr_metric_required")
    return record


def _quality(row, paired):
    if row["asr"] is not None:
        asr = row["asr"]
        _shape(asr, {"metric", "edits", "reference_units"}, "invalid_asr_evidence")
        _require(asr["metric"] == paired["asr_metric"]
                 and _integer(asr["edits"], 0, 10**8)
                 and _integer(asr["reference_units"], 1, 10**8), "invalid_asr_evidence")
    if row["translation"] is not None:
        t = row["translation"]
        _shape(t, {"metric", "chrf", "scorer_sha256"}, "invalid_translation_evidence")
        _require(t["metric"] == "chrf2" and _number(t["chrf"], 0, 100)
                 and is_sha256(t["scorer_sha256"]), "invalid_translation_evidence")


def _timing(timing, count, repeats):
    if timing is None:
        return
    _require(isinstance(timing, list) and len(timing) == repeats, "timing_repetition_count_mismatch")
    tracked = set()
    for sample in timing:
        _shape(sample, {"repetition_id", "position", "stable_ms", "first_caption_ms"},
               "invalid_timing_trial")
        identity = sample["repetition_id"]
        _require(isinstance(identity, str) and _REPETITION.fullmatch(identity) is not None
                 and identity not in tracked and _integer(sample["position"], 1, 2),
                 "nonunique_or_invalid_timing_trial")
        tracked.add(identity)
        for key in ("stable_ms", "first_caption_ms"):
            values = sample[key]
            if key == "first_caption_ms" and values is None:
                continue
            _require(isinstance(values, list) and len(values) == count
                     and all(_number(x, 0, 3_600_000) for x in values),
                     "unpaired_or_invalid_timing_samples")


def _validate_receipt(receipt, manifest, manifest_sha, arm):
    _dev(receipt, _RECEIPT, {"arm", "manifest_sha256", "identity_sha256", "profile", "resources",
                             "directions"})
    _require(receipt["arm"] == arm and receipt["manifest_sha256"] == manifest_sha
             and receipt["identity_sha256"] == manifest["models"][arm]["identity_sha256"]
             and receipt["profile"] == manifest["profile"]
             and type(receipt["profile"]["cpu_threads"]) is int,
             "receipt_manifest_model_or_profile_mismatch")
    if receipt["resources"] is not None:
        r = receipt["resources"]
        _shape(r, _RESOURCE_FIELDS, "invalid_resource_evidence")
        _require(_number(r["cpu_seconds"], 0, 10**8) and _number(r["wall_seconds"], 0, 10**8)
                 and r["cpu_seconds"] > 0 and r["wall_seconds"] > 0
                 and _integer(r["peak_rss_bytes"], 1, 2**50), "invalid_resource_evidence")
    dirs = receipt["directions"]
    _require(isinstance(dirs, Mapping) and set(dirs) <= set(manifest["directions"]),
             "receipt_claims_unplanned_direction")
    for name, row in dirs.items():
        paired = manifest["directions"][name]
        _shape(row, _BIND | {"asr", "translation", "timing"}, "invalid_direction_evidence")
        _bind(row)
        _require(all(row[k] == paired[k] for k in _BIND), "receipt_does_not_use_paired_audio")
        _quality(row, paired)
        _timing(row["timing"], paired["clips"], manifest["budget"]["repetitions"])


def _d(x):
    return Decimal(str(x))


def _p95(samples):
    return sorted(samples)[math.ceil(len(samples) * .95) - 1]


def _asr(before, after):
    if before is None or after is None or before["reference_units"] != after["reference_units"]:
        return {"status": "not_verified", "reason": "comparable_asr_counts_required"}
    n = before["reference_units"]
    sources = [100 * Decimal(a["edits"]) / Decimal(n) for a in (before, after)]
    delta = sources[1] - sources[0]
    return {"status": "fail" if delta > _ASR_REGRESSION_PP else "reported_within_bounds",
            "metric": before["metric"], "incumbent_percent": float(sources[0]),
            "candidate_percent": float(sources[1]), "delta_percentage_points": float(delta),
            "reference_units": n}


def _translation(before, after):
    if before is None or after is None or before["scorer_sha256"] != after["scorer_sha256"]:
        return {"status": "not_verified", "reason": "same_translation_scorer_required"}
    delta = _d(after["chrf"]) - _d(before["chrf"])
    return {"status": "fail" if delta < -_CHRF_REGRESSION else "reported_within_bounds",
            "incumbent_chrf": before["chrf"], "candidate_chrf": after["chrf"], "delta": float(delta)}


def _timings(before, after):
    if before is None or after is None:
        return {"status": "not_verified", "reason": "paired_timing_required"}
    if any(x["repetition_id"] != y["repetition_id"] or x["position"] == y["position"]
           for x, y in zip(before, after)) or any(x["position"] == y["position"]
                                                  for x, y in zip(before, before[1:])):
        return {"status": "not_verified", "reason": "matching_alternating_ab_ba_trials_required"}
    warm = [_p95([v for trial in arm for v in trial["stable_ms"]]) for arm in (before, after)]
    first = [_p95([v for trial in arm for v in trial["first_caption_ms"]])
             if all(trial["first_caption_ms"] is not None for trial in arm) else None
             for arm in (before, after)]
    threshold = _d(warm[0]) * (1 + _LATENCY_REGRESSION)
    n = sum(len(x["stable_ms"]) for x in before)
    result = {"status": "fail" if _d(warm[1]) > threshold else "reported_within_bounds",
              "stable_translation_p95_ms": {"incumbent": warm[0], "candidate": warm[1]},
              "first_caption_p95_ms": {"incumbent": first[0], "candidate": first[1]},
              "paired_trials": len(before), "timed_clips_per_arm": n}
    if n < _MIN_TIMED_CLIPS_PER_ARM:
        result.update(status="not_verified", reason="too_few_paired_timing_observations")
        return result

    trials = [[_d(_p95(trial["stable_ms"])) for trial in arm] for arm in (before, after)]
    if min(*trials[0], *trials[1]) <= 0:
        result.update(status="not_verified", reason="zero_p95_timing_resolution")
        return result
    spread = [(max(per_arm) - min(per_arm)) / median(per_arm) for per_arm in trials]
    paired_ratios = [(b - a) / a for a, b in zip(*trials)]
    issues = []
    if max(spread) > _MAX_REPEAT_RANGE:
        issues.append("per_arm_timing_repeats_too_noisy")
    if max(paired_ratios) - min(paired_ratios) > _MAX_PAIRED_RATIO_RANGE:
        issues.append("paired_timing_ratios_too_noisy")
    passed_pooled = _d(warm[1]) <= threshold
    if any((ratio <= _LATENCY_REGRESSION) != passed_pooled for ratio in paired_ratios):
        issues.append("paired_repeats_disagree_at_latency_threshold")
    result.update(repeat_range_fractions={"incumbent": float(spread[0]), "candidate": float(spread[1])},
                  paired_regression_fractions=[float(x) for x in paired_ratios])
    if issues:
        result.update(status="not_verified", reasons=issues)
    return result


def _resources(manifest, before, after):
    cap = manifest["budget"]
    if before is None or after is None:
        available = before if before is not None else after
        if available is not None and any(_d(available[name]) > cap[name] for name in _RESOURCE_FIELDS):
            return {"status": "fail", "reasons": ["available_arm_alone_exceeded_combined_budget"]}
        return {"status": "not_verified", "reason": "resource_receipts_missing"}
    cpu = sum(_d(r["cpu_seconds"]) for r in (before, after))
    wall = sum(_d(r["wall_seconds"]) for r in (before, after))
    rss = [r["peak_rss_bytes"] for r in (before, after)]
    reasons = []
    if cpu > cap["cpu_seconds"] or wall > cap["wall_seconds"] or max(rss) > cap["peak_rss_bytes"]:
        reasons.append("combined_run_budget_exceeded")
    if _d(rss[1]) > _d(rss[0]) * (1 + _LATENCY_REGRESSION):
        reasons.append("peak_rss_regressed")
    return {"status": "fail" if reasons else "reported_within_bounds", "reasons": reasons,
            "combined_cpu_seconds": float(cpu), "combined_wall_seconds": float(wall),
            "peak_rss_bytes": {"incumbent": rss[0], "candidate": rss[1]}}


def compare(manifest, *, manifest_sha256, incumbent, candidate):
    validate_manifest(manifest)
    _require(is_sha256(manifest_sha256), "manifest_pin_required")
    _validate_receipt(incumbent, manifest, manifest_sha256, "incumbent")
    _validate_receipt(candidate, manifest, manifest_sha256, "candidate")
    resources = _resources(manifest, incumbent["resources"], candidate["resources"])
    results = {}
    statuses = [resources["status"]]
    for name, paired in sorted(manifest["directions"].items()):
        arm = [receipt["directions"].get(name) for receipt in (incumbent, candidate)]
        results[name] = {"clips": paired["clips"]}
        for key, fn in (("asr", _asr), ("translation", _translation), ("timing", _timings)):
            value = fn(*(row[key] if row else None for row in arm))
            results[name][key] = value
            statuses.append(value["status"])
    status = ("fail" if "fail" in statuses else "not_verified" if "not_verified" in statuses
              else "reported_within_bounds")
    return {"schema_version": 1, "kind": "bridgecaption_speech_benchmark_comparison",
            "manifest_sha256": manifest_sha256, "scope": "development", "status": status,
            "resources": resources, "directions": results, "budget": dict(manifest["budget"]),
            "human_adequacy": "not_evaluated", "model_selected": False, "promotion_eligible": False}


def _read(path, digest):
    return read_pinned_json(path.parent, path.name, expected_sha256=digest)


class _Parser(argparse.ArgumentParser):
    def error(self, _):
        raise BenchmarkError("invalid_benchmark_arguments")


def main(argv=None):
    p = _Parser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True, parser_class=_Parser)
    for name in ("preflight", "compare"):
        child = sub.add_parser(name)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--manifest-sha256", required=True)
        if name == "compare":
            for arm in ("incumbent", "candidate"):
                child.add_argument(f"--{arm}", type=Path, required=True)
                child.add_argument(f"--{arm}-sha256", required=True)
    try:
        args = p.parse_args(argv)
        manifest = validate_manifest(_read(args.manifest, args.manifest_sha256))
        if args.operation == "preflight":
            output = {"status": "protocol_only_not_measured", "scope": "development",
                      "manifest_sha256": args.manifest_sha256, "budget": manifest["budget"],
                      "directions": {d: x["clips"] for d, x in sorted(manifest["directions"].items())},
                      "model_selected": False, "promotion_eligible": False}
            code = 0
        else:
            output = compare(manifest, manifest_sha256=args.manifest_sha256,
                             incumbent=_read(args.incumbent, args.incumbent_sha256),
                             candidate=_read(args.candidate, args.candidate_sha256))
            output["receipt_sha256"] = {"incumbent": args.incumbent_sha256,
                                         "candidate": args.candidate_sha256}
            code = 0 if output["status"] == "reported_within_bounds" else 3
        print(json.dumps(output, sort_keys=True, allow_nan=False))
        return code
    except BenchmarkError as exc:
        code = str(exc)  # Fixed strings defined by this module only.
    except (ContractError, OSError, ValueError, TypeError, KeyError, RecursionError):
        code = "invalid_or_unverified_local_benchmark_evidence"
    print(json.dumps({"status": "rejected", "code": code, "promotion_eligible": False}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
