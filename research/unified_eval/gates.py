"""Paired, development-only, offline regression decision aid.

"pass" means all technical checks in one lane/direction met their thresholds.
It never denotes human translation accuracy or authorization to promote models.
No data files, weights, references, private review corpora, or networks are read.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from decimal import Decimal
import hashlib
import json
import math
import re

from .review import is_issue_token

MAX_DEV_CHRF_DROP = Decimal("0.5")
DIRECTIONS = re.compile(r"([a-z]{2,3})->([a-z]{2,3})\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")


def fingerprint(data: Mapping) -> str:
    """Hash canonical, UTF-8 JSON without normalizing text."""
    return hashlib.sha256(json.dumps(dict(data), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _pins(value):
    """Require immutable identity plus base, tokenizer, active and rollback pins."""
    if not isinstance(value, Mapping) or set(value) != {
            "base_revision", "files", "adapter_sha256", "rollback_sha256"}:
        return False
    revision, files = value["base_revision"], value["files"]
    if not isinstance(revision, str) or REVISION.fullmatch(revision) is None:
        return False
    current, rollback = value["adapter_sha256"], value["rollback_sha256"]
    if (current is None) != (rollback is None) or (
            current is not None and (not _sha(current) or not _sha(rollback) or current == rollback)):
        return False
    if not isinstance(files, Mapping) or not all(
            isinstance(name, str) and name and re.fullmatch(r"[a-zA-Z0-9_./-]+", name)
            and all(part not in {"", ".", ".."} for part in name.split("/")) and _sha(digest)
            for name, digest in files.items()):
        return False
    return ("config.json" in files
            and ({"source.spm", "target.spm", "vocab.json"} <= files.keys()
                 or "tokenizer.json" in files)
            and any(re.search(r"(?:\.safetensors|\.bin)\Z", name) for name in files))


def _outputs(value):
    if not isinstance(value, Mapping) or not value:
        return False
    for key, text in value.items():
        if not isinstance(key, str) or not key or not isinstance(text, str) or not text.strip():
            return False
        try:
            key.encode("utf-8"); text.encode("utf-8")
        except UnicodeError:
            return False
    return True


def _chrf(value):
    if not isinstance(value, Mapping):
        return None
    value = value.get("chrf")
    if (type(value) not in (int, float) or not 0 <= value <= 100
            or not math.isfinite(value)):
        return None
    return Decimal(str(value))


def _scores(score, *, direction, ids, baseline_hash, candidate_hash):
    """Check matched score/row/group coverage; automatic diagnostics only."""
    if not isinstance(score, Mapping) or not isinstance(score.get("hashes"), Mapping):
        return None
    hashes = score["hashes"]
    if (score.get("direction") != direction or type(score.get("n")) is not int
            or score["n"] != len(ids)
            or hashes.get("baseline_outputs") != baseline_hash
            or hashes.get("candidate_outputs") != candidate_hash
            or not _sha(hashes.get("case_spec"))):
        return None
    original, proposed = _chrf(score.get("baseline")), _chrf(score.get("candidate"))
    if original is None or proposed is None:
        return None
    delta = proposed - original
    groups, rows = score.get("subgroups"), score.get("rows")
    if not isinstance(groups, Mapping) or not groups or not isinstance(rows, list):
        return delta, None
    seen, coverage = set(), Counter()
    for row in rows:
        if not isinstance(row, Mapping):
            return delta, None
        key, tags = row.get("id"), row.get("groups")
        if (not isinstance(key, str) or key not in ids or key in seen
                or not isinstance(tags, (list, tuple))
                or any(not isinstance(group, str) or not group or group not in groups
                       for group in tags)
                or len(tags) != len(set(tags))):
            return delta, None
        seen.add(key)
        coverage.update(tags)
    if seen != ids or coverage.keys() != groups.keys():
        return delta, None
    scored_groups = []
    for name, group in groups.items():
        if (not isinstance(group, Mapping) or type(group.get("n")) is not int
                or coverage[name] != group["n"]):
            return delta, None
        before, after = _chrf(group.get("baseline")), _chrf(group.get("candidate"))
        if before is None or after is None:
            return delta, None
        scored_groups.append(after - before)
    return delta, scored_groups


def _critical_delta(critical, ids):
    """Only caller-verified independent, blinded DEV labels enter this function."""
    if not isinstance(critical, Mapping) or critical.keys() != ids:
        return None
    newly_confirmed = 0
    for pair in critical.values():
        if not isinstance(pair, Mapping) or set(pair) != {"baseline", "candidate"}:
            return None
        a, b = pair["baseline"], pair["candidate"]
        for labels in (a,b):
            if (not isinstance(labels, (list, tuple))
                    or any(not is_issue_token(code) for code in labels)
                    or len(labels) != len(set(labels))):
                return None
        newly_confirmed += len(set(b) - set(a))
    return newly_confirmed


def evaluate_direction(baseline, candidate, *, lane, baseline_pins=None, candidate_pins=None,
                       expected_fixture_sha256=None, expected_baseline_outputs_sha256=None,
                       expected_input_mix_sha256=None, expected_warm_requests=None, score=None, critical=None) -> dict:
    """Evaluate an entire paired DEV direction independently.

    Receipts contain `direction`, `scope='development'`, `fixture_sha256`, verified
    `checkpoint` identity, `outputs={id: text}`, and comparable `performance`.
    External expected fixture, baseline output/checkpoint, and candidate checkpoint
    pins and the measured timing input mix must be verified independently of the
    candidate receipts. The scorer
    and verified human-review adapter are separate trust boundaries; never pass
    user-asserted or synthetic critical labels as independent human evidence.
    """
    checks, summaries = {}, {}
    failures, blockers = [], []

    def check(name, state, reason=None):
        checks[name] = state
        if state == "fail":
            failures.append(reason or name)
        if state == "blocked":
            blockers.append(reason or name)

    def result(direction=None):
        return {
            "direction": direction, "lane": lane if lane in ("infrastructure", "model_dev") else None,
            "status": "fail" if failures else "blocked" if blockers else "pass",
            "checks": checks, "failures": failures, "blockers": blockers, "summary": summaries,
            "independent_human_evaluation": "not_evaluated",
            "promotion_eligible": False, "external_user_approval_required": True,
        }

    if lane not in ("infrastructure", "model_dev"):
        check("lane", "blocked", "unknown_gate_lane")
        return result()
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        check("receipts", "blocked", "paired_dev_receipts_required")
        return result()
    direction = baseline.get("direction")
    match = DIRECTIONS.fullmatch(direction) if isinstance(direction, str) else None
    if match is None or match[1] == match[2] or candidate.get("direction") != direction:
        check("direction_pairing", "blocked", "same_explicit_direction_required")
        return result()
    if baseline.get("scope") != "development" or candidate.get("scope") != "development":
        check("scope", "blocked", "non_dev_or_unverified_scope")
        return result(direction)
    if (not _sha(expected_fixture_sha256)
            or baseline.get("fixture_sha256") != expected_fixture_sha256
            or candidate.get("fixture_sha256") != expected_fixture_sha256):
        check("fixture", "blocked", "independently_pinned_matching_dev_fixture_required")
        return result(direction)
    before, after = baseline.get("outputs"), candidate.get("outputs")
    if not _outputs(before) or not _outputs(after) or before.keys() != after.keys():
        check("paired_rows", "blocked", "identical_explicit_nonempty_case_coverage_required")
        return result(direction)
    baseline_hash = fingerprint(before)
    if not _sha(expected_baseline_outputs_sha256) or baseline_hash != expected_baseline_outputs_sha256:
        check("baseline_outputs", "blocked", "independently_pinned_exact_baseline_outputs_required")
        return result(direction)
    candidate_hash = fingerprint(after)
    check("paired_rows", "pass")
    summaries["paired_rows"] = len(before)
    if not (_pins(baseline_pins) and _pins(candidate_pins)):
        check("checkpoint_pins", "blocked", "independently_verified_complete_checkpoint_pins_required")
    elif baseline.get("checkpoint") != baseline_pins or candidate.get("checkpoint") != candidate_pins:
        check("checkpoint_pins", "fail", "receipt_checkpoint_disagrees_with_trusted_pin")
    else:
        check("checkpoint_pins", "pass")

    if lane == "infrastructure":
        if checks["checkpoint_pins"] != "pass":
            check("checkpoint_parity", "blocked", "unverified_checkpoints_cannot_show_parity")
        else:
            exact = baseline_pins == candidate_pins
            check("checkpoint_parity", "pass" if exact else "fail", "checkpoint_identity_changed")
        changed = sum(before[key] != after[key] for key in before)
        summaries["changed_output_rows"] = changed
        check("output_parity", "pass" if changed == 0 else "fail", "exact_output_bytes_changed")
        # A text similarity allowance cannot override an infrastructure mismatch.
    else:
        paired_scores = _scores(score, direction=direction, ids=before.keys(),
                                baseline_hash=baseline_hash, candidate_hash=candidate_hash)
        if paired_scores is None:
            check("direction_chrf", "blocked", "paired_dev_chrF_missing_or_unverified")
            group_deltas = None
        else:
            delta, group_deltas = paired_scores
            summaries["chrf_delta"] = float(delta)
            check("direction_chrf", "fail" if delta < -MAX_DEV_CHRF_DROP else "pass",
                  "dev_chrF_drop_exceeds_half_point")
        if group_deltas is None:
            check("subgroup_chrf", "blocked", "complete_paired_subgroups_required")
        else:
            summaries["scored_subgroups"] = len(group_deltas)
            summaries["regressing_subgroups"] = sum(d < 0 for d in group_deltas)
            check("subgroup_chrf", "fail" if any(d < 0 for d in group_deltas) else "pass",
                  "development_subgroup_regressed")
        newly_confirmed = _critical_delta(critical, before.keys())
        if newly_confirmed is None:
            check("critical_errors", "blocked", "paired_verified_human_dev_labels_required")
        else:
            summaries["new_confirmed_critical_errors"] = newly_confirmed
            check("critical_errors", "fail" if newly_confirmed else "pass",
                  "new_confirmed_critical_meaning_error")

    if checks["checkpoint_pins"] != "pass":
        check("performance", "blocked", "performance_not_bound_to_verified_checkpoint_pins")
    elif not _sha(expected_input_mix_sha256):
        check("performance", "blocked", "independently_pinned_performance_input_mix_required")
    elif type(expected_warm_requests) is not int or expected_warm_requests < 40:
        check("performance", "blocked", "independently_pinned_warm_request_count_required")
    elif not all(isinstance(row.get("performance"), Mapping) for row in (baseline, candidate)):
        check("performance", "blocked", "comparable_performance_receipts_required")
    elif any(row["performance"].get("input_mix_sha256") != expected_input_mix_sha256
             for row in (baseline, candidate)):
        check("performance", "blocked", "performance_input_mix_does_not_match_frozen_measurement")
    else:
        from .performance import compare_performance
        perf = compare_performance(
            baseline["performance"], candidate["performance"], baseline_pins=baseline_pins,
            candidate_pins=candidate_pins, change_kind="model" if lane == "model_dev" else lane,
            expected_direction=direction, expected_fixture_sha256=expected_fixture_sha256,
            expected_input_mix_sha256=expected_input_mix_sha256,
            expected_warm_requests=expected_warm_requests,
        )
        state = perf.get("status")
        summaries["performance"] = perf
        check("performance", state if state in {"pass","fail"} else "blocked",
              "latency_or_rss_regression" if state == "fail" else "performance_evidence_incomparable")
    return result(direction)
