"""Offline adapter for paired human development judgments, never final holdout data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

CODES = frozenset(("negation", "quantity", "named_entity", "omission", "context_leakage"))
_ISSUE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}\Z")
_KIND = "paired_blinded_development_review"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REVIEW_KEYS = {"version", "kind", "fixture_sha256", "baseline_outputs_sha256",
                "candidate_outputs_sha256", "judgments"}
_MAX_RECEIPT_BYTES = 8 * 1024 * 1024


class ReviewError(ValueError):
    """Untrusted, mismatched or incomplete development review metadata."""


def is_issue_token(value: object) -> bool:
    """Validate a category plus a stable, non-text issue ID shared by both arms."""
    if not isinstance(value, str):
        return False
    category, separator, issue_id = value.partition(":")
    return bool(separator and category in CODES and _ISSUE_ID.fullmatch(issue_id))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReviewError("Duplicate field in development review")
        result[key] = value
    return result


def load_development_review(
    path: Path,
    *,
    expected_sha256: str,
    fixture_sha256: str,
    baseline_outputs_sha256: str,
    candidate_outputs_sha256: str,
    case_directions: Mapping[str, str],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Bind separately trusted DEV labels to exact corresponding texts.

    The expected digest must come independently from the bilingual review
    custodian, who assigns and unblinds the pairs. Hashing a file yourself
    cannot prove how it was reviewed. This is not an independent final-gate
    result, a license review, or user approval to activate a model. No raw
    prompts, translations, reviewer names, or final-test labels are accepted.
    """
    if not all(isinstance(v, str) and _SHA.fullmatch(v) for v in (
        expected_sha256, fixture_sha256, baseline_outputs_sha256,
        candidate_outputs_sha256,
    )):
        raise ReviewError("Development review requires independently known SHA-256 pins")
    try:
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ReviewError("Development review must be a regular local file")
        with path.open("rb") as stream:
            raw = stream.read(_MAX_RECEIPT_BYTES + 1)
    except OSError:
        raise ReviewError("Could not read development review") from None
    if len(raw) > _MAX_RECEIPT_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ReviewError("Development review checksum or size mismatch")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
    except (ValueError, RecursionError):
        raise ReviewError("Malformed development review") from None
    if (not isinstance(data, dict) or data.keys() != _REVIEW_KEYS
            or type(data["version"]) is not int or data["version"] != 1
            or data["kind"] != _KIND):
        raise ReviewError("Only the paired, blinded development-review schema is supported")
    expected_pins = {"fixture_sha256": fixture_sha256,
                     "baseline_outputs_sha256": baseline_outputs_sha256,
                     "candidate_outputs_sha256": candidate_outputs_sha256}
    if any(data[key] != pin for key, pin in expected_pins.items()):
        raise ReviewError("Development review pins do not match the selected suite and outputs")
    judgments = data["judgments"]
    if (not case_directions or not isinstance(judgments, dict)
            or judgments.keys() != case_directions.keys()):
        raise ReviewError("Development review must cover every selected case exactly once")
    grouped: dict[str, dict[str, dict[str, list[str]]]] = {}
    for key, pair in judgments.items():
        if not isinstance(pair, dict) or pair.keys() != {"baseline", "candidate"}:
            raise ReviewError("Paired development judgments are required for each case")
        for codes in pair.values():
            if (not isinstance(codes, list) or not all(is_issue_token(c) for c in codes)
                    or len(codes) != len(set(codes))):
                raise ReviewError("Unknown or duplicate critical-meaning code in development review")
        grouped.setdefault(case_directions[key], {})[key] = pair
    return grouped
