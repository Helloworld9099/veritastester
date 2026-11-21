"""Offline artifact and public-report contracts for the manual research pipeline.

Digests are supplied by an independent caller; a file cannot attest to itself.
Local readers are for small JSON metadata, not training rows or model binaries.
No networking, uploading, logging, backend import, or promotion occurs here.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType


class ContractError(ValueError):
    """A receipt or its local binding cannot be verified."""


_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,119}\Z")
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\..*)?\Z", re.I)
_DIRECTION = re.compile(r"([a-z]{2,3})->([a-z]{2,3})\Z")
MAX_JSON_BYTES = 1 << 20
MAX_CHECKPOINT_BYTES = 16 << 30


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require(value, reason: str) -> None:
    if not value:
        # Never include raw JSON values, source text, filesystem paths or IDs.
        raise ContractError(reason)


def _parts(relative_path: str, *, suffixes=(".json",)) -> tuple[str, ...]:
    _require(isinstance(relative_path, str) and 0 < len(relative_path) <= 512,
             "invalid_receipt_path")
    parts = tuple(relative_path.split("/"))
    _require(len(parts) <= 24 and all(_COMPONENT.fullmatch(x) and not x.endswith(".")
             and _RESERVED.fullmatch(x) is None for x in parts)
             and parts[-1].endswith(suffixes), "invalid_receipt_path")
    return parts


def _regular_fd(fd: int, *, size_limit: int) -> bytes:
    with os.fdopen(fd, "rb", closefd=False) as file:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), "receipt_must_be_regular_file")
        _require(0 < before.st_size <= size_limit, "invalid_receipt_size")
        raw = file.read(size_limit + 1)
        after = os.fstat(fd)
        _require(0 < len(raw) <= size_limit and len(raw) == before.st_size == after.st_size
                 and before.st_mtime_ns == after.st_mtime_ns, "receipt_changed_during_read")
    return raw


def _is_reparse(info) -> bool:
    return bool(getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


@contextmanager
def _opened_relative(root: Path, parts: tuple[str,...]):
    # Root is an operator-selected trust boundary. Do not follow a symlink in
    # the provided relative path. Use directory file descriptors on POSIX to
    # keep checked directory handles bound when an entry is renamed.
    dflags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fflags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os,"O_NONBLOCK",0)
              | getattr(os, "O_BINARY", 0))
    with ExitStack() as closes:
        try:
            root_info = root.lstat()
            _require(stat.S_ISDIR(root_info.st_mode) and not _is_reparse(root_info),
                     "invalid_receipt_root")
            root = root.resolve(strict=True)
            _require(os.path.samestat(root_info, root.stat()), "receipt_root_changed")
            if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW") and hasattr(os,"O_DIRECTORY"):
                directory = os.open(root, dflags)
                closes.callback(os.close,directory)
                _require(os.path.samestat(root_info, os.fstat(directory)), "receipt_root_changed")
                for segment in parts[:-1]:
                    directory = os.open(segment, dflags, dir_fd=directory)
                    closes.callback(os.close,directory)
                # Avoid opening devices or FIFOs even though fd check also rejects them.
                info = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
                _require(stat.S_ISREG(info.st_mode), "receipt_must_be_regular_file")
                fd = os.open(parts[-1], fflags, dir_fd=directory)
                closes.callback(os.close,fd)
            else:
                # Conservative portable fallback. Pins still bind exact bytes;
                # verify every component and compare the opened file's identity.
                path = root
                for segment in parts[:-1]:
                    path = path / segment
                    item = path.lstat()
                    _require(stat.S_ISDIR(item.st_mode) and not _is_reparse(item),
                             "invalid_receipt_directory")
                path = path / parts[-1]
                info = path.lstat()
                _require(stat.S_ISREG(info.st_mode) and not _is_reparse(info)
                         and path.resolve(strict=True).is_relative_to(root),
                         "receipt_must_be_regular_file")
                fd = os.open(path, fflags)
                closes.callback(os.close,fd)
            _require(os.path.samestat(info,os.fstat(fd)), "receipt_changed_during_open")
            yield fd
        except (OSError, RuntimeError, TypeError):
            raise ContractError("receipt_unavailable_or_unsafe_path") from None


def _read_relative(root: Path, parts: tuple[str,...], *, size_limit: int) -> bytes:
    with _opened_relative(root, parts) as fd:
        return _regular_fd(fd, size_limit=size_limit)


def _unique(pairs):
    out = {}
    for key, val in pairs:
        _require(key not in out, "duplicate_receipt_json_key")
        out[key] = val
    return out


def _bad_constant(_):
    raise ContractError("nonfinite_receipt_number")


def _finite_json_float(token):
    number = float(token)
    _require(math.isfinite(number), "nonfinite_receipt_number")
    return number


def read_pinned_json(root: str | Path, relative_path: str, *, expected_sha256: str) -> dict:
    """Read guarded JSON metadata and check raw bytes *before* decoding.

    Result is for local verification only. Callers must not log or publish it
    directly; use the narrow public receipt contract below for rendered output.
    """
    _require(is_sha256(expected_sha256), "independent_receipt_sha256_required")
    raw = _read_relative(Path(root), _parts(relative_path),size_limit=MAX_JSON_BYTES)
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "receipt_raw_sha256_mismatch")
    try:
        metadata = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                              parse_constant=_bad_constant, parse_float=_finite_json_float)
    except ContractError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        raise ContractError("invalid_receipt_json") from None
    _require(isinstance(metadata, dict), "receipt_json_object_required")
    return metadata


KINDS = frozenset({"bridgecaption_aggregate_receipt"})
STAGES = frozenset({"data", "train", "eval", "report"})
STATUSES = frozenset({"planned", "inspection_only", "completed", "pass", "fail",
                      "blocked", "not_verified"})
COUNTS = frozenset({"train_rows", "dev_rows", "paired_rows", "reviewed_rows",
                    "confirmed_critical_errors", "optimizer_updates"})
METRICS = {"chrf":100, "bleu":100, "warm_p95_ms":86_400_000, "peak_rss_mib":1_048_576}
RECEIPT_FIELDS = frozenset({"schema_version", "kind", "stage", "scope", "direction",
                            "fixture_sha256", "method_sha256", "checkpoint_sha256",
                            "output_sha256", "counts", "metrics", "status"})
_CHECKPOINT_FIELDS = frozenset({"direction", "architecture", "base_revision", "tokenizer_sha256",
                                "checkpoint_path", "checkpoint_sha256"})


def _direction(value):
    parsed = _DIRECTION.fullmatch(value) if isinstance(value,str) else None
    return bool(parsed and parsed[1] != parsed[2])


def _allowed(value, values) -> bool:
    return isinstance(value,str) and value in values


def validate_receipt(value: Mapping) -> dict:
    """Validate and detach a strictly aggregate-only public receipt.

    Construct from checked local evidence; this function itself cannot prove
    where the aggregate originated or attest to translation quality.
    """
    _require(isinstance(value,Mapping) and set(value) == RECEIPT_FIELDS,
             "unexpected_or_missing_public_receipt_fields")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
             and _allowed(value["kind"], KINDS), "invalid_public_receipt_schema")
    _require(_allowed(value["stage"],STAGES) and _allowed(value["status"],STATUSES)
             and value["scope"] == "development" and _direction(value["direction"]),
             "invalid_public_receipt_scope_or_stage")
    for key in ("fixture_sha256", "method_sha256"):
        _require(is_sha256(value[key]), "invalid_public_receipt_digest")
    for key in ("checkpoint_sha256", "output_sha256"):
        _require(value[key] is None or is_sha256(value[key]), "invalid_optional_receipt_digest")
    counts, metrics = value["counts"], value["metrics"]
    _require(isinstance(counts,Mapping) and set(counts) <= COUNTS
             and all(type(x) is int and 0 <= x <= (1 << 53)-1 for x in counts.values()),
             "invalid_public_receipt_counts")
    _require(isinstance(metrics,Mapping) and set(metrics) <= METRICS.keys()
             and all(type(x) in (int,float) and 0 <= x <= METRICS[key] and math.isfinite(x)
                     for key,x in metrics.items()), "invalid_public_receipt_metrics")
    _require("reviewed_rows" not in counts or
             ("paired_rows" in counts and counts["reviewed_rows"] <= counts["paired_rows"]),
             "invalid_public_review_counts")
    if value["stage"] == "eval" and value["status"] in {"completed","pass","fail"}:
        _require(counts.get("paired_rows",0) > 0, "completed_evaluation_requires_paired_rows")
    # Construct from the approved field set rather than forward arbitrary
    # metadata, even when consumers only render a subset of this receipt.
    output = {key:value[key] for key in sorted(RECEIPT_FIELDS)}
    output["counts"], output["metrics"] = dict(counts), dict(metrics)
    return output


_VERIFICATION = object()


@dataclass(frozen=True, init=False)
class VerifiedReceipt:
    """Typed immutable result of a local raw-byte pin and schema check."""

    raw_sha256: str
    _public: Mapping

    def __init__(self, raw_sha256: str, public: Mapping, *, _verification):
        _require(_verification is _VERIFICATION and is_sha256(raw_sha256),
                 "receipt_must_be_verified_from_pinned_bytes")
        output = validate_receipt(public)
        for key in ("counts","metrics"):
            output[key] = MappingProxyType(output[key])
        object.__setattr__(self,"raw_sha256",raw_sha256)
        object.__setattr__(self,"_public",MappingProxyType(output))

    def to_public(self) -> dict:
        """Return safe detached metadata, without any local filesystem locator."""
        return validate_receipt(self._public)


def verify_receipt(root: str | Path, relative_path: str, *, expected_sha256: str) -> VerifiedReceipt:
    """Require an independent pin for the exact JSON bytes, not a self-hash."""
    report = read_pinned_json(root, relative_path, expected_sha256=expected_sha256)
    return VerifiedReceipt(expected_sha256, report, _verification=_VERIFICATION)


def verify_checkpoint(root: str | Path, spec: Mapping) -> dict:
    """Verify only local checkpoint byte identity. Never deserialize the blob.

    Direction, architecture and base/tokenizer pins are declared bindings. This
    does not check that a checkpoint implements them, certify rights, or authorize
    using or exporting the artifact. Use the training compatibility checks for
    that separate decision.
    """
    _require(isinstance(spec,Mapping) and set(spec) == _CHECKPOINT_FIELDS,
             "unexpected_or_missing_checkpoint_pin_fields")
    _require(_direction(spec["direction"]) and _allowed(
        spec["architecture"], {"marian","causal_lm","seq2seq_lm"}),
        "invalid_declared_checkpoint_binding")
    _require(isinstance(spec["base_revision"],str)
             and re.fullmatch(r"[0-9a-f]{40}",spec["base_revision"])
             and is_sha256(spec["tokenizer_sha256"]) and is_sha256(spec["checkpoint_sha256"]),
             "independent_checkpoint_pins_required")
    parts = _parts(spec["checkpoint_path"], suffixes=(".safetensors",".bin",".pt",".pth"))
    count = 0
    digest = hashlib.sha256()
    with _opened_relative(Path(root), parts) as fd, os.fdopen(fd,"rb",closefd=False) as stream:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_CHECKPOINT_BYTES,
                 "invalid_checkpoint_size")
        for block in iter(lambda:stream.read(1024*1024), b""):
            count += len(block)
            _require(count <= before.st_size, "checkpoint_grew_during_read")
            digest.update(block)
        after = os.fstat(fd)
        _require(count == before.st_size == after.st_size
                 and before.st_mtime_ns == after.st_mtime_ns, "checkpoint_changed_during_read")
    _require(digest.hexdigest() == spec["checkpoint_sha256"], "checkpoint_raw_sha256_mismatch")
    return {
        "schema_version":1,"kind":"bridgecaption_checkpoint_metadata",
        "scope":"development","status":"verified_pinned_bytes",
        "direction":spec["direction"],"checkpoint_sha256":digest.hexdigest(),"size_bytes":count,
        "declared_binding":{"architecture":spec["architecture"],
                            "base_revision":spec["base_revision"],
                            "tokenizer_sha256":spec["tokenizer_sha256"]},
        "load_compatibility":"not_verified","promotion_eligible":False,
    }


def compare_receipts(baseline: VerifiedReceipt, candidate: VerifiedReceipt, *,
                     lane: str = "infrastructure") -> dict:
    """Compare independently byte-pinned summaries for one explicit DEV direction.

    Infra compares *reported* checkpoint/output identities exactly. Model-dev
    chrF is a limited diagnostic with a 0.5 point allowance; these receipts
    cannot prove paired critical/subgroup or comparable performance gates.
    """
    _require(type(baseline) is VerifiedReceipt and type(candidate) is VerifiedReceipt,
             "comparison_requires_two_verified_receipts")
    _require(_allowed(lane, {"infrastructure","model_dev"}), "invalid_comparison_lane")
    old,new = baseline.to_public(),candidate.to_public()
    result = {
        "schema_version":1, "kind":"bridgecaption_aggregate_comparison", "scope":"development",
        "lane":lane, "status":"not_verified",
        "baseline_receipt_sha256":baseline.raw_sha256, "candidate_receipt_sha256":candidate.raw_sha256,
        "direction":None, "checks":{}, "diagnostics":{},
        "baseline_evaluation_status":old["status"], "candidate_evaluation_status":new["status"],
        "reason_codes":[], "independent_human_evaluation":"not_evaluated",
        "performance_gate":"not_evaluated", "promotion_eligible":False,
        "external_user_approval_required":True,
    }
    reasons = result["reason_codes"]
    if old["stage"] != "eval" or new["stage"] != "eval":
        reasons.append("comparison_requires_evaluation_stage")
    if old["direction"] != new["direction"]:
        reasons.append("evaluation_direction_mismatch")
    else:
        result["direction"] = old["direction"]
    if old["fixture_sha256"] != new["fixture_sha256"]:
        reasons.append("evaluation_fixture_mismatch")
    if old["method_sha256"] != new["method_sha256"]:
        reasons.append("evaluation_method_mismatch")
    old_n,new_n = old["counts"].get("paired_rows",0),new["counts"].get("paired_rows",0)
    if old_n == 0 or old_n != new_n:
        reasons.append("evaluation_paired_counts_not_comparable")
    if any(value["status"] not in {"completed","pass","fail"} for value in (old,new)):
        reasons.append("evaluation_not_completed")
    if reasons:
        result["status"] = "not_comparable"
        return result
    result["diagnostics"]["paired_rows"] = old_n
    if lane == "infrastructure":
        for field,status in (("checkpoint_sha256","reported_checkpoint_parity"),
                             ("output_sha256","reported_output_parity")):
            a,b = old[field],new[field]
            result["checks"][status] = "not_verified" if a is None or b is None else (
                "equal" if a == b else "different")
        checks = result["checks"].values()
        if "different" in checks:
            reasons.append("reported_infrastructure_parity_mismatch")
            result["status"] = "fail"
        elif "fail" in (old["status"],new["status"]):
            reasons.append("source_evaluation_reported_failure")
            result["status"] = "fail"
        elif "not_verified" in checks:
            reasons.append("reported_infrastructure_parity_missing")
        else:
            result["status"] = "reported_exact_match"
    elif "chrf" in old["metrics"] and "chrf" in new["metrics"]:
        delta = Decimal(str(new["metrics"]["chrf"])) - Decimal(str(old["metrics"]["chrf"]))
        result["diagnostics"]["chrf_delta"] = float(delta)
        result["checks"]["chrf_dev_triage"] = (
            "within_half_point" if delta >= Decimal("-0.5") else "regressed")
        reasons.append("full_paired_dev_and_independent_human_evidence_not_verified")
    else:
        reasons.append("paired_chrf_diagnostics_missing")
    return result
