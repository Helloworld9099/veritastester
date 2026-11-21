"""Pin one whole multilingual checkpoint; no direction-to-model routing."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

MANIFEST = Path(__file__).with_name("registry.json")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_REVISION = re.compile(r"[a-f0-9]{40}\Z")
_CODE = re.compile(r"([a-z]{2,3})(?:-[a-z]{4})?(?:-(?:[a-z]{2}|[0-9]{3}))?\Z")
_ROUTE = re.compile(r"([a-z]{2,3})->([a-z]{2,3})\Z")
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}\Z")
_FILE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}\Z")
_MAX_METADATA_BYTES = 1 << 20
_TOKENIZERS = {"tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model"}
_METADATA = _TOKENIZERS | {
    "config.json", "generation_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
}


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _fingerprint(value):
    return _digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), allow_nan=False).encode())


def _unique(pairs):
    out = {}
    for key, val in pairs:
        _check(key not in out, "Duplicate manifest field")
        out[key] = val
    return out


def _invalid_constant(_):
    raise ValueError("Nonfinite manifest value")


def _json(path):
    path = Path(path)
    _check(path.is_file() and not path.is_symlink(), "Expected a regular local metadata file")
    with path.open("rb") as stream:
        raw = stream.read(_MAX_METADATA_BYTES + 1)
    _check(0 < len(raw) <= _MAX_METADATA_BYTES, "Invalid metadata size")
    try:
        return raw, json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                               parse_constant=_invalid_constant)
    except (UnicodeError, ValueError, RecursionError):
        raise ValueError("Invalid registry or direction evidence metadata") from None


def normalize_language(value):
    parsed = _CODE.fullmatch(value.strip().lower()) if isinstance(value, str) else None
    _check(parsed is not None and parsed[1] != "auto", "An explicit language code is required")
    return parsed[1]


def _directions(rows):
    _check(isinstance(rows, list) and bool(rows), "Explicit target directions required")
    result = set()
    for value in rows:
        parsed = _ROUTE.fullmatch(value) if isinstance(value, str) else None
        _check(parsed is not None and parsed[1] != parsed[2] and value not in result,
               "Invalid or duplicate direction")
        result.add(value)
    return frozenset(result)


def _file_sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


@dataclass(frozen=True)
class SingleModelSpec:
    """Read-only checkpoint plus the complete advertised direction set."""

    model_id: str
    revision: str
    architecture: str
    checkpoint_sha256: str
    files: Mapping = field(repr=False)
    verified_directions: frozenset[str]


class ModelRegistry:
    """A registry for exactly one candidate, or no operational model at all."""

    def __init__(self, path: Path = MANIFEST, *, expected_sha256: str | None = None,
                 evaluation_receipt: Path | None = None):
        raw, value = _json(path)
        _check(isinstance(value, dict) and set(value) == {
            "schema_version", "scope", "architecture", "target_directions", "candidate",
        } and type(value["schema_version"]) is int and value["schema_version"] == 1
            and value["scope"] == "research_only"
            and value["architecture"] == "single_multilingual_model",
            "Only the research-only single-model schema is supported")
        if expected_sha256 is not None:
            _check(_sha(expected_sha256) and _digest(raw) == expected_sha256,
                   "Registry does not match independent pin")
        targets = _directions(value["target_directions"])
        candidate = value["candidate"]
        self._target_directions = targets
        self._candidate = None
        if candidate is None:
            _check(evaluation_receipt is None, "No candidate to bind an evaluation receipt")
            return
        _check(_sha(expected_sha256) and evaluation_receipt is not None,
               "Candidate requires an independently pinned registry and evaluation receipt")
        _check(isinstance(candidate, dict) and set(candidate) == {
            "model_id", "revision", "architecture", "files", "verified_directions",
            "evaluation_receipt_sha256", "evaluation_authority",
        }, "Candidate must contain exactly one complete checkpoint and direction coverage")
        _check(isinstance(candidate["model_id"], str) and _ID.fullmatch(candidate["model_id"])
               and isinstance(candidate["revision"], str) and _REVISION.fullmatch(candidate["revision"])
               and isinstance(candidate["architecture"], str) and _ID.fullmatch(candidate["architecture"])
               and candidate["evaluation_authority"] == "independent_bilingual_review",
               "Invalid candidate identity or evaluation authority")
        verified = _directions(candidate["verified_directions"])
        _check(verified == targets, "Candidate evidence must cover every target direction")
        files = candidate["files"]
        _check(isinstance(files, dict) and len(files) <= 64 and "config.json" in files
               and bool(files.keys() & _TOKENIZERS) and "model.safetensors" in files
               and files.keys() <= _METADATA | {"model.safetensors"}
               and all(_FILE.fullmatch(name) and _sha(pin) for name, pin in files.items()),
               "Pin one complete, non-executable Safetensors checkpoint and tokenizer")
        identity = {key: candidate[key] for key in ("model_id", "revision", "architecture", "files")}
        pin = _fingerprint(identity)
        receipt_raw, receipt = _json(evaluation_receipt)
        _check(_sha(candidate["evaluation_receipt_sha256"]) and
               _digest(receipt_raw) == candidate["evaluation_receipt_sha256"],
               "Direction evaluation receipt does not match its independent pin")
        _check(isinstance(receipt, dict) and set(receipt) == {
            "schema_version", "kind", "checkpoint_sha256", "directions",
        } and type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
            and receipt["kind"] == "independent_bilingual_direction_receipt"
            and receipt["checkpoint_sha256"] == pin and isinstance(receipt["directions"], dict)
            and receipt["directions"].keys() == verified,
            "One checkpoint must independently pass every stated direction")
        for row in receipt["directions"].values():
            _check(isinstance(row, dict) and set(row) == {"status", "case_count", "evaluation_sha256"}
                   and row["status"] == "passed" and type(row["case_count"]) is int
                   and row["case_count"] > 0 and _sha(row["evaluation_sha256"]),
                   "Invalid, missing or failing per-direction evaluation")
        self._candidate = SingleModelSpec(
            candidate["model_id"], candidate["revision"], candidate["architecture"],
            pin, MappingProxyType(dict(files)), verified,
        )

    @property
    def target_directions(self) -> frozenset[str]:
        return self._target_directions

    @property
    def candidate(self) -> SingleModelSpec | None:
        return self._candidate

    def require_direction(self, source, target) -> tuple[str, str]:
        a, b = normalize_language(source), normalize_language(target)
        _check(f"{a}->{b}" in self._target_directions, "Unsupported translation direction")
        _check(self._candidate is not None, "No verified single multilingual model is configured")
        return a, b

    def verify_checkpoint(self, model_root: Path) -> Path:
        spec = self._candidate
        _check(spec is not None, "No verified single multilingual model is configured")
        root = Path(model_root)
        _check(root.is_dir() and not root.is_symlink(), "Expected an explicit local checkpoint directory")
        paths = {p.name: p for p in root.iterdir()}
        _check(paths.keys() == spec.files.keys(), "Local checkpoint files differ from immutable pins")
        for name, pin in spec.files.items():
            item = paths[name]
            _check(item.is_file() and not item.is_symlink() and _file_sha(item) == pin,
                   "Local checkpoint bytes differ from immutable pins")
        return root.resolve()
