"""Local-only train/development manifest preparation for BridgeCaption.

Use a separately reviewed configuration and pinned source files. No downloads,
fixed-benchmark or sealed holdout reads, corpus writes, training, or publication.
By contract, put only authorized public train/development JSONL shards in the
provided local data root. This tool checks declarations, not actual consent or
license validity, and does not claim exact-text checks catch semantic overlap.

Configuration schema v1 (all listed keys required):

    {"schema_version": 1, "purpose": "public_research", "datasets": [{
      "dataset_id": "example", "version": "v1", "role": "train",
      "path": "public/example-train.jsonl", "sha256": "<trusted bytes digest>",
      "source_language": "zh", "target_language": "en",
      "origin_uri": "<versioned provenance>", "license": "CC-BY-4.0",
      "rights": "public_research", "rights_evidence": "<reviewed evidence>",
      "privacy": "public", "review_status": "reviewed",
      "review_evidence": "<review evidence>"
    }, ...]}

Every selected language direction must have both train and development shards.
Rows have exactly `id`, `source`, `target`, and optionally `context` (text or
ordered text array), `group_id`, `speaker_id`, `role`/`split`, `privacy`, or
`sealed`. Context must have a stable group ID. IDs, texts, paths, and terms are
never included in output. Compare the config and bundle hashes with separate
trusted pins to establish provenance; a receipt is not a signature.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
import hashlib
import json
import os
import re
import stat
import unicodedata

SCHEMA_VERSION = 1
RECEIPT_KIND = "bridgecaption_workflow_data"
ALLOWED_ROLES = frozenset(("train", "dev"))
ALLOWED_USES = frozenset(("public_research", "commercial"))
_BANNED_PATH_WORDS = frozenset(("test", "holdout", "heldout", "sealed", "private", "restricted", "quarantine"))
_CONFIG_FIELDS = frozenset(("schema_version", "purpose", "datasets"))
_ENTRY_FIELDS = frozenset((
    "dataset_id", "version", "role", "path", "sha256", "source_language", "target_language",
    "origin_uri", "license", "rights", "rights_evidence", "privacy", "review_status", "review_evidence",
))
_ROW_REQUIRED = frozenset(("id", "source", "target"))
_ROW_OPTIONAL = frozenset(("context", "group_id", "speaker_id", "role", "split", "privacy", "sealed"))
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
_LANG = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", re.ASCII)
_HEX = re.compile(r"[0-9a-f]{64}", re.ASCII)
_PATH_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", re.ASCII)
# Punctuation/space-insensitive audit keys are deliberately conservative; data
# files themselves are never rewritten or normalized.


class DataError(ValueError):
    """Invalid manifest, provenance, source, or split; diagnostic is redacted."""


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str) or not value.strip() or "\0" in value
        or any(0xD800 <= ord(char) <= 0xDFFF for char in value)
    ):
        raise DataError(f"{label} must be nonempty Unicode text")
    return value


def _identity(value: Any, label: str) -> str | int:
    return value if type(value) is int else _text(value, label)


def _language(value: Any) -> str:
    if not isinstance(value, str) or _LANG.fullmatch(value) is None:
        raise DataError("use explicit lowercase BCP 47-style language codes")
    return value


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _json(data: bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise DataError("duplicate JSON field")
            value[key] = item
        return value

    def nonfinite(_: str) -> None:
        raise DataError("nonfinite JSON number")

    return json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=nonfinite)


def _public_path(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        raise DataError("dataset path must be a relative JSONL path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or path.suffix.lower() != ".jsonl":
        raise DataError("dataset path must be a relative JSONL path")
    for part in path.parts:
        if _PATH_SEGMENT.fullmatch(part) is None:
            raise DataError("dataset path contains an unsafe segment")
        if _BANNED_PATH_WORDS.intersection(re.split(r"[^a-z0-9]+", part.lower())):
            raise DataError("sealed, test, restricted, or private dataset paths are not allowed")
        if re.search(r"(?:heldout|holdout|sealed)", part, re.IGNORECASE):
            raise DataError("sealed dataset paths are not allowed")
    return path.parts


def _guard_restrictions(entry: Mapping[str, Any], purpose: str) -> None:
    if entry["privacy"] != "public" or entry["review_status"] != "reviewed":
        raise DataError("only explicitly reviewed public datasets are allowed")
    if not isinstance(entry["rights"], str) or entry["rights"] not in ALLOWED_USES:
        raise DataError("explicit allowed data use is required")
    if purpose == "commercial" and entry["rights"] != "commercial":
        raise DataError("commercial workflow requires commercial data permission")
    license_name = _text(entry["license"], "license").casefold()
    if license_name.strip() in {"unknown", "unverified", "n/a", "unlicensed", "none", "no license"}:
        raise DataError("known license is required by the manifest")
    if re.search(r"(?:^|[^a-z])nd(?:$|[^a-z])|no[ -]?derivat", license_name):
        raise DataError("no-derivatives dataset is excluded by the workflow policy")
    if purpose == "commercial" or entry["rights"] == "commercial":
        if re.search(r"(?:^|[^a-z])nc(?:$|[^a-z])|non[ -]?commercial", license_name):
            raise DataError("commercial permission conflicts with the declared restriction")
    for field in ("origin_uri", "rights_evidence", "review_evidence"):
        value = _text(entry[field], field)
        if value.strip().casefold() in {"unknown", "unverified", "n/a", "none", "tbd"}:
            raise DataError("concrete origin and review evidence is required")


@dataclass(frozen=True)
class _Entry:
    dataset_id: str
    version: str
    role: str
    path_parts: tuple[str, ...]
    sha256: str
    source_language: str
    target_language: str

    @property
    def direction(self) -> str:
        return f"{self.source_language}->{self.target_language}"


def _config(config: Any) -> tuple[str, list[_Entry], str]:
    if not isinstance(config, Mapping) or set(config) != _CONFIG_FIELDS:
        raise DataError("invalid dataset configuration fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != SCHEMA_VERSION:
        raise DataError("unsupported dataset configuration version")
    purpose = config["purpose"]
    if not isinstance(purpose, str) or purpose not in ALLOWED_USES:
        raise DataError("explicit valid dataset purpose is required")
    items = config["datasets"]
    if not isinstance(items, list) or not items:
        raise DataError("configuration needs at least one train and one development shard")
    paths: set[tuple[str, ...]] = set()
    entries = []
    roles_by_direction: dict[str, set[str]] = {}
    # All declarations are validated before even the first dataset file opens.
    for item in items:
        if not isinstance(item, Mapping) or set(item) != _ENTRY_FIELDS:
            raise DataError("invalid dataset entry fields")
        role = item["role"]
        if not isinstance(role, str) or role not in ALLOWED_ROLES:
            raise DataError("only train and development roles may be read")
        _guard_restrictions(item, purpose)
        identity, version = item["dataset_id"], item["version"]
        if any(not isinstance(v, str) or _SLUG.fullmatch(v) is None for v in (identity, version)):
            raise DataError("dataset identity and version must be stable portable identifiers")
        parts = _public_path(item["path"])
        if parts in paths:
            raise DataError("one file cannot be declared twice")
        paths.add(parts)
        sha256 = item["sha256"]
        if not isinstance(sha256, str) or _HEX.fullmatch(sha256) is None:
            raise DataError("each input needs an explicit SHA-256 pin")
        source, target = _language(item["source_language"]), _language(item["target_language"])
        if source == target:
            raise DataError("translation direction must use distinct languages")
        entry = _Entry(identity, version, role, parts, sha256, source, target)
        roles_by_direction.setdefault(entry.direction, set()).add(role)
        entries.append(entry)
    if any(roles != ALLOWED_ROLES for roles in roles_by_direction.values()):
        raise DataError("each direction needs both a train and a development shard")
    return purpose, entries, _digest(_canonical(dict(config)))


def _open_public_regular(root: Path, segments: tuple[str, ...]) -> BinaryIO:
    """Descriptor-relative, no-symlink walk. No opens occur before config checks."""
    if any(not hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
        raise DataError("this platform does not support the safe local file-open policy")
    opened: list[int] = []
    fd: int | None = None
    try:
        prefix = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        opened.append(os.open(root, prefix))
        for piece in segments[:-1]:
            opened.append(os.open(piece, prefix, dir_fd=opened[-1]))
        fd = os.open(segments[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=opened[-1])
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise DataError("only regular local dataset files are permitted")
        stream = os.fdopen(fd, "rb")
        fd = None
        return stream
    except (OSError, NotImplementedError, TypeError):
        raise DataError("could not safely open a declared local dataset shard") from None
    finally:
        if fd is not None:
            os.close(fd)
        for directory in reversed(opened):
            os.close(directory)


def _sha_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _text_key(language: str, text: str) -> tuple[str, str]:
    value = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(ch for ch in value if unicodedata.category(ch)[0] in "LMNS")
    return (language.split("-", 1)[0], _digest((normalized or value.strip()).encode("utf-8")))


def _row(raw: bytes, role: str) -> tuple[dict[str, Any], list[str]]:
    try:
        current = _json(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise DataError("invalid JSONL record") from None
    if not isinstance(current, dict) or not _ROW_REQUIRED <= set(current) or set(current) - (_ROW_REQUIRED | _ROW_OPTIONAL):
        raise DataError("invalid dataset row fields")
    if any(key in current and current[key] != role for key in ("role", "split")):
        raise DataError("record role conflicts with the train/development manifest")
    if current.get("privacy", "public") != "public" or current.get("sealed", False) is not False:
        raise DataError("private, restricted, or sealed records are excluded")
    _identity(current["id"], "original row ID")
    for key in ("source", "target"):
        _text(current[key], key)
    for key in ("group_id", "speaker_id"):
        if current.get(key) is not None:
            _identity(current[key], key)
    context = current.get("context", [])
    if isinstance(context, str):
        context = [context]
    if not isinstance(context, list):
        raise DataError("context must be text or an array of prior text turns")
    for turn in context:
        _text(turn, "prior turn")
    if context and current.get("group_id") is None:
        raise DataError("context rows require a stable group ID")
    return current, context


def prepare(config: Mapping[str, Any], *, root: Path | str) -> dict[str, Any]:
    """Verify pinned local shards, audit train/dev leakage, return a safe receipt."""
    try:
        purpose, entries, config_sha = _config(config)
    except (TypeError, UnicodeError, RecursionError) as exc:
        raise DataError("malformed dataset configuration") from None
    root = Path(root)
    roles: dict[str, int] = {"train": 0, "dev": 0}
    directions: dict[str, dict[str, int]] = {}
    texts: dict[tuple[str, str], str] = {}
    grouped: dict[tuple[Any, ...], str] = {}
    identities: set[tuple[Any, ...]] = set()
    membership = hashlib.sha256()
    membership.update(b"bridgecaption-workflow-train-dev-membership-v1\n")
    for entry in entries:
        family = entry.dataset_id.casefold()
        counts = directions.setdefault(entry.direction, {"train": 0, "dev": 0})
        try:
            with _open_public_regular(root, entry.path_parts) as stream:
                if _sha_stream(stream) != entry.sha256:
                    raise DataError("declared dataset bytes do not match their SHA-256 pin")
                stream.seek(0)
                second_sha = hashlib.sha256()
                n = 0
                for raw in stream:
                    second_sha.update(raw)
                    n += 1
                    record, context = _row(raw, entry.role)
                    row_id = (family, entry.version, record["id"])
                    if row_id in identities:
                        raise DataError("duplicate original ID in a dataset release")
                    identities.add(row_id)
                    for kind, identifier in [("id", record["id"]), ("group", record.get("group_id")),
                                             ("speaker", record.get("speaker_id"))]:
                        if identifier is not None:
                            key = (family, kind, identifier)
                            if grouped.setdefault(key, entry.role) != entry.role:
                                raise DataError("train/development overlap in original identity, group, or speaker")
                    keys = [_text_key(entry.source_language, text) for text in (record["source"], *context)]
                    keys.append(_text_key(entry.target_language, record["target"]))
                    for key in keys:
                        if texts.setdefault(key, entry.role) != entry.role:
                            raise DataError("normalized train/development text overlap")
                    # Feed row anchors in pinned input order; only the final
                    # aggregate fingerprint leaves this function.
                    membership.update(_canonical((entry.role, entry.direction, row_id, keys)))
                    membership.update(b"\n")
                if second_sha.hexdigest() != entry.sha256:
                    raise DataError("dataset bytes changed during verification")
        except (OSError, UnicodeError, TypeError, RecursionError):
            raise DataError("unable to validate declared local dataset") from None
        if not n:
            raise DataError("empty dataset shard")
        roles[entry.role] += n
        counts[entry.role] += n
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": "verified",
        "config_sha256": config_sha,
        "membership_sha256": membership.hexdigest(),
        "purpose": purpose,
        "files": len(entries),
        "rows": roles,
        "directions": dict(sorted(directions.items())),
    }


def verify(config: Mapping[str, Any], receipt: Mapping[str, Any], *, root: Path | str) -> dict[str, Any]:
    """Re-read exact local inputs and require a byte-canonical identical receipt."""
    try:
        supplied = _canonical(dict(receipt)) if isinstance(receipt, Mapping) else b""
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise DataError("invalid pinned dataset receipt") from None
    fresh = prepare(config, root=root)
    if supplied != _canonical(fresh):
        raise DataError("dataset receipt does not match the reverified pinned inputs")
    return fresh


prepare_dataset_manifest = prepare
verify_dataset_manifest = verify
