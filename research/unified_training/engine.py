"""Offline orchestration for explicitly supplied, reviewed local training backends.

The built-in legacy adapter is inspection-only. No JSON field can load Python
or spawn a command. The executable integration point is a trusted, in-process
backend explicitly supplied by a developer; it is not a network/OS sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import stat
import unicodedata
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from .artifacts import ArtifactStore, preflight_config
from .config import TrainingConfig

_MANIFEST_MAX_BYTES = 256 * 1024
_RECORDS_MAX_BYTES = 64 * 1024 * 1024
_TOTAL_RECORDS_MAX_BYTES = 128 * 1024 * 1024
_DEV_SPLITS = {"train", "dev"}


class EngineError(ValueError):
    """Training is blocked; error text excludes local paths and data."""


def inspect_legacy(config: TrainingConfig, *, repo_root: Path):
    """Report which immutable legacy path/config is known, never invoke it."""
    from .backends import backend_for
    backend = backend_for(config.backend["id"], config.backend["version"])
    return backend.plan(config, repo_root=repo_root)


@dataclass(frozen=True)
class VerifiedData:
    dataset_id: str = field(repr=False)
    version: str = field(repr=False)
    split: str
    manifest_sha256: str
    records_sha256: str
    records: bytes = field(repr=False)


@dataclass(frozen=True)
class RunInputs:
    config: TrainingConfig = field(repr=False)
    base_root: Path = field(repr=False)
    train_data: tuple[VerifiedData, ...] = field(repr=False)
    development_data: tuple[VerifiedData, ...] = field(repr=False)
    source_components: Mapping[str, bytes] = field(repr=False)
    start_step: int
    target_step: int


@dataclass(frozen=True)
class BackendResult:
    completed_step: int
    components: Mapping[str, bytes] = field(repr=False)
    restored_source_sha256: Mapping[str, str] | None = field(default=None, repr=False)
    interrupted: bool = False


@dataclass(frozen=True)
class LocalBackend:
    """Trusted injection boundary. No config-controlled callable lookup."""

    id: str
    version: str
    architectures: frozenset[str]
    strategies: frozenset[str]
    devices: frozenset[str]
    supports_exact_resume: bool
    run: Callable[[RunInputs], BackendResult]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or "\0" in relative:
        raise EngineError("Invalid relative input path")
    parts = relative.split("/")
    if any(p in {"", ".", ".."} for p in parts):
        raise EngineError("Invalid relative input path")
    target = root
    for part in parts:
        target /= part
        if target.is_symlink():
            raise EngineError("Symlinked input is not supported")
    return target


def _root(root: Path) -> Path:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise EngineError("A local, nonsymlink input directory is required")
    return root.resolve()


def _open_regular(path: Path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise EngineError("Local input must be a regular file")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def _pinned_bytes(path: Path, expected_digest: str, maximum: int) -> bytes:
    if not isinstance(expected_digest, str) or len(expected_digest) != 64 or not path.is_file():
        raise EngineError("Pinned local input is missing or invalid")
    try:
        with _open_regular(path) as stream:
            data = stream.read(maximum + 1)
    except OSError:
        raise EngineError("Could not read pinned local input") from None
    if not data or len(data) > maximum or _sha(data) != expected_digest:
        raise EngineError("Pinned local input checksum or size mismatch")
    return data


def _no_duplicates(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise EngineError("Duplicate data metadata field")
        data[key] = value
    return data


def _reject_constant(_):
    raise EngineError("Non-finite data literals are not supported")


def _json(data):
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicates,
                          parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError):
        raise EngineError("Invalid data metadata or record encoding") from None


def _clean_text(value: object) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and "\0" not in value
            and all(not 0xD800 <= ord(c) <= 0xDFFF for c in value))


def _datasets(config: TrainingConfig, directory: Path) -> tuple[VerifiedData, ...]:
    root = _root(directory)
    source, target = config.direction.split("->")
    result = []
    total_bytes = 0
    seen: dict[tuple[str, str], str] = {}
    ids, groups = {}, {}
    required = {"split", "source_language", "target_language", "source", "target"}
    allowed = required | {"original_id", "group_id"}
    for spec in config.data:
        split = spec["split"]
        if split not in _DEV_SPLITS:
            raise EngineError("Only prepared, unsealed train and development inputs are accepted")
        path = _safe_path(root, spec["local_path"])
        meta = _json(_pinned_bytes(path, spec["manifest_sha256"], _MANIFEST_MAX_BYTES))
        fields = {
            "schema_version", "kind", "dataset_id", "version", "split", "sealed", "license",
            "visibility", "records_path", "records_sha256", "row_count",
        }
        if (not isinstance(meta, dict) or set(meta) != fields
                or type(meta.get("schema_version")) is not int or meta["schema_version"] != 1
                or meta.get("kind") != "reviewed_research_training_split" or meta.get("sealed") is not False
                or any(meta.get(k) != spec[k] for k in ("dataset_id", "version", "split", "license", "visibility"))
                or type(meta.get("row_count")) is not int or meta["row_count"] < 1):
            raise EngineError("Dataset manifest is not a matching, unsealed train/dev release")
        # Manifest records are local and relative to their own review directory.
        data = _pinned_bytes(_safe_path(path.parent, meta["records_path"]),
                             meta["records_sha256"], _RECORDS_MAX_BYTES)
        total_bytes += len(data)
        if total_bytes > _TOTAL_RECORDS_MAX_BYTES:
            raise EngineError("Prepared dataset inputs exceed the offline snapshot budget")
        rows = data.splitlines()
        if len(rows) != meta["row_count"]:
            raise EngineError("Prepared dataset row count does not match its manifest")
        for row in rows:
            parsed = _json(row)
            if (not isinstance(parsed, dict) or not required <= set(parsed) <= allowed
                    or parsed.get("split") != split
                    or parsed.get("source_language") != source or parsed.get("target_language") != target
                    or not _clean_text(parsed.get("source")) or not _clean_text(parsed.get("target"))):
                raise EngineError("Dataset record has incompatible schema, language, split or content")
            for field_name, table in (("original_id", ids), ("group_id", groups)):
                if field_name not in parsed:
                    continue
                token = parsed[field_name]
                if type(token) is not int and not _clean_text(token):
                    raise EngineError("Dataset identity or group is invalid")
                key = (spec["dataset_id"].casefold(), token)
                if table.setdefault(key, split) != split:
                    raise EngineError("Dataset identities or groups overlap across train and development")
            # Defense in depth; this is not M1 license review or semantic
            # near-duplicate detection. Text is compared but not rewritten.
            for language, text in ((source, parsed["source"]), (target, parsed["target"])):
                normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
                if seen.setdefault((language, normalized), split) != split:
                    raise EngineError("Text overlap between train and development inputs")
        result.append(VerifiedData(spec["dataset_id"], spec["version"], split,
                                   spec["manifest_sha256"], meta["records_sha256"], data))
    return tuple(result)


def _verify_base(config: TrainingConfig, root: Path) -> Path:
    root = _root(root)
    expected = config.base["files"]
    actual = set()
    # Compare the complete local loader directory; fail on additional files
    # that could alter tokenizer/config choice even if pinned files are intact.
    for p in root.rglob("*"):
        if p.is_symlink() or (not p.is_dir() and not p.is_file()):
            raise EngineError("Unexpected artifact in pinned base directory")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if actual != set(expected):
        raise EngineError("Base directory contents differ from immutable base pins")
    for name, pin in expected.items():
        path = _safe_path(root, name)
        try:
            digest = hashlib.sha256()
            with _open_regular(path) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError:
            raise EngineError("Could not verify pinned base") from None
        if digest.hexdigest() != pin:
            raise EngineError("Pinned base/tokenizer checksum mismatch")
    return root


def run_local(
    config: TrainingConfig, *, backend: LocalBackend, store, base_root: Path, data_root: Path,
    authorized: bool = False,
):
    """Check all pins, execute only a trusted adapter, save an immutable result.

    No default backend executes. The adapter owns model-specific tensor and
    exact state restoration; this orchestrator gates pinned state and evidence.
    Backend registration is code review, not a string or command in a config.
    """
    if authorized is not True:
        raise EngineError("Execution requires explicit local authorization")
    if not isinstance(config, TrainingConfig):
        raise EngineError("Use a validated immutable training configuration")
    if not isinstance(backend, LocalBackend) or (
        (backend.id, backend.version) != (config.backend["id"], config.backend["version"])
        or config.architecture not in backend.architectures
        or config.strategy["kind"] not in backend.strategies
        or config.device["kind"] not in backend.devices
        or not callable(backend.run)
    ):
        raise EngineError("Unsupported training backend, architecture, strategy or device")
    mode = config.source["mode"]
    if mode == "exact_resume" and backend.supports_exact_resume is not True:
        raise EngineError("This backend does not support exact optimizer-state resume")
    if not isinstance(store, ArtifactStore):
        raise EngineError("A local immutable artifact store is required")
    destination = store.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    # Do not put a new write destination inside an input snapshot or a checkout.
    for protected in (Path(base_root).resolve(), Path(data_root).resolve(), repo):
        if destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise EngineError("Artifact store must be separate from input roots and the checkout")
    preflight_config(config)
    store.require_new(config.output_id)
    parent = store.verify_source(config)
    start = parent.manifest["step"] if mode == "exact_resume" else 0
    target = config.schedule["total_updates"]
    if start >= target:
        raise EngineError("Exact resume has no remaining configured updates")
    base = _verify_base(config, base_root)
    data = _datasets(config, data_root)
    components = parent.component_bytes if parent else MappingProxyType({})
    try:
        result = backend.run(RunInputs(
            config, base, tuple(release for release in data if release.split == "train"),
            tuple(release for release in data if release.split == "dev"), components, start, target,
        ))
    except Exception:
        raise EngineError("Local backend did not complete; no final artifact published") from None
    if (not isinstance(result, BackendResult) or type(result.completed_step) is not int
            or type(result.interrupted) is not bool):
        raise EngineError("Backend returned an invalid progress receipt")
    if result.interrupted:
        if not start < result.completed_step < target or backend.supports_exact_resume is not True:
            raise EngineError("Interrupted run must return a supported intermediate state")
    elif result.completed_step != target:
        raise EngineError("Backend did not complete the requested update target")
    if not isinstance(result.components, Mapping):
        raise EngineError("Backend did not supply checkpoint components")
    if (mode == "exact_resume" or result.interrupted) and set(result.components) != {
        "adapter", "optimizer", "scheduler", "sampler", "rng",
    }:
        raise EngineError("A resumable result must return all five state components")
    if mode == "exact_resume":
        expected_restore = {name: _sha(data) for name, data in components.items()}
        if result.restored_source_sha256 != expected_restore:
            raise EngineError("Exact continuation lacks the reviewed backend's bound restore receipt")
    elif result.restored_source_sha256 is not None:
        raise EngineError("A base or warm start must not claim exact state restoration")
    # A reviewed backend must treat the base location as read-only. Recheck
    # before issuing a receipt instead of certifying a mutated loader directory.
    _verify_base(config, base)
    return store.publish(config.output_id, config=config, components=result.components,
                         step=result.completed_step, parent=parent, mode=mode)
