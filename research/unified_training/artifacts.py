"""Immutable, byte-verified checkpoints for the offline research coordinator.

The store only deals in opaque bytes. It does not load weights, deserialize
optimizer state, have a production-promotion API, or write to an old artifact.
A backend must consume the verified byte snapshot, not reopen a mutable path.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType

from .config import ConfigError, DIR, SEMVER, _base, _exact, _one_of, _strategy, _text


MANIFEST_NAME = "manifest.json"
VERSION = 1
FULL_STATE_SNAPSHOT = "full_state_snapshot"
ALL_COMPONENTS = frozenset(("adapter", "optimizer", "scheduler", "sampler", "rng"))
RESUME_COMPONENTS = ALL_COMPONENTS - {"adapter"}
COMPONENT_FILES = {k: f"{k}.bin" for k in ALL_COMPONENTS}
SOURCE_MODES = frozenset(("base", "warm_start", "exact_resume"))
PURPOSES = frozenset(("integrity", "warm_start", "exact_resume"))
MAX_MANIFEST_BYTES = 1 << 20
_MANIFEST_KEYS = {
    "schema_version", "scope", "artifact_id", "step", "compatibility", "compatibility_sha256",
    "resume_fingerprint", "rights", "components", "resumability", "lineage", "ancestry_sha256",
}
_PARENT_KEYS = {"artifact_id", "manifest_sha256", "weights_sha256", "compatibility_sha256", "ancestry_sha256", "step", "rights"}
MAX_ANCESTORS = 4096


class ArtifactError(ValueError):
    """The store cannot prove the requested immutable checkpoint."""


def _require(value, message):
    if not value:
        raise ArtifactError(message)


def _hex(value, length=64):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _safe_id(value):
    return (
        type(value) is str
        and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", value) is not None
        and not value.endswith(".")
        and re.fullmatch(r"(?i)(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?", value) is None
    )


def _normal(value):
    if isinstance(value, Mapping):
        return {k: _normal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _canonical(value):
    return json.dumps(_normal(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json_sha(value):
    return _sha(_canonical(value))


def _unique_keys(pairs):
    out = {}
    for k, v in pairs:
        _require(k not in out, "duplicate manifest key")
        out[k] = v
    return out


def _validate_compatibility(info):
    # Reuse the config limits, including +build versions, 4-bit load flags and
    # full dotted targets. Device authorization is intentionally not warm
    # compatibility; the complete TrainingConfig enforces device/precision.
    try:
        obj = _exact(info, {"direction", "architecture", "backend", "base", "strategy"}, "checkpoint compatibility")
        direction = DIR.fullmatch(_text(obj["direction"], DIR, "explicit language direction"))
        _require(direction[1] != direction[2], "invalid artifact direction")
        arch = _one_of(obj["architecture"], {"marian", "causal_lm", "seq2seq_lm"}, "architecture")
        backend = _exact(obj["backend"], {"id", "version"}, "backend selector")
        _text(backend["id"], re.compile(r"[a-z][a-z0-9_]{0,47}\Z"), "backend identity")
        _text(backend["version"], SEMVER, "concrete backend version")
        _base(obj["base"], arch)
        _strategy(obj["strategy"], {"kind": "gpu", "gpu_opt_in": True})
    except ConfigError as exc:
        raise ArtifactError("invalid checkpoint compatibility: " + str(exc)) from None


def _check_rights(value):
    _require(type(value) is str and value in {"commercial_candidate", "private_noncommercial"},
             "unknown or invalid checkpoint rights")


def _config_rights(config):
    try:
        value = config.rights
    except AttributeError as exc:
        raise ArtifactError("training config lacks effective rights") from exc
    _check_rights(value)
    return value


def preflight_config(config):
    """Validate publishable immutable metadata before a backend is called."""
    _require(_safe_id(getattr(config, "output_id", None)), "invalid configured checkpoint output ID")
    compat = _normal(config.artifact_compatibility())
    _validate_compatibility(compat)
    _require(config.compatibility_sha256 == _json_sha(compat), "training config compatibility fingerprint mismatch")
    _require(_hex(config.resume_fingerprint), "invalid configured resume fingerprint")
    _config_rights(config)
    return compat


@dataclass(frozen=True)
class VerifiedArtifact:
    artifact_id: str
    manifest_sha256: str
    manifest: Mapping = field(repr=False)
    component_paths: Mapping = field(repr=False)
    component_bytes: Mapping = field(repr=False)

    @property
    def step(self):
        return self.manifest["step"]

    @property
    def weights_sha256(self):
        return self.manifest["components"]["adapter"]["sha256"]

    @property
    def compatibility_sha256(self):
        return self.manifest["compatibility_sha256"]

    @property
    def ancestry_sha256(self):
        return self.manifest["ancestry_sha256"]

    @property
    def resume_fingerprint(self):
        return self.manifest["resume_fingerprint"]

    @property
    def rights(self):
        return self.manifest["rights"]

    def source_pin(self, mode="warm_start"):
        """Return only config pin fields, not private paths or component bytes."""
        _require(mode in ("warm_start", "exact_resume"), "source pin must specify warm start or exact resume")
        p = {
            "mode": mode, "artifact_id": self.artifact_id, "manifest_sha256": self.manifest_sha256,
            "weights_sha256": self.weights_sha256, "compatibility_sha256": self.compatibility_sha256,
            "ancestry_sha256": self.ancestry_sha256, "rights": self.rights,
        }
        if mode == "exact_resume":
            _require(self.manifest["resumability"] == FULL_STATE_SNAPSHOT, "weights-only artifact cannot pin exact resume")
            p.update(resume_fingerprint=self.resume_fingerprint,
                     component_sha256={k: self.manifest["components"][k]["sha256"] for k in sorted(RESUME_COMPONENTS)})
        return p


def _manifest_dict(raw):
    try:
        return json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ArtifactError("invalid checkpoint manifest JSON") from exc


def _read_regular(path, *, exact_size=None, maximum_size=None):
    """Avoid following symlinks and never deserialize arbitrary checkpoint state."""
    _require(not path.is_symlink(), f"unexpected symlink: {path.name}")
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as f:
            fd = None
            observed = os.fstat(f.fileno())
            _require(stat.S_ISREG(observed.st_mode), f"not a regular artifact file: {path.name}")
            _require(exact_size is None or observed.st_size == exact_size, f"artifact byte length mismatch: {path.name}")
            _require(maximum_size is None or observed.st_size <= maximum_size, f"artifact file exceeds size limit: {path.name}")
            expected = exact_size if exact_size is not None else maximum_size if maximum_size is not None else observed.st_size
            data = f.read(expected + 1)
            _require(len(data) == observed.st_size, f"artifact file changed while being verified: {path.name}")
            return data
    except OSError:
        raise ArtifactError(f"cannot read immutable artifact file: {path.name}") from None
    finally:
        if fd is not None:
            os.close(fd)


def _validate_manifest(m, artifact_id):
    _require(isinstance(m, dict) and set(m) == _MANIFEST_KEYS, "invalid checkpoint manifest shape")
    _require(type(m["schema_version"]) is int and m["schema_version"] == VERSION and m["scope"] == "research-only",
             "unsupported checkpoint manifest schema or scope")
    _require(m["artifact_id"] == artifact_id and _safe_id(artifact_id), "artifact ID does not match manifest")
    _require(type(m["step"]) is int and 0 < m["step"] < 1 << 63, "invalid checkpoint step")
    _validate_compatibility(m["compatibility"])
    _require(_json_sha(m["compatibility"]) == m["compatibility_sha256"], "artifact compatibility checksum mismatch")
    _check_rights(m["rights"])
    _require(_hex(m["resume_fingerprint"]), "invalid checkpoint resume fingerprint")
    comps = m["components"]
    _require(isinstance(comps, dict) and (set(comps) == {"adapter"} or set(comps) == ALL_COMPONENTS),
             "checkpoint must contain adapter alone or all exact-resume components")
    for k, c in comps.items():
        _require(isinstance(c, dict) and set(c) == {"file", "sha256", "size_bytes"}
                 and c["file"] == COMPONENT_FILES[k] and _hex(c["sha256"])
                 and type(c["size_bytes"]) is int and 0 < c["size_bytes"] < 1 << 48,
                 f"invalid manifest metadata for {k}")
    _require(m["resumability"] == (FULL_STATE_SNAPSHOT if set(comps) == ALL_COMPONENTS else "weights_only"),
             "resumability does not match checkpoint components")
    lin = m["lineage"]
    _require(isinstance(lin, dict) and set(lin) == {"mode", "parent"} and lin["mode"] in SOURCE_MODES,
             "invalid artifact lineage shape")
    _require(_json_sha(lin) == m["ancestry_sha256"], "artifact ancestry checksum mismatch")
    p = lin["parent"]
    if lin["mode"] == "base":
        _require(p is None, "base artifact cannot claim a parent")
    else:
        _require(isinstance(p, dict) and set(p) == _PARENT_KEYS and _safe_id(p["artifact_id"]) and p["artifact_id"] != artifact_id
                 and all(_hex(p[k]) for k in ("manifest_sha256", "weights_sha256", "compatibility_sha256", "ancestry_sha256"))
                 and type(p["step"]) is int and p["step"] > 0, "invalid pinned ancestor")
        _require(p["compatibility_sha256"] == m["compatibility_sha256"], "checkpoint and parent architectures differ")
        _check_rights(p["rights"])
        _require(p["rights"] != "private_noncommercial" or m["rights"] == p["rights"],
                 "checkpoint rights cannot be broadened by descendant publication")
        if lin["mode"] == "exact_resume":
            _require(m["step"] > p["step"], "exact resume must publish a strictly later step")
            _require(m["rights"] == p["rights"], "exact resume cannot change checkpoint rights")


def _write_new(path, data):
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


class ArtifactStore:
    """A checkpoint directory shared by explicit immutable references and new outputs."""

    def __init__(self, root):
        # Do not dereference a root symlink or embed this value in manifests.
        self.root = Path(root).absolute()

    def _dir(self, artifact_id):
        _require(_safe_id(artifact_id), "invalid checkpoint artifact ID")
        _require(not self.root.is_symlink(), "artifact store root is a symlink")
        path = self.root / artifact_id
        _require(not path.is_symlink(), "artifact directory is a symlink")
        return path

    def require_new(self, artifact_id):
        """Read-only duplicate check before doing work; publication also reserves atomically."""
        d = self._dir(artifact_id)
        _require(not d.exists(), "checkpoint artifact ID already exists; use a new ID")

    def _read_manifest(self, artifact_id, expected_manifest_sha256):
        _require(_hex(expected_manifest_sha256), "independent manifest SHA-256 is required")
        d = self._dir(artifact_id)
        _require(d.is_dir(), "checkpoint artifact is absent")
        raw = _read_regular(d / MANIFEST_NAME, maximum_size=MAX_MANIFEST_BYTES)
        _require(_sha(raw) == expected_manifest_sha256, "checkpoint manifest SHA-256 mismatch")
        m = _manifest_dict(raw)
        _validate_manifest(m, artifact_id)
        try:
            canonical = _canonical(m)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ArtifactError("checkpoint manifest is not canonical") from exc
        _require(raw == canonical, "checkpoint manifest is not canonical")
        return d, m

    def _verify_ancestors(self, descendant):
        """Pin each historical manifest; only current components need a byte snapshot."""
        seen = {descendant["artifact_id"]}
        parent = descendant["lineage"]["parent"]
        while parent is not None:
            ident = parent["artifact_id"]
            _require(ident not in seen and len(seen) <= MAX_ANCESTORS, "invalid or unbounded artifact ancestry")
            seen.add(ident)
            try:
                _, previous = self._read_manifest(ident, parent["manifest_sha256"])
            except ArtifactError:
                raise ArtifactError("missing or invalid pinned ancestor manifest") from None
            _require(
                all(previous[k] == parent[k] for k in ("step", "rights", "ancestry_sha256", "compatibility_sha256"))
                and previous["components"]["adapter"]["sha256"] == parent["weights_sha256"],
                "ancestor manifest disagrees with declared lineage",
            )
            parent = previous["lineage"]["parent"]

    def verify(self, artifact_id, *, expected_manifest_sha256, config=None, purpose="integrity"):
        """Verify manifest bytes, all component bytes and the pinned ancestry.

        Warm start returns only the adapter snapshot even if the source checkpoint
        contains saved optimizer, scheduler, sampler or RNG bytes. Exact resume
        never substitutes fresh state for a missing or incompatible component.
        """
        _require(purpose in PURPOSES, "unknown artifact verification purpose")
        d, m = self._read_manifest(artifact_id, expected_manifest_sha256)
        self._verify_ancestors(m)
        if purpose in ("warm_start", "exact_resume"):
            _require(config is not None, "training config is required for checkpoint source verification")
        if config is not None:
            wanted = _normal(config.artifact_compatibility())
            _require(wanted == m["compatibility"] and config.compatibility_sha256 == m["compatibility_sha256"],
                     "checkpoint is incompatible with configured backend, base, tokenizer or adapter protocol")
        if purpose == "exact_resume":
            _require(m["resumability"] == FULL_STATE_SNAPSHOT, "weights-only checkpoint cannot be resumed exactly")
            _require(config.resume_fingerprint == m["resume_fingerprint"], "exact-resume fingerprint mismatch")
        keep = {"adapter"} if purpose == "warm_start" else set(m["components"])
        paths, components = {}, {}
        for k, info in m["components"].items():
            p = d / info["file"]
            b = _read_regular(p, exact_size=info["size_bytes"])
            _require(_sha(b) == info["sha256"], f"checkpoint component SHA-256 mismatch: {k}")
            if k in keep:
                paths[k], components[k] = p, b
        return VerifiedArtifact(artifact_id, expected_manifest_sha256, _freeze(m),
                                MappingProxyType(paths), MappingProxyType(components))

    def verify_source(self, config):
        """Lock the precise source named in the validated training config."""
        src = _normal(config.source)
        mode = src["mode"]
        _require(mode in SOURCE_MODES, "unrecognized checkpoint source mode")
        if mode == "base":
            _require(set(src) == {"mode"}, "base checkpoint source must not claim a parent")
            return None
        verified = self.verify(src.get("artifact_id"), expected_manifest_sha256=src.get("manifest_sha256"), config=config, purpose=mode)
        _require(verified.source_pin(mode) == src, "configured source pins do not match verified weights, rights, state or ancestry")
        return verified

    def publish(self, artifact_id, *, config, components, step, parent=None, mode=None):
        """Exclusively reserve a new ID, then seal its manifest after its bytes.

        ID reservation is atomic. A failed write remains an incomplete reserved
        ID, never a loadable checkpoint and never recycled. Existing artifacts,
        old aliases, and caller files are neither overwritten nor removed.
        """
        self.require_new(artifact_id)
        d = self._dir(artifact_id)
        _require(artifact_id == getattr(config, "output_id", None), "checkpoint artifact ID must match the configured output ID")
        _require(type(step) is int and 0 < step < 1 << 63, "invalid checkpoint step")
        requested_mode = config.source["mode"]
        _require(requested_mode in SOURCE_MODES and (mode is None or mode == requested_mode),
                 "published checkpoint mode must be the declared config source")
        source = self.verify_source(config)
        if source is None:
            _require(parent is None, "base checkpoint cannot name an ancestor")
            lineage = {"mode": requested_mode, "parent": None}
        else:
            _require(parent is None or (isinstance(parent, VerifiedArtifact) and parent.artifact_id == source.artifact_id
                     and parent.manifest_sha256 == source.manifest_sha256), "caller supplied a different checkpoint ancestor")
            lineage = {"mode": requested_mode, "parent": {"artifact_id": source.artifact_id,
                       "manifest_sha256": source.manifest_sha256, "weights_sha256": source.weights_sha256,
                       "compatibility_sha256": source.compatibility_sha256, "ancestry_sha256": source.ancestry_sha256,
                       "step": source.step, "rights": source.rights}}
        compat = preflight_config(config)
        rights = _config_rights(config)
        if source is not None:
            _require(source.rights != "private_noncommercial" or rights == source.rights,
                     "checkpoint rights cannot be broadened by descendant publication")
        _require(isinstance(components, Mapping) and (set(components) == {"adapter"} or set(components) == ALL_COMPONENTS),
                 "publication requires adapter bytes alone or all five exact-resume components")
        snapshots = dict(components)
        _require(all(type(b) is bytes and bool(b) for b in snapshots.values()), "checkpoint components must be nonempty bytes")
        m = {"schema_version": VERSION, "scope": "research-only", "artifact_id": artifact_id, "step": step,
             "compatibility": compat, "compatibility_sha256": config.compatibility_sha256,
             "resume_fingerprint": config.resume_fingerprint, "rights": rights, "lineage": lineage,
             "ancestry_sha256": _json_sha(lineage),
             "components": {k: {"file": COMPONENT_FILES[k], "sha256": _sha(b), "size_bytes": len(b)} for k, b in snapshots.items()},
             "resumability": FULL_STATE_SNAPSHOT if set(snapshots) == ALL_COMPONENTS else "weights_only"}
        _validate_manifest(m, artifact_id)
        raw = _canonical(m)
        _require(len(raw) <= MAX_MANIFEST_BYTES, "checkpoint manifest exceeds size limit")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            _require(not self.root.is_symlink(), "artifact store root is a symlink")
            d.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            raise ArtifactError("checkpoint artifact ID already exists; use a new ID") from None
        except OSError:
            raise ArtifactError("cannot reserve an immutable artifact ID in the configured store") from None
        try:
            for k, data in snapshots.items():
                _write_new(d / COMPONENT_FILES[k], data)
            # An exclusive hard link is the atomic publication point: it fails
            # if another manifest already exists and never replaces that file.
            tmp = d / ".manifest.pending"
            _write_new(tmp, raw)
            os.link(tmp, d / MANIFEST_NAME)
        except OSError:
            raise ArtifactError("checkpoint was not published; immutable ID reservation must not be reused") from None
        try:
            tmp.unlink()
        except OSError:
            pass  # valid manifest sealed; pending file is not loadable state.
        return self.verify(artifact_id, expected_manifest_sha256=_sha(raw), config=config)
