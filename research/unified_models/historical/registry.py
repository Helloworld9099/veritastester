"""Archived schema and compatibility checks for explicit offline baseline replay."""
from __future__ import annotations

import hashlib
import json
import re
import struct
from array import array
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("registry.json")
_CODE = re.compile(r"[a-z]{2,3}\Z")
_LOCALE = re.compile(r"([a-z]{2,3})(?:-[a-z]{4})?(?:-(?:[a-z]{2}|[0-9]{3}))?\Z")
_REPO = re.compile(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\Z")
_MODULE = re.compile(r"model\.(encoder|decoder)\.layers\.(\d+)\.(self_attn|encoder_attn)\.(q_proj|v_proj)\Z")
_BASE_FILES = {
    "config.json", "generation_config.json", "source.spm", "target.spm",
    "vocab.json", "tokenizer_config.json", "README.md", "pytorch_model.bin",
    "model.safetensors",
}
_REQUIRED = {"config.json", "source.spm", "target.spm", "vocab.json"}


def _sha(value: object, bits: int = 256) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%s}" % (bits // 4), value) is not None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, val in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object field: " + key)
        result[key] = val
    return result


def _read_json(content: str | bytes):
    return json.loads(content, object_pairs_hook=_unique)


def compatibility_digest(base: dict, adapter_protocol: dict | None) -> str:
    """Binds model and revision, every model/tokenizer hash, and adapter semantics."""
    return hashlib.sha256(json.dumps({"base": base, "adapter_protocol": adapter_protocol},
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def safe_local_path(root: Path, relative: str) -> Path:
    """Reject noncanonical paths and every symlink in the relative path."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Only repository-relative POSIX paths are allowed")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("Noncanonical relative artifact path")
    target = root.resolve()
    for part in parts:
        target = target / part
        if target.is_symlink():
            raise ValueError("Symlinked artifact paths are not supported")
    return target


def normalize_language(code: str, *, languages: Mapping | None = None) -> str:
    """Resolve an explicit declared language ID with optional BCP-47 script/region."""
    if languages is None:
        languages = _read_json(MANIFEST.read_bytes())["languages"]
    match = _LOCALE.fullmatch(code.strip().lower()) if isinstance(code, str) else None
    if match is None or match[1] not in languages:
        raise ValueError("Explicit declared language code required; ASR 'auto' is unsupported")
    return match[1]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    source: str
    target: str
    model: str
    revision: str
    weights_safetensors: bool
    files: Mapping
    protocol: Mapping | None
    compatibility_sha256: str
    active_adapter: Mapping | None
    rollback_adapter: Mapping | None
    rights_review_status: str
    supports_context: bool = False
    human_meaning_verified: bool = False

    def adapter(self, *, use_previous: bool = False) -> Mapping | None:
        return self.rollback_adapter if use_previous else self.active_adapter

    def cache_key(self, *, use_previous: bool = False) -> str:
        chosen = self.adapter(use_previous=use_previous)
        return f"{self.model_id}/{self.compatibility_sha256}/" + (
            chosen["sha256"] if chosen else self.revision)


def _parse_protocol(value):
    if value is None:
        return None
    if set(value) != {"type", "rank", "alpha", "dtype", "projection_size",
                      "target_modules", "protocol_sha256"}:
        raise ValueError("Incomplete LoRA protocol")
    targets = value["target_modules"]
    if (value["type"] != "lora_attention_projection" or value["dtype"] != "F32"
            or not _sha(value["protocol_sha256"]) or not isinstance(targets, list)
            or not 1 <= len(targets) <= 128 or len(targets) != len(set(targets))
            or any(not _MODULE.fullmatch(x) for x in targets)):
        raise ValueError("Invalid attention-projection protocol")
    for key, limit in (("rank", 128), ("alpha", 1024), ("projection_size", 4096)):
        if type(value[key]) is not int or not 0 < value[key] <= limit:
            raise ValueError("Invalid LoRA dimensions")
    if value["rank"] > value["projection_size"]:
        raise ValueError("LoRA rank exceeds projection width")
    return MappingProxyType({**value, "target_modules": tuple(targets)})


def _parse_base(base):
    if set(base) != {"model", "revision", "architecture", "use_safetensors", "files"}:
        raise ValueError("Incomplete pinned base")
    if (not _REPO.fullmatch(base["model"]) or not _sha(base["revision"], 160)
            or base["architecture"] != "marian" or type(base["use_safetensors"]) is not bool):
        raise ValueError("Expected immutable Marian base identity")
    files = base["files"]
    weight = "model.safetensors" if base["use_safetensors"] else "pytorch_model.bin"
    other_weight = "pytorch_model.bin" if base["use_safetensors"] else "model.safetensors"
    if not (_REQUIRED | {weight}) <= files.keys() or not files.keys() <= _BASE_FILES or other_weight in files:
        raise ValueError("Incomplete or unexpected pinned Marian base manifest")
    for name, row in files.items():
        if (set(row) - {"file", "sha256", "size_bytes"} or row.get("file") != name
                or not _sha(row.get("sha256")) or ("size_bytes" in row and (
                    type(row["size_bytes"]) is not int or row["size_bytes"] <= 0))):
            raise ValueError("Invalid base file pin")
    return MappingProxyType({name: MappingProxyType(row) for name, row in files.items()})


class ModelRegistry:
    def __init__(self, path: Path = MANIFEST, *, root: Path = ROOT):
        self.root = root.resolve()
        raw = _read_json(Path(path).read_bytes())
        if (raw.get("schema_version") != 2 or raw.get("status") != "research_only_opt_in"
                or type(raw.get("cache_max_loaded_models")) is not int
                or raw["cache_max_loaded_models"] != 1):
            raise ValueError("Unsafe registry version, deployment state or cache policy")
        languages = {}
        for code, value in raw["languages"].items():
            if (not _CODE.fullmatch(code) or set(value) != {
                    "name", "script", "writing_direction", "normalization",
                    "tokenization", "dialects", "code_switching",
                } or not isinstance(value["name"], str) or not value["name"].strip()
                    or not re.fullmatch(r"[A-Z][a-z]{3}", value["script"])
                    or value["writing_direction"] not in {"ltr", "rtl"}
                    or value["normalization"] != "preserve_current_case_punctuation_collapse_whitespace"
                    or value["tokenization"] not in {"route_specific_sentencepiece", "not_available"}
                    or value["code_switching"] != "unsupported_no_language_guess"
                    or not isinstance(value["dialects"], list)):
                raise ValueError("Invalid declared language metadata")
            if any(not isinstance(locale, str) or locale == code
                   or normalize_language(locale, languages=raw["languages"]) != code
                   for locale in value["dialects"]):
                raise ValueError("Invalid dialect declaration")
            languages[code] = MappingProxyType({**value, "dialects": tuple(value["dialects"])})
        if not languages:
            raise ValueError("No declared languages")
        self.languages = MappingProxyType(languages)
        archives = raw["archived_noncommercial"]
        for item in archives:
            if (not _sha(item.get("sha256")) or not _sha(item.get("commit"), 160)
                    or not _REPO.fullmatch(item["repository"]) or item.get("routable") is not False):
                raise ValueError("Noncommercial archive cannot be a route")
            safe_local_path(self.root, item["path"])
        self.archived_noncommercial = tuple(MappingProxyType(x) for x in archives)
        nc_hashes = {x["sha256"] for x in archives}
        inactive = raw["inactive_private"]
        if inactive != {"direction": "cmn-en", "status": "private_noncommercial_inactive", "routable": False}:
            raise ValueError("Private inventory cannot disclose artifact metadata or be routed")
        self.inactive_private = MappingProxyType(inactive)

        entries = {}
        claimed = set()
        seen_hashes = set()
        for row in raw["models"]:
            key, src, tgt = row["model_id"], row["source"], row["target"]
            if (key in entries or key != f"{src}-{tgt}" or src == tgt
                    or src not in self.languages or tgt not in self.languages
                    or self.languages[src]["tokenization"] == "not_available"
                    or self.languages[tgt]["tokenization"] == "not_available"
                    or key == inactive["direction"]):
                raise ValueError("Invalid/duplicate direction or undeclared language")
            for state in ("supports_context", "human_meaning_verified", "noncommercial_data_in_training"):
                if row.get(state) is not False:
                    raise ValueError("Registry cannot enable NC, context, or human evaluation claims")
            if not row["rights_review_status"].startswith("research_only_"):
                raise ValueError("Production activation is outside this registry")
            if not row.get("training_origin") or not all(isinstance(s, str) for s in row["training_origin"]):
                raise ValueError("Training provenance required")
            if any(re.search(r"(?:\b(?:noncommercial|non-commercial)\b|CC[ -]BY[ -]NC)", s, re.I)
                   for s in row["training_origin"]):
                raise ValueError("Restricted training ancestry cannot be routed")
            base = row["base"]
            files = _parse_base(base)
            protocol = _parse_protocol(row["adapter_protocol"])
            compat = compatibility_digest(base, row["adapter_protocol"])
            if row["compatibility_sha256"] != compat:
                raise ValueError("Base/tokenizer/adapter compatibility digest does not match: " + key)
            if bool(row.get("active_adapter")) != bool(protocol) or bool(row.get("rollback_adapter")) != bool(protocol):
                raise ValueError("Adapted routes need both current and rollback; pretrained ones need neither")
            selected = {}
            for slot, role in (("active_adapter", "current"), ("rollback_adapter", "rollback")):
                ident = row.get(slot)
                if ident is None:
                    selected[slot] = None
                    continue
                checkpoint = raw["checkpoints"].get(ident)
                if not checkpoint or ident in claimed:
                    raise ValueError("Missing/duplicate checkpoint identity")
                sha = checkpoint.get("sha256")
                if (checkpoint.get("role") != role or checkpoint.get("model_id") != key
                        or checkpoint.get("compatibility_sha256") != compat
                        or checkpoint.get("format") != "safetensors"
                        or not _sha(checkpoint.get("commit"), 160) or not _sha(sha)
                        or not _REPO.fullmatch(checkpoint.get("repository", ""))
                        or type(checkpoint.get("bytes")) is not int or not 0 < checkpoint["bytes"] <= 32_000_000):
                    raise ValueError("Checkpoint identity, role or base compatibility mismatch")
                if sha in nc_hashes or sha in seen_hashes:
                    raise ValueError("Restricted or duplicate checkpoint SHA")
                if not checkpoint["path"].endswith(".safetensors"):
                    raise ValueError("Adapter must use safetensors")
                safe_local_path(self.root, checkpoint["path"])
                claimed.add(ident)
                seen_hashes.add(sha)
                selected[slot] = MappingProxyType(checkpoint)
            entries[key] = ModelSpec(key, src, tgt, base["model"], base["revision"],
                                     base["use_safetensors"], files, protocol, compat,
                                     selected["active_adapter"], selected["rollback_adapter"],
                                     row["rights_review_status"])
        if not entries or claimed != raw["checkpoints"].keys():
            raise ValueError("Missing routes or unclaimed checkpoint metadata")
        self.entries = MappingProxyType(entries)

    def select(self, source: str, target: str = "en") -> ModelSpec:
        key = normalize_language(source, languages=self.languages) + "-" + normalize_language(
            target, languages=self.languages)
        if key not in self.entries:
            raise ValueError(f"No pinned local model for {key}; no provider fallback attempted")
        return self.entries[key]

    def verify_base(self, spec: ModelSpec, base_cache: Path) -> Path:
        path = safe_local_path(base_cache, f"{spec.model_id}/{spec.revision}")
        if not path.is_dir():
            raise FileNotFoundError("Pinned research base cache unavailable: " + spec.model_id)
        if set(p.name for p in path.iterdir()) != spec.files.keys():
            raise ValueError("Missing or unpinned file in research base cache: " + spec.model_id)
        for name, row in spec.files.items():
            local = safe_local_path(path, name)
            if (not local.is_file() or ("size_bytes" in row and local.stat().st_size != row["size_bytes"])
                    or file_sha256(local) != row["sha256"]):
                raise ValueError("Pinned base/tokenizer checksum mismatch: " + name)
        config = _read_json((path / "config.json").read_bytes())
        if config.get("model_type") != "marian" or config.get("is_encoder_decoder") is not True:
            raise ValueError("Base architecture incompatible with Marian encoder-decoder")
        if spec.protocol:
            if config.get("d_model") != spec.protocol["projection_size"]:
                raise ValueError("Pinned adapter projection width differs from base")
            for layer in spec.protocol["target_modules"]:
                kind, index, attention, _ = _MODULE.fullmatch(layer).groups()
                if int(index) >= config[kind + "_layers"] or (kind == "encoder" and attention != "self_attn"):
                    raise ValueError("Adapter target unavailable in pinned architecture")
        return path

    def verify_adapter(self, spec: ModelSpec, artifact_root: Path | None, *, use_previous=False) -> bytes | None:
        record = spec.adapter(use_previous=use_previous)
        if record is None:
            if use_previous:
                raise ValueError("No previous trained adapter exists for " + spec.model_id)
            return None
        if artifact_root is None:
            raise FileNotFoundError("An explicit local pinned research artifact source is required")
        path = safe_local_path(artifact_root, record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError("Pinned adapter missing or wrong length: " + spec.model_id)
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError("Pinned adapter checksum mismatch: " + spec.model_id)
        _verify_tensor_bytes(blob, spec.protocol)
        return blob

    def verify_sources(self, *, artifact_root: Path | None = None,
                       base_cache: Path | None = None) -> dict:
        """Verify runnable pins locally; NC/private inventory never supplies weights here."""
        result = {}
        for key, spec in self.entries.items():
            if base_cache is not None:
                self.verify_base(spec, base_cache)
            for use_previous in (False, True) if spec.rollback_adapter else (False,):
                self.verify_adapter(spec, artifact_root, use_previous=use_previous)
            result[key] = {
                "model": spec.model, "revision": spec.revision,
                "tokenizer_hashes": {name: spec.files[name]["sha256"]
                                     for name in ("source.spm", "target.spm", "vocab.json")},
                "active_sha256": spec.active_adapter["sha256"] if spec.active_adapter else None,
                "rollback_sha256": spec.rollback_adapter["sha256"] if spec.rollback_adapter else None,
            }
        return result


def _verify_tensor_bytes(blob: bytes, spec: Mapping) -> None:
    """Validate key layout, float type/shape/offsets, and finite values before model load."""
    if len(blob) < 9:
        raise ValueError("Truncated safetensors file")
    header_size = struct.unpack_from("<Q", blob)[0]
    if not 0 < header_size <= min(len(blob) - 8, 100_000):
        raise ValueError("Invalid safetensors header length")
    header = _read_json(blob[8:8+header_size])
    if not isinstance(header, dict):
        raise ValueError("Invalid safetensors header")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and (not isinstance(metadata, dict)
                                 or any(not isinstance(v, str) for v in metadata.values())):
        raise ValueError("Invalid safetensors metadata")
    r, d = spec["rank"], spec["projection_size"]
    expected = {f"{target}.lora_{s}": shape for target in spec["target_modules"]
                for s, shape in (("A", [r, d]), ("B", [d, r]))}
    if header.keys() != expected.keys():
        raise ValueError("Adapter tensor keys incompatible with pinned model")
    intervals = []
    for key, shape in expected.items():
        rec = header[key]
        if (rec.get("shape") != shape or rec.get("dtype") != spec["dtype"]
                or any(type(x) is not int for x in rec["shape"])):
            raise ValueError("Adapter tensor dtype or shape incompatible with pinned model")
        offsets = rec.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets) or offsets[0] < 0
                or offsets[1] - offsets[0] != 4 * r * d):
            raise ValueError("Invalid safetensors offsets")
        intervals.append(offsets)
    consumed = 0
    for start, end in sorted(intervals):
        if consumed != start:
            raise ValueError("Safetensors data ranges overlap or have gaps")
        consumed = end
    if 8 + header_size + consumed != len(blob):
        raise ValueError("Safetensors body length mismatch")
    data = array("f")
    data.frombytes(blob[8+header_size:])
    import sys
    if sys.byteorder != "little":
        data.byteswap()
    if any(not isfinite(x) for x in data):
        raise ValueError("Nonfinite adapter tensor")
