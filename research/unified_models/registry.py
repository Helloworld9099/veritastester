"""One pinned research registry for ES->EN, FR->EN and EN->ES Marian.

No model weights are mixed, no provider default changes, no silent language
fallback, no WMT24/NC checkpoints allowed into the active model registry.
This module is pure Python stdlib; it does not download models.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research/unified_models/registry.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SUPPORTED = frozenset({"es-en", "fr-en", "en-es"})


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_local_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Only nonempty repository-relative POSIX paths are allowed")
    source = Path(relative)
    if source.is_absolute() or ".." in source.parts or "." in source.parts:
        raise ValueError("Model path must stay inside repository")
    base = root.resolve()
    target = (base / source).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Model path escaped repository")
    return target


def normalize_language(code: str) -> str:
    """Normalize exact known language identifiers; never guess ASR language."""
    if not isinstance(code, str) or not re.fullmatch(r"[a-zA-Z]{2}(?:-[a-zA-Z]{2})?", code.strip()):
        raise ValueError("Explicit valid language code required; ASR 'auto' is unsupported")
    language = code.lower().split("-")[0]
    if language not in {"en", "es", "fr"}:
        raise ValueError("Local research translator does not support language: " + language)
    return language


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    source: str
    target: str
    model: str
    revision: str
    base_manifest: str | None
    baseline_report: str | None
    protocol: str | None
    weights_safetensors: bool
    active_adapter: dict | None
    rollback_adapter: dict | None
    rights_review_status: str
    supports_context: bool
    human_meaning_verified: bool

    def adapter(self, *, use_previous: bool = False) -> dict | None:
        return self.rollback_adapter if use_previous else self.active_adapter

    def cache_key(self, *, use_previous: bool = False) -> str:
        candidate = self.adapter(use_previous=use_previous)
        return self.model_id + "/" + (candidate["sha256"] if candidate else self.revision)


class ModelRegistry:
    def __init__(self, path: Path = MANIFEST, *, root: Path = ROOT):
        self.root = root.resolve()
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1 or raw.get("status") != "research_only_opt_in":
            raise ValueError("Unsafe model registry version or deployment state")
        if raw.get("never_merge_weights_between_language_specific_tokenizers") is not True:
            raise ValueError("Cannot blend ES and FR adapters or tokenizers")
        if raw.get("cache_max_loaded_models") != 1:
            raise ValueError("Consolidated model cache must be bounded to one base")
        if len(raw.get("models", [])) != len(_SUPPORTED):
            raise ValueError("Expected exactly the 3 separately authorized directions")
        self.entries: dict[str, ModelSpec] = {}
        for item in raw["models"]:
            key = item["model_id"]
            if key not in _SUPPORTED or key in self.entries:
                raise ValueError("Duplicate or unsupported model identity")
            src, tgt = key.split("-")
            if item["source"] != src or item["target"] != tgt:
                raise ValueError("Model route disagrees with source/target")
            if not _SHA1.fullmatch(item["base_revision"]):
                raise ValueError("Base model must use immutable 40-character Git revision")
            if item["supports_context"] is not False or item["human_meaning_verified"] is not False:
                raise ValueError("Do not claim independent human adequacy or context support")
            if item["noncommercial_data_in_training"] is not False:
                raise ValueError("Active model is descended from restricted data")
            if not item["rights_review_status"].startswith("research_only_"):
                raise ValueError("No production model can be auto-enabled from this registry")
            for field in ("active_adapter", "rollback_adapter"):
                record = item.get(field)
                if record is None:
                    continue
                if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                    raise ValueError("Adapter requires exact local path and checksum")
                safe_local_path(self.root, record["path"])
                if not _SHA256.fullmatch(record["sha256"]):
                    raise ValueError("Invalid adapter SHA256")
            if key == "en-es":
                if item["active_adapter"] is not None or item["rollback_adapter"] is not None:
                    raise ValueError("EN->ES is an original-weight baseline, not a trained adapter")
                if not item.get("base_report") or item.get("base_manifest"):
                    raise ValueError("EN->ES baseline provenance missing")
            else:
                if not item.get("active_adapter") or not item.get("rollback_adapter"):
                    raise ValueError("A trained direction requires pinned adapter and rollback")
                if not item.get("base_manifest") or not item.get("adapter_protocol"):
                    raise ValueError("Pinned adapter protocol and base file manifest required")
            for field in ("base_manifest", "base_report", "adapter_protocol"):
                if item.get(field):
                    safe_local_path(self.root, item[field])
            self.entries[key] = ModelSpec(
                model_id=key, source=src, target=tgt, model=item["base_model"],
                revision=item["base_revision"], base_manifest=item.get("base_manifest"),
                baseline_report=item.get("base_report"),
                protocol=item.get("adapter_protocol"),
                weights_safetensors=item["use_safetensors"],
                active_adapter=item.get("active_adapter"),
                rollback_adapter=item.get("rollback_adapter"),
                rights_review_status=item["rights_review_status"],
                supports_context=False, human_meaning_verified=False,
            )
        self.archived_noncommercial: tuple[dict, ...] = tuple(raw.get("archived_noncommercial", []))
        if {row["direction"] for row in self.archived_noncommercial} != {"fr-en", "es-en"}:
            raise ValueError("Both historic noncommercial checkpoint lineages must be preserved")
        for record in self.archived_noncommercial:
            safe_local_path(self.root, record["path"])
            if not _SHA256.fullmatch(record["sha256"]):
                raise ValueError("Invalid restricted-archive SHA")
            if any(spec.adapter() and
                   spec.adapter()["sha256"] == record["sha256"] for spec in self.entries.values()):
                raise ValueError("NC checkpoint leaked into active model")
            if any(spec.rollback_adapter and
                   spec.rollback_adapter["sha256"] == record["sha256"] for spec in self.entries.values()):
                raise ValueError("NC checkpoint leaked into rollback")

    def select(self, source: str, target: str = "en") -> ModelSpec:
        key = normalize_language(source) + "-" + normalize_language(target)
        if key not in self.entries:
            raise ValueError(f"No pinned local model for {key}; use the existing provider explicitly")
        return self.entries[key]

    def verify_sources(self) -> dict[str, Any]:
        """Check committed adapter bytes, ancestry and separate base/tokenizers."""
        details = {}
        for key, spec in self.entries.items():
            if spec.protocol:
                protocol = json.loads(safe_local_path(self.root, spec.protocol).read_text())
                source = protocol["model"]
                if source["model"] != spec.model or source["revision"] != spec.revision:
                    raise RuntimeError("Protocol base/revision mismatch: " + key)
                targets = protocol.get("target_modules")
                if len(targets) != 36 or len(set(targets)) != 36:
                    raise RuntimeError("Unexpected LoRA target architecture: " + key)
                base_files = json.loads(safe_local_path(self.root, spec.base_manifest).read_text())
                spm = {r["file"]: r["sha256"] for r in base_files if r["file"] in ("source.spm", "target.spm")}
                if set(spm) != {"source.spm", "target.spm"}:
                    raise RuntimeError("Separate SentencePiece tokenizer pin missing: " + key)
            else:
                report = json.loads(safe_local_path(self.root, spec.baseline_report).read_text())
                if report["model_sha"] != spec.revision or report["model"] != spec.model:
                    raise RuntimeError("Original baseline revision changed")
                if not report.get("model_file_sha256"):
                    raise RuntimeError("Original baseline file pins missing")
                spm = {name:sha for name,sha in report["model_file_sha256"].items()
                       if name in ("source.spm", "target.spm")}
                if set(spm) != {"source.spm", "target.spm"}:
                    raise RuntimeError("Baseline tokenizer pins missing")
            record = {"model": spec.model, "revision": spec.revision,
                      "tokenizer_hashes": spm, "model_status": spec.rights_review_status}
            for label in ("active", "rollback"):
                candidate = spec.active_adapter if label == "active" else spec.rollback_adapter
                if candidate is None:
                    record[label + "_sha256"] = None
                    continue
                blob = safe_local_path(self.root, candidate["path"])
                if not blob.is_file() or blob.is_symlink():
                    raise RuntimeError("Pinned model adapter unavailable: "+candidate["path"])
                if file_sha256(blob) != candidate["sha256"]:
                    raise RuntimeError("Pinned adapter file hash mismatch: "+key+" "+label)
                record[label+"_sha256"] = candidate["sha256"]
            details[key] = record
        # Same architecture rank does NOT make model-specific embeddings and
        # tokenizers interchangeable: no model carries other model's adapters.
        es = details["es-en"]["tokenizer_hashes"]
        fr = details["fr-en"]["tokenizer_hashes"]
        if es["source.spm"] == fr["source.spm"] or es["target.spm"] == fr["target.spm"]:
            raise RuntimeError("Unexpectedly identical French/Spanish tokenizer provenance")
        for nc in self.archived_noncommercial:
            blob = safe_local_path(self.root,nc["path"])
            if not blob.is_file() or blob.is_symlink() or file_sha256(blob) != nc["sha256"]:
                raise RuntimeError("Preserved noncommercial archive changed or went missing")
        return details
