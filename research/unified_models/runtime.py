"""Single opt-in Marian inference runtime for the *separate* ES/FR/EN baselines.

This research-only module is intentionally NOT imported by the current
default provider factory. It keeps precisely one CPU model loaded at a time,
pins all base files and adapter tensors to SHA256, and shares bounded decoder
safety across languages without blending incompatible tokenizer embeddings.

Do not interpret an adapter loaded into one runtime as one merged neural model.
"""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import ModelRegistry, ModelSpec, safe_local_path, file_sha256
from research.es_en_v2.inference_safety import (
    RETRY_GENERATION_OPTIONS, retry_if_loop, RepetitionLoopError,
)


@dataclass
class _Loaded:
    spec: ModelSpec
    key: str
    model: Any
    tokenizer: Any


class SharedMarianResearchTranslator:
    """Research-only dispatch by (source,target), no automatic ASR detection.

    Existing BridgeCaption Gemini/etc remains the app default; call this
    explicitly to try pinned local models. Context injection is disabled
    until a licensed, validated model is actually trained on previous turns.
    """
    supports_context = False

    def __init__(self, *, registry: ModelRegistry | None = None,
                 cache_dir: Path | None = None,
                 retry_repetition: bool = True):
        self.registry = registry if registry is not None else ModelRegistry()
        self.cache_dir = (Path(cache_dir) if cache_dir is not None
                          else self.registry.root / "research/unified_models/.base_cache")
        self.retry_repetition = bool(retry_repetition)
        self._lock = threading.RLock()
        self._loaded: _Loaded | None = None
        self.load_count = 0
        self.cache_hits = 0
        self.repetition_retries = 0
        self.last_run: dict | None = None

    @property
    def loaded_model_count(self) -> int:
        return int(self._loaded is not None)

    def routes(self) -> list[dict]:
        """Safe metadata; never emit transcript text or hidden final data."""
        return [
            {"direction": key, "base": spec.model,
             "revision": spec.revision,
             "adapter_sha256": spec.active_adapter["sha256"] if spec.active_adapter else None,
             "human_95_percent_verified": False,
             "model_ready_for_production": False,
             "supports_context": False,
             "license_status": spec.rights_review_status}
            for key, spec in sorted(self.registry.entries.items())
        ]

    def _pinned_files(self, spec: ModelSpec) -> dict[str, dict]:
        if spec.base_manifest is not None:
            records = json.loads(safe_local_path(
                self.registry.root, spec.base_manifest).read_text())
            files = {r["file"]: r for r in records}
        else:
            if spec.baseline_report is None:
                raise RuntimeError("Missing original Marian provenance")
            baseline = json.loads(safe_local_path(
                self.registry.root, spec.baseline_report).read_text())
            if baseline["model"] != spec.model or baseline["model_sha"] != spec.revision:
                raise RuntimeError("Original baseline SHA not pinned to model identity")
            files = {name: {"file": name, "sha256": sha}
                     for name, sha in baseline["model_file_sha256"].items()}
        required = {"config.json", "source.spm", "target.spm", "vocab.json"}
        required.add("model.safetensors" if spec.weights_safetensors else "pytorch_model.bin")
        if not required.issubset(files):
            raise RuntimeError("Pinned Marian base file/weight manifest is incomplete")
        for name, record in files.items():
            if not name or Path(name).name != name or any(c in name for c in ("/", "\\")):
                raise ValueError("Manifest attempted path traversal")
            sha = record["sha256"]
            if not isinstance(sha, str) or len(sha) != 64:
                raise ValueError("Missing SHA256 of frozen base file")
        return files

    def _download_base(self, spec: ModelSpec) -> Path:
        """Use immutable Hugging Face Git revision and verify EVERY file hash."""
        from huggingface_hub import hf_hub_download
        dest = self.cache_dir / spec.model_id / spec.revision
        files = self._pinned_files(spec)
        dest.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            raise RuntimeError("Symlinked model directory rejected")
        for name, row in files.items():
            path = dest / name
            if path.is_symlink():
                raise RuntimeError("Symlinked model file rejected")
            if path.exists() and file_sha256(path) == row["sha256"]:
                if "size_bytes" not in row or path.stat().st_size == row["size_bytes"]:
                    continue
            source = Path(hf_hub_download(
                repo_id=spec.model, filename=name, revision=spec.revision))
            if source.stat().st_size != row.get("size_bytes", source.stat().st_size):
                raise RuntimeError("Pinned base-model file size mismatch: " + name)
            if file_sha256(source) != row["sha256"]:
                raise RuntimeError("Pinned base-model SHA256 mismatch: " + name)
            shutil.copyfile(source, path)
            if file_sha256(path) != row["sha256"]:
                raise RuntimeError("Local cached base-model copy corrupted: " + name)
        return dest

    def _create(self, spec: ModelSpec, *, use_previous: bool) -> _Loaded:
        import torch
        from torch import nn
        from safetensors.torch import load_file
        from transformers import MarianMTModel, MarianTokenizer

        if use_previous and spec.rollback_adapter is None:
            raise ValueError("No previous trained adapter exists for " + spec.model_id)
        model_dir = self._download_base(spec)
        tokenizer = MarianTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = MarianMTModel.from_pretrained(
            str(model_dir), local_files_only=True,
            use_safetensors=spec.weights_safetensors, weights_only=True)
        for param in model.parameters():
            param.requires_grad_(False)

        adapter_spec = spec.adapter(use_previous=use_previous)
        if adapter_spec is not None:
            protocol = json.loads(safe_local_path(
                self.registry.root, spec.protocol).read_text())
            if (protocol["model"]["model"] != spec.model or
                protocol["model"]["revision"] != spec.revision):
                raise RuntimeError("Cannot use LoRA trained on different base model")
            targets = protocol["target_modules"]
            if len(set(targets)) != 36:
                raise RuntimeError("Expected pinned 36 attention-projection LoRA targets")

            class LoRA(nn.Module):
                def __init__(self, linear):
                    super().__init__()
                    self.base = linear
                    self.lora_A = nn.Parameter(torch.zeros(8, linear.in_features))
                    self.lora_B = nn.Parameter(torch.zeros(linear.out_features, 8))

                def forward(self, inputs):
                    return self.base(inputs) + ((
                        (inputs @ self.lora_A.T) @ self.lora_B.T) * 2.0)

            for name in targets:
                layer = model.get_submodule(name)
                if not isinstance(layer, nn.Linear):
                    raise RuntimeError("Wrong LoRA attention projection: " + name)
                prefix, key = name.rsplit(".", 1)
                setattr(model.get_submodule(prefix), key, LoRA(layer))
            path = safe_local_path(self.registry.root, adapter_spec["path"])
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("Pinned adapter is unavailable")
            if file_sha256(path) != adapter_spec["sha256"]:
                raise RuntimeError("Pinned adapter changed: " + spec.model_id)
            weights = load_file(str(path))
            state = model.state_dict()
            required = {key for key in state if ".lora_" in key}
            if set(weights) != required:
                raise RuntimeError("Adapter keys incompatible with pinned base")
            with torch.no_grad():
                for key, value in weights.items():
                    if value.shape != state[key].shape or not torch.isfinite(value).all():
                        raise RuntimeError("Wrong-shaped or nonfinite LoRA tensor")
                    state[key].copy_(value)
        model.eval()
        self.load_count += 1
        return _Loaded(spec=spec, key=spec.cache_key(use_previous=use_previous),
                       model=model, tokenizer=tokenizer)

    def _get(self, spec: ModelSpec, *, use_previous: bool) -> _Loaded:
        key = spec.cache_key(use_previous=use_previous)
        if self._loaded is not None and self._loaded.key == key:
            self.cache_hits += 1
            return self._loaded
        # Clear before constructing another model: never keep two 300MB
        # model weights resident in this resource-constrained process.
        self._loaded = None
        gc.collect()
        loaded = self._create(spec, use_previous=use_previous)
        self._loaded = loaded
        return loaded

    def unload(self) -> None:
        with self._lock:
            self._loaded = None
            gc.collect()

    def _generate(self, loaded: _Loaded, text: str,
                  *, override: dict | None = None) -> str:
        import torch
        encoded = loaded.tokenizer(
            text, return_tensors="pt", padding=False, truncation=False)
        if int(encoded["input_ids"].shape[-1]) > 128:
            # Avoid the old truncation that silently dropped the final clause.
            # Chunked/streaming translation must be validated separately.
            raise ValueError("Utterance exceeds the validated 128-token window; split at a sentence boundary")
        options = override if override is not None else {
            "num_beams": 4, "max_new_tokens": 96, "use_cache": True}
        with torch.no_grad():
            prediction = loaded.model.generate(**encoded, **options)
        return loaded.tokenizer.decode(prediction[0], skip_special_tokens=True).strip()

    def translate(self, request, *, use_previous: bool = False) -> str:
        """TranslationRequest-compatible, without changing app's default provider."""
        if getattr(request, "previous", ()) and getattr(request, "context_enabled", True):
            raise ValueError("These local checkpoints are NOT trained for live previous-turn context")
        text = getattr(request, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Current speech must be nonempty text")
        text = " ".join(text.split())
        if len(text) > 1500:
            raise ValueError("Utterance too long; never silently drop clauses")
        spec = self.registry.select(request.source, request.target)
        with self._lock:
            loaded = self._get(spec, use_previous=use_previous)
            initial = self._generate(loaded, text)
            if not initial:
                raise RuntimeError("Local Marian returned blank translation")
            if self.retry_repetition:
                try:
                    status = retry_if_loop(
                        initial,
                        lambda: self._generate(loaded, text, override=RETRY_GENERATION_OPTIONS))
                except RepetitionLoopError:
                    self.repetition_retries += 1
                    raise RuntimeError(
                        "Local Marian repeated output even after a bounded retry; "
                        "do not present degenerate captions as valid translations") from None
                if status.was_loop:
                    self.repetition_retries += 1
                final = status.text
            else:
                final = initial
            self.last_run = {
                "direction": spec.model_id,
                "version_sha256": spec.adapter(use_previous=use_previous)["sha256"]
                    if spec.adapter(use_previous=use_previous) else spec.revision,
                "rollback_selected": bool(use_previous),
                "generation_retry_for_loop": final != initial,
                "supports_previous_turn_context": False,
                "human_95_accuracy_verified": False,
                "default_application_provider_changed": False,
            }
            return final
