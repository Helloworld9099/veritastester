"""Historical route-specific Marian baselines for pinned, explicit offline replay only.

This research-only module is intentionally NOT imported by the current
default provider factory. It keeps precisely one CPU model loaded at a time,
pins all base files and adapter tensors to SHA256, and shares bounded decoder
safety across languages without blending incompatible tokenizer embeddings.

Do not interpret an adapter loaded into one runtime as one merged neural model.
"""
from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import ModelRegistry, ModelSpec
from ..inference_safety import (
    RETRY_GENERATION_OPTIONS, retry_if_loop, RepetitionLoopError,
)


@dataclass
class _Loaded:
    spec: ModelSpec
    key: str
    model: Any
    tokenizer: Any


class HistoricalMarianBaseline:
    """Archived specialist replay by (source,target), never an operational translator.

    Existing BridgeCaption Gemini/etc remains the app default; call this
    explicitly to try pinned local models. Context injection is disabled
    until a licensed, validated model is actually trained on previous turns.
    """
    supports_context = False

    def __init__(self, *, registry: ModelRegistry | None = None,
                 cache_dir: Path | None = None,
                 artifact_root: Path | None = None,
                 retry_repetition: bool = True, historical_replay: bool = False):
        if historical_replay is not True:
            raise ValueError("Archived specialist baselines may only be loaded for explicit offline replay")
        self.registry = registry if registry is not None else ModelRegistry()
        self.cache_dir = (Path(cache_dir) if cache_dir is not None
                          else self.registry.root / "research/unified_models/.base_cache")
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
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
        return {name: dict(row) for name, row in spec.files.items()}

    def _create(self, spec: ModelSpec, *, use_previous: bool) -> _Loaded:
        # Preflight before importing inference dependencies or loading a model.
        # Local artifacts only. Explicitly no network, fetch, or backup provider.
        adapter_bytes = self.registry.verify_adapter(
            spec, self.artifact_root, use_previous=use_previous)
        model_dir = self.registry.verify_base(spec, self.cache_dir)

        import torch
        from torch import nn
        from safetensors.torch import load
        from transformers import MarianMTModel, MarianTokenizer

        tokenizer = MarianTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = MarianMTModel.from_pretrained(
            str(model_dir), local_files_only=True,
            use_safetensors=spec.weights_safetensors, weights_only=True)
        for param in model.parameters():
            param.requires_grad_(False)

        if adapter_bytes is not None:
            protocol = spec.protocol
            rank = protocol["rank"]
            factor = protocol["alpha"] / rank
            weights = load(adapter_bytes)
            targets = protocol["target_modules"]

            class LoRA(nn.Module):
                def __init__(self, linear):
                    super().__init__()
                    self.base = linear
                    self.lora_A = nn.Parameter(torch.zeros(rank, linear.in_features))
                    self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))

                def forward(self, inputs):
                    return self.base(inputs) + ((
                        (inputs @ self.lora_A.T) @ self.lora_B.T) * factor)

            for name in targets:
                layer = model.get_submodule(name)
                if not isinstance(layer, nn.Linear):
                    raise RuntimeError("Wrong LoRA attention projection")
                if (tuple(weights[name + ".lora_A"].shape) != (rank, layer.in_features)
                        or tuple(weights[name + ".lora_B"].shape) != (layer.out_features, rank)):
                    raise RuntimeError("Adapter incompatible with loaded Marian projection")
            for name in targets:
                prefix, key = name.rsplit(".", 1)
                setattr(model.get_submodule(prefix), key, LoRA(model.get_submodule(name)))
            state = model.state_dict()
            with torch.no_grad():
                for key, value in weights.items():
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
            self.last_run = None
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
        if getattr(request, "context_enabled", True):
            for attr, label in (("previous", "live previous-turn context"),
                                ("glossary", "a glossary"), ("sense_hints", "learned sense hints")):
                if getattr(request, attr, None):
                    raise ValueError(f"These local checkpoints do not support {label}")
        text = getattr(request, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Current speech must be nonempty text")
        text = " ".join(text.split())
        if len(text) > 1500:
            raise ValueError("Utterance too long; never silently drop clauses")
        spec = self.registry.select(request.source, request.target)
        with self._lock:
            self.last_run = None
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
