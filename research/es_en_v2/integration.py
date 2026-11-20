"""Backward-compatible Spanish research adapter backed by one shared loader.

This preserves the opt-in ExperimentalMarianTranslator interface used by the
previous Spanish real-Pipeline smoke and reviewer loop recovery scripts, while
consolidating all Marian loading/routing/checkpoint logic in
research.unified_models.runtime. The application's provider default is NOT
modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research.unified_models.runtime import SharedMarianResearchTranslator
from bridgecaption.provider import TranslationRequest


class ExperimentalMarianTranslator(SharedMarianResearchTranslator):
    """Drop-in wrapper for existing Spanish checkpoint evidence tooling."""
    supports_context = False
    DIRECTION = ("es", "en")
    ADAPTER_REL = "experiments/es_en_v2_runs/candidate_after_three.safetensors"
    ADAPTER_SHA = "3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71"
    FALLBACK_REL = "experiments/massive_stage3_20261117/saved/spanish/adapter_selected.safetensors"
    FALLBACK_SHA = "14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b"

    def __init__(self, *, use_previous: bool = False,
                 model_root: Path | None = None):
        self.use_previous = bool(use_previous)
        self.model_root = (Path(model_root) if model_root is not None
                           else ROOT / "research/es_en_v2/.local_integration_model")
        super().__init__(cache_dir=self.model_root)
        spec = self.registry.select("es", "en")
        self._adapter_rel = self.FALLBACK_REL if use_previous else self.ADAPTER_REL
        self.adapter_sha = self.FALLBACK_SHA if use_previous else self.ADAPTER_SHA
        candidate = spec.adapter(use_previous=use_previous)
        if candidate is None or candidate["sha256"] != self.adapter_sha or candidate["path"] != self._adapter_rel:
            raise RuntimeError("Spanish compatibility wrapper disagrees with pinned unified registry")
        self.loaded_provenance = None

    def _load_once(self):
        with self._lock:
            spec = self.registry.select("es", "en")
            self._get(spec, use_previous=self.use_previous)
            self.loaded_provenance = {
                "original_model": spec.model,
                "revision": spec.revision,
                "adapter_sha256": self.adapter_sha,
                "training_rights": spec.rights_review_status,
                "context_training_verified": False,
                "commercial_production_approved": False,
            }

    @property
    def _model(self):
        return self._loaded.model if self._loaded is not None else None

    @property
    def _tokenizer(self):
        return self._loaded.tokenizer if self._loaded is not None else None

    def translate(self, request: TranslationRequest) -> str:
        if (request.source not in {"es", "es-ES"} or request.target != "en"):
            raise ValueError("Legacy Spanish research translator supports Spanish to English only")
        result = super().translate(request, use_previous=self.use_previous)
        self._load_once()  # cache hit; populate the historical provenance surface.
        return result
