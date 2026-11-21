"""Transitional text-only research adapter; not the future streaming speech API."""
from __future__ import annotations

from pathlib import Path
import threading
from typing import Callable, Protocol

from .registry import ModelRegistry, SingleModelSpec


class ModelSession(Protocol):
    """A reviewed loader must return one already-loaded checkpoint session."""

    def translate(self, text: str, *, source: str, target: str) -> str:
        """Supply input and language controls to the same model instance."""


class SingleMultilingualResearchTranslator:
    """One immutable candidate for this runtime's lifetime.

    This is a disabled-by-default text harness, not a speech architecture.
    No default loader or model is included. Only an explicitly supplied and
    reviewed Python loader may execute. Deployers must use local-only loading
    with remote code disabled in an isolated process; this is not an OS sandbox.
    """

    supports_context = False

    def __init__(self, *, registry: ModelRegistry | None = None,
                 checkpoint_root: Path | None = None,
                 load_model: Callable[[SingleModelSpec, Path], ModelSession] | None = None):
        self._registry = registry if registry is not None else ModelRegistry()
        self._checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else None
        self._load_model = load_model
        self._session: ModelSession | None = None
        self._lock = threading.RLock()
        self._load_attempted = False

    @property
    def loaded_model_count(self) -> int:
        return int(self._session is not None)

    def status(self) -> dict:
        spec = self._registry.candidate
        return {
            "architecture": "single_multilingual_model",
            "interface": "transitional_text_research_only",
            "candidate_configured": spec is not None,
            "checkpoint_sha256": spec.checkpoint_sha256 if spec is not None else None,
            "target_directions": sorted(self._registry.target_directions),
            "verified_directions": sorted(spec.verified_directions) if spec is not None else [],
            "production_activated": False,
        }

    def _get(self) -> ModelSession:
        if self._session is not None:
            return self._session
        spec = self._registry.candidate
        if (spec is None or self._checkpoint_root is None or not callable(self._load_model)):
            raise ValueError("No verified local multilingual checkpoint and reviewed loader are configured")
        if self._load_attempted:
            raise ValueError("Checkpoint loader already attempted; no fallback model is available")
        self._load_attempted = True
        location = self._registry.verify_checkpoint(self._checkpoint_root)
        try:
            session = self._load_model(spec, location)
        except Exception:
            raise ValueError("The single multilingual checkpoint could not be loaded") from None
        if not callable(getattr(session, "translate", None)):
            raise ValueError("Invalid session returned for the multilingual checkpoint")
        self._session = session
        return session

    def translate(self, request) -> str:
        source, target = self._registry.require_direction(request.source, request.target)
        if getattr(request, "context_enabled", True) and any(
            getattr(request, key, None) for key in ("previous", "glossary", "sense_hints")
        ):
            raise ValueError("Context modifiers are not enabled in this research contract")
        text = getattr(request, "text", None)
        if not isinstance(text, str) or not text.strip() or "\0" in text:
            raise ValueError("Expected nonempty translation text")
        if len(text) > 1500 or any(0xD800 <= ord(c) <= 0xDFFF for c in text):
            raise ValueError("Translation text is outside the supported input bounds")
        # A loader takes no direction. Every call, in every direction and even
        # across concurrent requests, uses the same loaded session object.
        with self._lock:
            session = self._get()
            try:
                output = session.translate(text, source=source, target=target)
            except Exception:
                raise ValueError("The single multilingual session could not translate") from None
            if (not isinstance(output, str) or not output.strip() or "\0" in output
                    or any(0xD800 <= ord(c) <= 0xDFFF for c in output)):
                raise ValueError("Invalid translation; no fallback model is available")
            return output
