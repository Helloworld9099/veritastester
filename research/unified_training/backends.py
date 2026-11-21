"""Static, read-only adapters for known legacy research training backends.

The existing CPU experiment is a real trainer, but its public entrypoint
bundles dataset downloads, training and development inference. Neither the
entrypoint nor any dependency is imported or executed here. This module
describes its interface and checks repo-backed capabilities, not readiness to
run arbitrary data, a launch command, or a production authorization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import TrainingConfig

_LEGACY_SCRIPT = "experiments/cpu_continuation_20261117/train.py"
_LEGACY_SCRIPT_SHA256 = "eda50f249671a2d83a86fccf056a431765fbbe991f84154aa929483b8bc32689"


@dataclass(frozen=True)
class CapabilityIssue:
    code: str
    field: str


@dataclass(frozen=True)
class BackendCapabilities:
    directions: tuple[str, ...]
    architectures: tuple[str, ...]
    devices: tuple[str, ...]
    precisions: tuple[str, ...]
    strategies: tuple[str, ...]
    historical_start_modes: tuple[str, ...]
    offline_execution_supported: bool
    exact_resume_supported: bool
    execution_supported: bool

    def to_mapping(self) -> dict:
        return {
            "directions": list(self.directions), "architectures": list(self.architectures),
            "devices": list(self.devices), "precisions": list(self.precisions),
            "strategies": list(self.strategies),
            "historical_start_modes": list(self.historical_start_modes),
            "offline_execution_supported": self.offline_execution_supported,
            "exact_resume_supported": self.exact_resume_supported,
            "execution_supported": self.execution_supported,
        }


@dataclass(frozen=True)
class BackendPlan:
    backend_id: str
    backend_version: str
    status: str
    execution_supported: bool
    entrypoint: str
    entrypoint_sha256: str
    legacy_arguments: tuple[tuple[str, str], ...]
    capabilities: BackendCapabilities
    issues: tuple[CapabilityIssue, ...]
    observed_recipe: tuple[tuple[str, str | int | float | bool], ...]

    def to_mapping(self) -> dict:
        """Public-safe audit receipt. Paths and arguments are not an argv."""
        return {
            "backend_id": self.backend_id, "backend_version": self.backend_version,
            "status": self.status, "execution_supported": self.execution_supported,
            "entrypoint": self.entrypoint, "entrypoint_sha256": self.entrypoint_sha256,
            "legacy_arguments": dict(self.legacy_arguments),
            "capabilities": self.capabilities.to_mapping(),
            "issues": [{"code": i.code, "field": i.field} for i in self.issues],
            "observed_recipe": dict(self.observed_recipe),
        }


class PlanningBackend(Protocol):
    id: str
    version: str
    capabilities: BackendCapabilities

    def plan(self, config: TrainingConfig, *, repo_root: Path) -> BackendPlan: ...


class UnsupportedBackend(ValueError):
    """Unknown static backend identity; never dynamically look it up."""


class BackendSourceError(ValueError):
    """A pinned repository source needed to describe a backend is invalid."""


@dataclass(frozen=True)
class _LegacyLanguage:
    cli_language: str
    base_model: str
    base_revision: str
    base_manifest: str
    base_manifest_sha256: str
    protocol_sha256: str
    historical_adapter_sha256: str


_ES = _LegacyLanguage(
    "spanish", "Helsinki-NLP/opus-mt-es-en", "c96e2c5399ebfae4fc43d9669556b9afa74bb69d",
    "research/translation_training_20261116/model/download_manifest.json",
    "f65878b125b933bce9847597533a6d05475941e8708cc27b1e2cbddafbf4a355",
    "d42b1b2685ba05b7413e720cd8b0d6c1d267be37976c47835daf6c3a0a4c1b93",
    "55d4b872a6807e0d9bc317b72ef63eddf77d5828c427f91497d5bc23ac2e65f3",
)
_FR = _LegacyLanguage(
    "french", "Helsinki-NLP/opus-mt-fr-en", "c4aed37b318c763fd177aa449b44e3b783cc6c02",
    "research/translation_fr_training_20261116/model/download_manifest.json",
    "b0ae20ad81357178b6924a11ac3e68ec2953017e7bb81ebef15a293309b81220",
    "a6e656326bf6b1b880bbde74b6728024d1ad2b543518bd63d2139718b051772a",
    "600990e205c1238ea1fba3a8c36e5ef199bdfc10b33e4e0068e5b9695b7e8665",
)


def _verify_repo_file(repo_root: Path, relative: str, expected: str, limit: int) -> bytes:
    # Every checked filename is an internal constant or fixed EN direction
    # binding, not an artifact locator or path obtained from user configuration.
    path = repo_root.resolve()
    try:
        for part in relative.split("/"):
            path = path / part
            if path.is_symlink():
                raise BackendSourceError("legacy_source_symlink_rejected")
        if not path.is_file():
            raise BackendSourceError("legacy_source_unavailable")
        with path.open("rb") as file:
            raw = file.read(limit + 1)
    except OSError:
        raise BackendSourceError("legacy_source_unavailable") from None
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != expected:
        raise BackendSourceError("legacy_source_pin_mismatch")
    return raw


def _legacy_pins(root: Path, route: _LegacyLanguage) -> tuple[dict[str, str], tuple[str, ...]]:
    base = _verify_repo_file(root, route.base_manifest, route.base_manifest_sha256, 65536)
    prior = f"research/neural_continuation_20261116/{route.cli_language}"
    protocol = _verify_repo_file(root, f"{prior}/protocol.json", route.protocol_sha256, 65536)
    # Verifies bytes only; never interprets the historical adapter or opens base weights.
    _verify_repo_file(root, f"{prior}/adapter_selected.safetensors",
                      route.historical_adapter_sha256, 2 * 1024 * 1024)
    records = json.loads(base)
    spec = json.loads(protocol)
    files = {row["file"]: row["sha256"] for row in records}
    if len(files) != len(records) or spec["model"]["model"] != route.base_model or spec["model"]["revision"] != route.base_revision:
        raise BackendSourceError("legacy_source_binding_mismatch")
    return files, tuple(spec["target_modules"])


# These are static facts of train.py at the pinned digest, not knobs that the
# script accepts. It relies on torch AdamW's default betas and epsilon.
_SCHEDULE = (
    ("kind", "constant"), ("total_updates", 96), ("warmup_updates", 0),
    ("micro_batch_size", 4), ("gradient_accumulation_steps", 4),
    ("max_source_tokens", 96), ("max_target_tokens", 96),
)
_OPTIMIZER = (
    ("name", "adamw"), ("learning_rate", 2.5e-5), ("weight_decay", 0.01),
    ("beta1", 0.9), ("beta2", 0.999), ("epsilon", 1e-8),
)
_RECIPE = (
    ("seed", 20261117), ("cpu_threads", 4), ("interop_threads", 1),
    ("precision", "fp32"), ("lora_rank", 8), ("lora_alpha", 16),
    ("gradient_clip", 1), ("effective_batch", 16), ("train_rows", 1536),
    ("dev_rows", 128), ("checkpoint_interval_updates", 48),
    ("checkpoint_component", "adapter_weights_only"),
    ("dataset_input_mode", "legacy_network_download_then_internal_split"),
    ("base_model_input_mode", "legacy_network_download_of_pinned_revision"),
    ("offline_run", False),
    *((f"schedule_{key}", value) for key, value in _SCHEDULE),
    *((f"optimizer_{key}", value) for key, value in _OPTIMIZER),
)


class LegacyMarianPlanner:
    """Describe the preserved main-branch trainer without importing or calling it."""

    id = "marian_hf"
    version = "1.0.0"
    capabilities = BackendCapabilities(
        directions=("es->en", "fr->en"), architectures=("marian",), devices=("cpu",),
        precisions=("fp32",), strategies=("lora",), historical_start_modes=("warm_start",),
        offline_execution_supported=False, exact_resume_supported=False, execution_supported=False,
    )

    def plan(self, config: TrainingConfig, *, repo_root: Path) -> BackendPlan:
        if config.backend["id"] != self.id or config.backend["version"] != self.version:
            raise UnsupportedBackend("unsupported_backend_identity")
        _verify_repo_file(Path(repo_root), _LEGACY_SCRIPT, _LEGACY_SCRIPT_SHA256, 128 * 1024)
        issues: list[CapabilityIssue] = []
        route = {"es->en": _ES, "fr->en": _FR}.get(config.direction)
        args: tuple[tuple[str, str], ...] = ()

        def add(unmet: bool, code: str, field: str) -> None:
            if unmet:
                issues.append(CapabilityIssue(code, field))

        add(route is None, "unsupported_direction", "direction")
        add(config.architecture != "marian", "unsupported_architecture", "architecture")
        add(config.device["kind"] != "cpu" or config.device["gpu_opt_in"],
            "gpu_not_supported", "device.kind")
        add(config.device["precision"] != "fp32", "unsupported_precision", "device.precision")
        add(config.device["threads"] != 4, "legacy_fixed_cpu_threads", "device.threads")
        add(config.strategy["kind"] != "lora", "unsupported_strategy", "strategy.kind")
        if config.strategy["kind"] == "lora":
            add(config.strategy["rank"] != 8 or config.strategy["alpha"] != 16,
                "legacy_fixed_lora_geometry", "strategy")
        add(config.seed != 20261117, "legacy_fixed_seed", "seed")
        for key, actual in _SCHEDULE:
            add(config.schedule[key] != actual, "legacy_fixed_schedule_value", f"schedule.{key}")
        for key, actual in _OPTIMIZER:
            add(config.optimizer[key] != actual, "legacy_fixed_optimizer_value", f"optimizer.{key}")

        mode = config.source["mode"]
        add(mode == "exact_resume", "exact_resume_unsupported_no_state", "source.mode")
        add(mode == "base", "legacy_requires_historical_warm_weights", "source.mode")
        if route is not None:
            file_pins, targets = _legacy_pins(Path(repo_root), route)
            add((config.base["model_id"] != route.base_model or config.base["revision"] != route.base_revision
                 or dict(config.base["files"]) != file_pins), "legacy_fixed_base_pins", "base")
            add(config.strategy["protocol_sha256"] != route.protocol_sha256
                or tuple(config.strategy["target_modules"]) != targets,
                "legacy_fixed_adapter_protocol", "strategy")
            if mode == "warm_start":
                add(config.source["weights_sha256"] != route.historical_adapter_sha256,
                    "legacy_cannot_select_custom_warm_weights", "source.weights_sha256")
            args = (("--language", route.cli_language),
                    ("--repo-root", "<legacy-checkout-path>"),
                    ("--output-dir", "<new-legacy-output-directory>"))
        status = "blocked" if issues else "inspection_only"
        # These are unconditional execution blockers (even if the requested
        # recipe is historically identical). No fallback, network, or bypass.
        issues.extend((
            CapabilityIssue("legacy_cannot_bind_pinned_local_dataset", "data"),
            CapabilityIssue("network_fetch_required_by_legacy_call", "execution"),
            CapabilityIssue("no_execution_route_in_m3", "execution"),
        ))
        return BackendPlan(self.id, self.version, status, False, _LEGACY_SCRIPT,
                           _LEGACY_SCRIPT_SHA256, args, self.capabilities, tuple(issues), _RECIPE)


def backend_for(backend_id: str, backend_version: str) -> PlanningBackend:
    """Resolve only a known in-tree adapter; configuration is never import code."""
    if backend_id == LegacyMarianPlanner.id and backend_version == LegacyMarianPlanner.version:
        return LegacyMarianPlanner()
    raise UnsupportedBackend("unsupported_backend_identity")


def plan_backend(config: TrainingConfig, *, repo_root: Path) -> BackendPlan:
    return backend_for(config.backend["id"], config.backend["version"]).plan(config, repo_root=repo_root)
