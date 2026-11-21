"""Opt-in BridgeCaption text Pipeline adapter for ONE real multilingual model.

A single resident M2M100 network and one shared learned adapter support
ES/EN/FR in both directions. No per-language model selection, unloading or
reloading. Does not alter application provider defaults, ASR, or auto-updates.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .model import (
    ROOT, attach_shared_lora, generate, load_shared_model,
    load_adapter, read_config, route, sha256
)

EXPERIMENT=ROOT/"research/multilingual_student/experiment_001"
SUPPORTED_DIRECTIONS=frozenset(read_config()["directions"])


class SingleMultilingualTextTranslator:
    supports_context = False

    def __init__(self, *, experiment: Path = EXPERIMENT):
        self.experiment=Path(experiment)
        self._lock=threading.RLock()
        self._loaded=None
        self._load_count=0
        self.last_run=None

    @property
    def loaded_model_count(self):
        return int(self._loaded is not None)

    @property
    def model_load_count(self):
        return self._load_count

    def status(self):
        return {
            "architecture":"one_M2M100_network",
            "shared_adapter":True,
            "supported_directions":sorted(SUPPORTED_DIRECTIONS),
            "loaded_model_count":self.loaded_model_count,
            "model_load_count":self.model_load_count,
            "production_activated":False,
            "independently_verified_95_meaning_accuracy":False,
            "speech_recognition_integrated":False,
        }

    def _load_once(self):
        if self._loaded is not None:
            return self._loaded
        if not self.experiment.is_dir():
            raise RuntimeError("Single model not yet trained; no fallback to unverified source")
        report=json.loads((self.experiment/"report.json").read_text())
        candidate=self.experiment/f"shared_lora_step{report['optimizer_updates_completed']:02d}.safetensors"
        if (report.get("status") != "completed_actual_single_multilingual_student_training"
            or report["production_activation_authorized"] is not False
            or report["independently_verified_95_percent_meaning_accuracy"] is not False):
            raise RuntimeError("Unexpected or falsely promoted research checkpoint")
        if (not report.get("single_shared_weight_set") or
            set(report.get("trained_directions",[]))!=SUPPORTED_DIRECTIONS):
            raise RuntimeError("Saved model does not cover all six required directions")
        if sha256(candidate)!=report["checkpoint_sha256"]:
            raise RuntimeError("Shared checkpoint SHA mismatch")
        model,tokenizer=load_shared_model()
        cfg=read_config()["training"]
        arch=attach_shared_lora(model,rank=cfg["lora_rank"],alpha=cfg["lora_alpha"],
                                encoder_last=cfg["last_encoder_layers"],
                                decoder_last=cfg["last_decoder_layers"])
        if arch["trainable_parameters"]!=report["lora_architecture"]["trainable_parameters"]:
            raise RuntimeError("Checkpoint and model architecture mismatch")
        load_adapter(model,candidate,sha=report["checkpoint_sha256"])
        model.eval()
        self._loaded=(model,tokenizer)
        self._load_count+=1
        return self._loaded

    def unload(self):
        with self._lock:
            self._loaded=None

    def translate(self,request):
        if getattr(request,"context_enabled",False) and getattr(request,"previous",None):
            raise ValueError("This student has NOT learned safe previous-turn context")
        src,tgt=route(request.source,request.target)
        text=request.text
        with self._lock:
            model,tokenizer=self._load_once()
            result=generate(model,tokenizer,text,src,tgt)
            report=json.loads((self.experiment/"report.json").read_text())
            self.last_run={
                "model_id":"facebook/m2m100_418M",
                "source":src,"target":tgt,
                "same_checkpoint_for_all_languages":True,
                "checkpoint_sha256":report["checkpoint_sha256"],
                "human_meaning_accuracy_verified":False,
            }
            return result


def translation_compatible_with_bridgecaption():
    """Runtime integration is opt-in only; no global provider list changes."""
    from bridgecaption.provider import TranslationRequest
    from bridgecaption.pipeline import Pipeline
    translator=SingleMultilingualTextTranslator()
    return Pipeline(translator=translator,context_mode="all")
