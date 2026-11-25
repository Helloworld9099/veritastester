"""Opt-in 400+ language MADLAD research backend with ONE T5 base/LoRA.

MADLAD is the preferred new-language research architecture; the old M2M100
specialists remain production fallbacks. The adapter is only loaded if actual
GPU/CPU training completed with a hashed checkpoint and provenance receipt.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from research.global_languages.catalog import DocumentedCatalog
from research.global_languages.backend_registry import (
    BackendIdentity,WorldModelBackends
)
from .model import (
    EXPERIMENT,adapter_state,attach_shared_lora,config,
    file_sha256,language_input,load_backbone,supported_target_markers,
    translate_short
)


class MADLAD400ResearchBackend:
    def __init__(self,catalog:DocumentedCatalog,*,tokenizer=None,experiment=EXPERIMENT):
        self.catalog=catalog
        self.experiment=Path(experiment)
        cfg=config()["model"]
        self.identity=BackendIdentity(
            backend_id="shared_madlad400_3b",
            pretrained_model_id=cfg["repo_id"],
            pinned_revision=cfg["revision"],
            license=cfg["license"],modality="text",
            single_shared_base=True)
        self._loaded=None
        self._lock=threading.RLock()
        self.model_loads=0
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(
                cfg["repo_id"],revision=cfg["revision"],trust_remote_code=False)
        self.tokenizer=tokenizer
        self._codes=supported_target_markers(tokenizer)

    def native_codes(self)->frozenset[str]:
        return self._codes

    def supports_pair(self,source:str,target:str)->bool:
        # For new languages with BCP47 script/region distinctions, return
        # False until native variant handling is explicitly verified rather
        # than collapsing dialects to main language.
        from research.global_languages.catalog import parse_bcp47
        try:
            src=parse_bcp47(source)
            tgt=parse_bcp47(target)
        except ValueError:
            return False
        return (
            source==src.primary and target==tgt.primary
            and src.primary in self._codes and tgt.primary in self._codes
            and src.primary!=tgt.primary
            and self.catalog.cataloged(src.primary)
            and self.catalog.cataloged(tgt.primary)
        )

    def _get_trained_shared_model(self):
        if self._loaded is not None:
            return self._loaded
        receipt=self.experiment/"report.json"
        if not receipt.is_file():
            raise RuntimeError("MADLAD actual training is not yet verified; not a deployable model")
        report=json.loads(receipt.read_text())
        if (report.get("status")!="completed_real_MADLAD_3B_CPU_finetuning_pilot"
            or report.get("real_optimizer_updates")!=3
            or report.get("production_activated") is not False
            or report.get("human_95_meaning_accuracy_verified") is not False
            or report.get("model_commit_SHA")!=self.identity.pinned_revision):
            raise RuntimeError("Unverified or falsely certified 3B learned checkpoint")
        adapter=self.experiment/"madlad400_shared_lora_step03.safetensors"
        if file_sha256(adapter)!=report["trained_shared_model_adapter_sha256"]:
            raise RuntimeError("MADLAD adapter checksum mismatch")
        model,tok,markers=load_backbone(check_rights=True)
        meta=attach_shared_lora(
            model,rank=config()["pilot"]["LoRA"]["rank"],
            alpha=config()["pilot"]["LoRA"]["alpha"])
        if meta["trainable_parameters"]!=report["real_trainable_adapter"]["trainable_parameters"]:
            raise RuntimeError("MADLAD research model architecture changed")
        from safetensors.torch import load_file
        weights=load_file(str(adapter))
        parameters=dict(model.named_parameters())
        required={key for key in parameters if key.endswith((".lora_A",".lora_B"))}
        if set(weights)!=required:
            raise RuntimeError("Saved adapter does not match selected 3B model")
        import torch
        with torch.no_grad():
            for key,tensor in weights.items():
                if tensor.shape!=parameters[key].shape or not torch.isfinite(tensor).all():
                    raise RuntimeError("Corrupted MADLAD learned tensor: "+key)
                parameters[key].copy_(tensor)
        model.eval()
        self._loaded=(model,tok,markers)
        self.model_loads+=1
        return self._loaded

    def translate_research(self,text:str,source:str,target:str)->str:
        if not self.supports_pair(source,target):
            raise ValueError("Unknown/native-unverified MADLAD pair; no silent fallback")
        with self._lock:
            model,tok,codes=self._get_trained_shared_model()
            return translate_short(model,tok,text,source,target,codes)["text"]

    def experimental_status(self)->dict:
        return {
            "selected_research_backbone":self.identity.pretrained_model_id,
            "one_shared_neural_checkpoint":True,
            "native_target_codes":len(self._codes),
            "actually_loaded_model_count":int(self._loaded is not None),
            "model_loads":self.model_loads,
            "production_enabled":False,
            "independently_verified_meaning_accuracy":False,
            "Whisper_speech_recognition_replaced":False,
        }


def create_new_preferred_world_backend(*,catalog=None):
    catalog=catalog or DocumentedCatalog.load()
    world=WorldModelBackends(catalog)
    world.add(MADLAD400ResearchBackend(catalog))
    return world
