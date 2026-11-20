"""Opt-in local ES→EN integration smoke, NEVER the default BridgeCaption provider.

The current license-compatible v2 adapter was trained on isolated MASSIVE
utterances, not on reliable multi-turn, speaker-aware conversation data.
Consequently this translator correctly advertises supports_context=False
instead of pretending the research model can resolve slang from history.

This module intentionally lives in research/ and cannot change the working
app's default Gemini/provider path or desktop update flow.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments/cpu_continuation_20261117"))
sys.path.insert(0,str(ROOT/"src"))
import train as core
from bridgecaption.provider import TranslationRequest


class ExperimentalMarianTranslator:
    supports_context=False
    DIRECTION=("es","en")
    ADAPTER_REL="experiments/es_en_v2_runs/candidate_after_three.safetensors"
    ADAPTER_SHA="3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71"
    FALLBACK_REL="experiments/massive_stage3_20261117/saved/spanish/adapter_selected.safetensors"
    FALLBACK_SHA="14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b"

    def __init__(self, *, use_previous: bool=False, model_root:Path|None=None):
        self.model_root=model_root or ROOT/"research/es_en_v2/.local_integration_model"
        self._lock=threading.Lock()
        self._model=None
        self._tokenizer=None
        self.use_previous=use_previous
        self._adapter_rel=self.FALLBACK_REL if use_previous else self.ADAPTER_REL
        self.adapter_sha=self.FALLBACK_SHA if use_previous else self.ADAPTER_SHA
        self.loaded_provenance=None

    def _load_once(self):
        if self._model is not None:
            return
        import torch
        from torch import nn
        from safetensors.torch import load_file
        from transformers import MarianMTModel,MarianTokenizer

        adapter=ROOT/self._adapter_rel
        if core.sha256(adapter)!=self.adapter_sha:
            raise RuntimeError("Refusing to load missing or altered local research adapter")
        spec=core.LANGUAGES["spanish"]
        core.download_base(self.model_root,ROOT/"research"/spec["manifest_rel"],spec)
        model=MarianMTModel.from_pretrained(str(self.model_root),local_files_only=True,
                         use_safetensors=spec["use_safetensors"],weights_only=True)
        tokenizer=MarianTokenizer.from_pretrained(str(self.model_root),local_files_only=True)
        for p in model.parameters():
            p.requires_grad_(False)
        protocol=json.loads((ROOT/"research"/spec["protocol_rel"]).read_text())

        class LoRA(nn.Module):
            def __init__(self,base):
                super().__init__()
                self.base=base
                self.lora_A=nn.Parameter(torch.zeros(8,base.in_features))
                self.lora_B=nn.Parameter(torch.zeros(base.out_features,8))
            def forward(self,x):
                return self.base(x)+((x@self.lora_A.T)@self.lora_B.T)*2.0

        for name in protocol["target_modules"]:
            layer=model.get_submodule(name)
            if not isinstance(layer,nn.Linear):
                raise RuntimeError("Unexpected adapter layer")
            parent,key=name.rsplit(".",1)
            setattr(model.get_submodule(parent),key,LoRA(layer))
        state=model.state_dict()
        weights=load_file(str(adapter))
        expected={key for key in state if ".lora_" in key}
        if set(weights)!=expected:
            raise RuntimeError("Adapter incompatible with pinned base")
        with torch.no_grad():
            for key,tensor in weights.items():
                if tensor.shape!=state[key].shape or not torch.isfinite(tensor).all():
                    raise RuntimeError("Nonfinite or wrong shape LoRA weights")
                state[key].copy_(tensor)
        model.eval()
        self._tokenizer=tokenizer
        self._model=model
        self.loaded_provenance={
            "original_model":spec["model"],
            "revision":spec["revision"],
            "adapter_sha256":self.adapter_sha,
            "training_rights":"commercial_candidate_pending_legal_audit",
            "context_training_verified":False,
            "commercial_production_approved":False,
        }

    def translate(self,request:TranslationRequest) -> str:
        import torch
        if request.source not in {"es","es-ES"} or request.target!="en":
            raise ValueError("Experimental model supports Spanish to English only")
        if request.context_enabled and request.previous:
            raise ValueError("This checkpoint is not verified to use multi-turn context")
        if not request.text or not request.text.strip():
            raise ValueError("Blank request")
        with self._lock:
            self._load_once()
            assert self._model is not None and self._tokenizer is not None
            data=self._tokenizer(request.text[:1500],return_tensors="pt",
                                 truncation=True,max_length=128)
            with torch.no_grad():
                translated=self._model.generate(**data,num_beams=4,max_new_tokens=96)
            result=self._tokenizer.decode(translated[0],skip_special_tokens=True)
            if not result.strip():
                raise RuntimeError("Experimental local translator returned blank text")
            return result.strip()
