"""One local standalone M2M100 full-weight checkpoint, six translation routes.

The model file is assembled from the single learned adapter and exact pinned
multilingual pretrained base. No Marian routing, per-language model switch,
network request, or production provider override.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .model import ROOT, generate, route, sha256

RECEIPT=ROOT/"research/multilingual_student/experiment_001/standalone_export.json"


class OneWeightMultilingualResearchTranslator:
    supports_context=False

    def __init__(self,model_dir:Path,*,receipt_path:Path=RECEIPT):
        self.model_dir=Path(model_dir)
        self.receipt_path=Path(receipt_path)
        self._loaded=None
        self._loads=0
        self._lock=threading.RLock()
        self.last_run=None

    def status(self):
        return {
            "one_shared_model_weights":True,
            "loaded_model_count":int(self._loaded is not None),
            "model_loads":self._loads,
            "supports":["en-es","en-fr","es-en","es-fr","fr-en","fr-es"],
            "app_default_enabled":False,
            "human_95_percent_meaning_accuracy_verified":False,
            "full_audio_asr_model":False,
        }

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        if not self.model_dir.is_dir() or not self.receipt_path.is_file():
            raise FileNotFoundError("Standalone research checkpoint not prepared")
        info=json.loads(self.receipt_path.read_text())
        if (info.get("status")!="full_single_standalone_multilingual_neural_model_exported"
            or info.get("production_enabled") is not False
            or info.get("trained_student_meaning_95_percent_independently_verified") is not False):
            raise RuntimeError("Invalid standalone checkpoint provenance")
        for name,record in info["files"].items():
            if Path(name).name!=name or "\\" in name:
                raise ValueError("Standalone model manifest path rejected")
            file=self.model_dir/name
            if not file.is_file() or file.is_symlink():
                raise RuntimeError("Missing standalone full model file: "+name)
            if sha256(file)!=record["sha256"] or file.stat().st_size!=record["bytes"]:
                raise RuntimeError("Standalone model file was corrupted: "+name)
        from transformers import M2M100ForConditionalGeneration,M2M100Tokenizer
        tokenizer=M2M100Tokenizer.from_pretrained(str(self.model_dir),local_files_only=True)
        model=M2M100ForConditionalGeneration.from_pretrained(
            str(self.model_dir),local_files_only=True,weights_only=True,
            use_safetensors=True)
        if model.config.model_type!="m2m_100":
            raise RuntimeError("The research model is not the pinned shared multilingual net")
        if any(".lora_" in key for key in model.state_dict()):
            raise RuntimeError("Export accidentally still depends on an adapter")
        model.eval()
        self._loaded=(model,tokenizer)
        self._loads+=1
        return self._loaded

    def translate(self,request):
        src,tgt=route(request.source,request.target)
        if getattr(request,"context_enabled",False) and getattr(request,"previous",None):
            raise ValueError("No verified previous-turn conditioning has been trained")
        if not request.text or not request.text.strip():
            raise ValueError("Blank current spoken utterance")
        with self._lock:
            model,tok=self._load()
            translated=generate(model,tok,request.text,src,tgt)
            self.last_run={"source":src,"target":tgt,
                           "one_shared_network":True,
                           "model_switches":0,
                           "context_aware_claim":False,
                           "production_enabled":False}
            return translated
