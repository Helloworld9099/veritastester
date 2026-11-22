"""Opt-in globally expandable research translator with strict capability checks.

One M2M100 base and one saved six-direction shared student adapter; potentially
~100 native tokenizer languages, but ONLY en/es/fr had additional BridgeCaption
training. No coverage/human accuracy claims for other languages. Unknown
languages or untrained tokenizers raise typed errors, never translate as
a different language or silently fall back to a cloud API.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .catalog import DocumentedCatalog
from .capabilities import GlobalLanguageRouter,UnsupportedLanguageRoute
from research.multilingual_student.model import (
    ROOT,read_config,load_shared_model,attach_shared_lora,load_adapter,sha256
)

SELECTED=ROOT/"research/multilingual_student/experiment_002/checkpoint_selection.json"


class ExpandableResearchTranslator:
    supports_context=False

    def __init__(self,*,catalog:DocumentedCatalog|None=None,
                 selection_report:Path=SELECTED):
        self.catalog=catalog
        self.selection_report=Path(selection_report)
        self._lock=threading.RLock()
        self._loaded=None
        self.model_loads=0
        self.last_run=None

    def _prepare(self):
        if self._loaded is not None:
            return self._loaded
        if self.catalog is None:
            self.catalog=DocumentedCatalog.load()
        record=json.loads(self.selection_report.read_text())
        if (record["status"]!="actual_saved_shared_multilingual_checkpoint_sweep_completed"
            or record["human_95_percent_meaning_accuracy_verified"] is not False
            or record["production_activated"] is not False):
            raise RuntimeError("Invalid or falsely certified shared-model selection")
        relative=Path(record["selected_research_checkpoint_relative"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe checkpoint path")
        expected=record["selected_research_sha256"]
        checkpoint=ROOT/relative
        if sha256(checkpoint)!=expected:
            raise RuntimeError("Saved shared model checkpoint changed")
        model,tok=load_shared_model()
        cfg=read_config()["training"]
        attach_shared_lora(model,rank=cfg["lora_rank"],alpha=cfg["lora_alpha"],
                           encoder_last=cfg["last_encoder_layers"],
                           decoder_last=cfg["last_decoder_layers"])
        load_adapter(model,checkpoint,sha=expected)
        model.eval()
        selector=GlobalLanguageRouter(self.catalog,tok)
        self._loaded=(model,tok,selector,expected)
        self.model_loads+=1
        return self._loaded

    def status(self)->dict:
        with self._lock:
            if self._loaded is None:
                return {"loaded":False,"one_shared_weight_set":True,
                        "production_activated":False,"human_adequacy_verified":False}
            _,_,router,sha=self._loaded
            return {"loaded":True,"one_shared_weight_set":True,
                    "model_loads":self.model_loads,"selected_shared_adapter_sha256":sha,
                    "capabilities":router.coverage_stats(),
                    "production_activated":False,"human_adequacy_verified":False}

    def inspect(self,source:str,target:str):
        with self._lock:
            return self._prepare()[2].inspect(source,target)

    def translate(self,request):
        import torch
        if getattr(request,"context_enabled",False) and getattr(request,"previous",None):
            raise ValueError("The shared model has not passed past-turn-context training/QA")
        if not isinstance(request.text,str) or not request.text.strip():
            raise ValueError("Current utterance is required")
        with self._lock:
            model,tok,router,checkpoint=self._prepare()
            src,tgt=router.require_research_route(request.source,request.target)
            # Region/script identification remains in the request and is
            # flagged as unverified. The tokenizer only accepts base codes.
            tok.src_lang=src
            encoded=tok(request.text.strip(),return_tensors="pt",truncation=False)
            if int(encoded["input_ids"].shape[-1])>72:
                raise ValueError("Exceeded the validated short-utterance token window")
            with torch.inference_mode():
                generated=model.generate(
                    **encoded,forced_bos_token_id=tok.get_lang_id(tgt),
                    num_beams=2,max_new_tokens=72,use_cache=True)
            translated=tok.decode(generated[0],skip_special_tokens=True).strip()
            if not translated:
                raise RuntimeError("Single model emitted blank translation")
            capability=router.inspect(request.source,request.target)
            self.last_run={
                "model_id":"facebook/m2m100_418M",
                "selected_shared_adapter_sha256":checkpoint,
                "source":request.source,"target":request.target,
                "capability":capability.model_coverage,
                "production_approved":False,
                "human_accuracy_verified":False,
                "trained_context_support":False,
            }
            return translated

    def unload(self):
        with self._lock:
            self._loaded=None
