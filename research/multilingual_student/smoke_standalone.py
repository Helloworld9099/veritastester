#!/usr/bin/env python3
"""Real app Pipeline smoke using the exported ONE full multilingual file."""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from bridgecaption.pipeline import Pipeline
from bridgecaption.providers import PROVIDERS
from bridgecaption.provider import TranslationRequest
from research.multilingual_student.standalone_runtime import (
    OneWeightMultilingualResearchTranslator
)

MODEL=Path(os.environ["RUNNER_TEMP"])/"bridgecaption-standalone-trilingual"
RECEIPT=ROOT/"research/multilingual_student/experiment_001/standalone_runtime_smoke.json"

CASE={
    "es-en":"Hola, ¿puedes ayudarme con la reserva?",
    "fr-en":"Bonjour, pouvez-vous m'aider avec la réservation ?",
    "en-es":"Hello, I need your help with my booking.",
    "en-fr":"Hello, I need your help with my booking.",
    "es-fr":"¿Puedes llegar antes de las ocho?",
    "fr-es":"Je voudrais réserver une table pour demain.",
}


def run():
    import torch
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    if RECEIPT.exists():
        raise FileExistsError("Do not overwrite a completed inference smoke")
    if "single_m2m100" in PROVIDERS:
        raise RuntimeError("Multilingual experiment was unexpectedly activated for users")
    started=time.monotonic()
    engine=OneWeightMultilingualResearchTranslator(MODEL)
    out={}
    for route,text in CASE.items():
        src,tgt=route.split("-")
        begin=time.monotonic()
        translated=engine.translate(TranslationRequest(text,source=src,target=tgt))
        if not translated.strip() or engine.status()["loaded_model_count"]!=1:
            raise RuntimeError("Standalone model dropped translation or switched the network")
        out[route]={"translation":translated,"seconds":time.monotonic()-begin}
    if engine.status()["model_loads"]!=1:
        raise RuntimeError("A single model must not reload when languages switch")
    pipe=Pipeline(translator=engine,context_mode="all")
    for src,tgt,text in (("es","en","Necesito revisar mi horario."),
                         ("fr","en","Je n'ai toujours pas reçu le billet."),
                         ("en","es","Please repeat the last sentence.")):
        caption=pipe.translate(text,src,tgt)
        if not caption.translated.strip() or "disabled" not in caption.context_info:
            raise RuntimeError("App Pipeline incorrectly claims trained context support")
    if pipe.history.snapshot():
        raise RuntimeError("No previous dialogue should be stored in this context-free model")
    if engine.status()["model_loads"]!=1:
        raise RuntimeError("Pipeline routing unexpectedly loaded another weight copy")
    for src,tgt in (("auto","en"),("en","en"),("de","fr")):
        try:
            engine.translate(TranslationRequest("Hi",source=src,target=tgt))
        except ValueError:
            pass
        else:
            raise RuntimeError("Unsupported language direction silently allowed")
    report={
        "status":"real_one_full_weight_model_pipeline_smoke_passed",
        "total_model_loads":engine.status()["model_loads"],
        "same_network_kept_through_all_six_directions":True,
        "directions":list(CASE),
        "real_model_text_outputs":out,
        "additional_real_BridgeCaption_Pipeline_cases":3,
        "max_memory_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds":time.monotonic()-started,
        "no_production_app_changes":True,
        "no_substitute_ASR_or_microphone_added":True,
        "no_independent_95_percent_adequacy_claim":True,
        "test_holdout_never_accessed":True,
    }
    RECEIPT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("REAL_ONE_STANDALONE_MODEL_APP_PIPELINE_SMOKE",json.dumps({
        "status":report["status"],"total_model_loads":1,"routes":len(out),
        "max_rss_kib":report["max_memory_kib_linux"],"elapsed":report["elapsed_seconds"]
    }),flush=True)


if __name__=="__main__":
    run()
