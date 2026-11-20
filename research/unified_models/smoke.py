#!/usr/bin/env python3
"""Run the REAL pinned Spanish/French Marian through BridgeCaption's Pipeline.

No human-accuracy inference, no final holdout, no default provider changes.
Reference comparison is to the already-consumed 500-example development export:
it establishes decoding compatibility, NOT newly independent accuracy.
"""
from __future__ import annotations

import csv
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from bridgecaption.pipeline import Pipeline
from bridgecaption.provider import TranslationRequest
from bridgecaption.providers import create_translator, PROVIDERS
from research.unified_models.registry import ModelRegistry
from research.unified_models.runtime import SharedMarianResearchTranslator
from research.es_en_v2.review_feedback_500 import load_review

OUT = ROOT / "research/unified_models/receipts"


def translate_and_record(engine, request):
    began = time.perf_counter()
    output = engine.translate(request)
    duration = time.perf_counter() - began
    if not output.strip():
        raise RuntimeError("Unified model generated no translation")
    return {"source":request.text, "translated":output,
            "direction":request.source+"-"+request.target,
            "duration_seconds":duration,
            "version":engine.last_run["version_sha256"]}


def run():
    if OUT.exists():
        raise FileExistsError("Never overwrite a previous actual model-integration receipt")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    registry = ModelRegistry()
    model_sources = registry.verify_sources()
    if "local_marian" in PROVIDERS:
        raise RuntimeError("Research model appeared in live app's default provider list")
    engine = SharedMarianResearchTranslator(registry=registry, retry_repetition=True)
    direct = [
        TranslationRequest("Hola, ¿me puedes ayudar?",source="es",target="en"),
        TranslationRequest("No eran quince, eran cincuenta.",source="es",target="en"),
        TranslationRequest("Bonjour, comment allez-vous ?",source="fr",target="en"),
        TranslationRequest("Je voudrais réserver une table pour demain.",source="fr",target="en"),
        TranslationRequest("Can you help me, please?",source="en",target="es"),
    ]
    live = []
    for req in direct:
        live.append(translate_and_record(engine,req))
        if engine.loaded_model_count != 1:
            raise RuntimeError("Multiple large model copies in memory")
    # One shared BridgeCaption Pipeline works for all 3 routes. No ASR/media,
    # context learned from the previous utterance, or app-default takeover.
    pipe = Pipeline(translator=engine,context_mode="all")
    for source, target, text in (
        ("es","en","Necesito ayuda con mi reserva."),
        ("fr","en","Je n'ai pas encore reçu mon billet."),
        ("en","es","Please tell me the meeting time."),
    ):
        result = pipe.translate(text,source,target)
        if not result.translated.strip() or "disabled" not in result.context_info:
            raise RuntimeError("App Pipeline incorrectly enabled untrained history")
        if pipe.history.snapshot():
            raise RuntimeError("Context-free local model retained previous private text")
        live.append({"direction":source+"-"+target,"source":text,
                     "translated":result.translated,"pipeline_context":result.context_info,
                     "version":engine.last_run["version_sha256"]})

    # Compare single-request predictions against the 2026-11-20 frozen,
    # already-reviewed 500-response development export. Deliberately avoid
    # using them as new train examples or a final independent test.
    archived = load_review()
    old_examples = defaultdict(list)
    for item in archived:
        if (item["Source type"]=="Amazon MASSIVE author validation"
            and len(item["Original spoken text"].split())<20
            and "\\" not in item["Original spoken text"]):
            old_examples[item["Direction"]].append(item)
    comparisons=[]
    engine.retry_repetition = False  # preserve the exact old deterministic decoding
    for direction in ("es-en","en-es"):
        pool = old_examples[direction][:10]
        if len(pool)!=10:
            raise RuntimeError("Expected ten unchanged archived dev examples per direction")
        src,tgt=direction.split("-")
        for item in pool:
            req=TranslationRequest(item["Original spoken text"], source=src,target=tgt)
            new=engine.translate(req)
            old=item["Actual model translation"].strip()
            comparisons.append({
                "direction":direction, "id":item["Case ID"],
                "old_output":old,"unified_output":new,
                "same_exact_text":new==old,
                "similar_after_whitespace_case": " ".join(new.casefold().split())==" ".join(old.casefold().split()),
            })
    # Switching to the two previously accepted rollback weights proves
    # separate independent version loading (not mixing weights).
    rollbacks=[]
    for source,text in (("es","Hola, buenas tardes."),("fr","Bonjour, merci beaucoup.")):
        pred=engine.translate(TranslationRequest(text,source=source,target="en"),use_previous=True)
        info=engine.last_run
        expected=model_sources[source+"-en"]["rollback_sha256"]
        if not pred.strip() or info["version_sha256"]!=expected:
            raise RuntimeError("Same-language rollback weight verification failed")
        rollbacks.append({"direction":source+"-en","text":pred,
                          "sha256":info["version_sha256"]})
    # Noncommercial WMT24 or the old Spanish subtitle-derived adapter must
    # never become routable as part of the unified translation interface.
    permitted={x["adapter_sha256"] for x in engine.routes() if x["adapter_sha256"]}
    restricted={r["sha256"] for r in registry.archived_noncommercial}
    if permitted&restricted:
        raise RuntimeError("An old NC-trained adapter was inadvertently enabled")

    exact=sum(x["same_exact_text"] for x in comparisons)
    normalized=sum(x["similar_after_whitespace_case"] for x in comparisons)
    result={
        "status":"actual_shared_marian_3_direction_pipeline_smoke",
        "models":model_sources,
        "actual_nonempty_translation_calls":len(live),
        "actual_captions":live,
        "archived_dev_comparisons":comparisons,
        "archived_dev_comparison_count":len(comparisons),
        "archived_dev_exact_parity":exact,
        "archived_dev_normalized_parity":normalized,
        "all_archived_dev_parity":normalized==len(comparisons),
        "rollback_translations":rollbacks,
        "model_load_count":engine.load_count,
        "model_cache_hits":engine.cache_hits,
        "max_simultaneous_models_in_runtime":1,
        "max_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "total_elapsed_seconds":time.monotonic()-started,
        "default_bridgecaption_provider_unchanged":True,
        "context_aware_model_activated":False,
        "nc_research_models_excluded":True,
        "independent_human_semantic_accuracy_verified":False,
        "final_holdout_touched":False,
        "new_training_updates":0,
        "translation_only_no_microphone":True,
        "critical_note":"This verifies shared runtime and weight/version routing, not "
                        "French 95% adequacy. Exact historical output parity "
                        "is tested on previously reviewed MASSIVE dev rows, "
                        "not unseen test examples."
    }
    OUT.mkdir(parents=True)
    (OUT/"actual_multilingual_integration.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("ACTUAL_FRENCH_SPANISH_SHARED_MODEL_SMOKE",json.dumps({
        key:result[key] for key in (
          "status","actual_nonempty_translation_calls","archived_dev_comparison_count",
          "archived_dev_exact_parity","archived_dev_normalized_parity",
          "model_load_count","model_cache_hits","max_rss_kib_linux",
          "total_elapsed_seconds","default_bridgecaption_provider_unchanged",
          "independent_human_semantic_accuracy_verified")
    }),flush=True)


if __name__ == "__main__":
    run()
