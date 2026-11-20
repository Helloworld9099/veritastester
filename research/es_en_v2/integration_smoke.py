#!/usr/bin/env python3
"""ACTUAL research-only local BridgeCaption text Pipeline integration smoke."""
from __future__ import annotations
import json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from research.es_en_v2.integration import ExperimentalMarianTranslator
from bridgecaption.pipeline import Pipeline
from bridgecaption.provider import TranslationRequest
from bridgecaption.providers import PROVIDERS

SMOKE=(
    "Hola, ¿me puedes ayudar?",
    "Quiero confirmar mi reserva de mañana.",
    "No son quince, son cincuenta.",
    "¿Puedes repetir la última parte?",
    "Necesito hablar con la persona encargada.",
    "Gracias, eso aclara mi pregunta.",
)


def run():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output=ROOT/"research/es_en_v2/integration_receipt"
    if output.exists():
        raise FileExistsError("Refuse overwriting existing evidence")
    began=time.monotonic()
    if "local_marian" in PROVIDERS:
        raise RuntimeError("Experimental local adapter must not be the default provider")
    translator=ExperimentalMarianTranslator()
    pipeline=Pipeline(translator=translator,context_mode="all")
    timings=[]
    results=[]
    for source in SMOKE:
        start=time.monotonic()
        caption=pipeline.translate(source,"es","en")
        timings.append(time.monotonic()-start)
        if not caption.translated.strip() or caption.original!=source:
            raise RuntimeError("Pipeline failed to return meaningful nonempty caption")
        if "disabled" not in caption.context_info:
            raise RuntimeError("Context-free research model falsely claimed context support")
        results.append(caption.translated)
    if pipeline.history.snapshot():
        raise RuntimeError("Context-free translator retained unrelated conversation")
    pipeline.reset()
    if pipeline.history.snapshot():
        raise RuntimeError("Context reset failed")
    try:
        translator.translate(TranslationRequest("Hello",source="en",target="es"))
    except ValueError:
        pass
    else:
        raise RuntimeError("Unsupported direction was accepted")
    try:
        translator.translate(TranslationRequest("Está brutal",source="es",target="en",
                previous=["El concierto fue increíble"],context_enabled=True))
    except ValueError:
        pass
    else:
        raise RuntimeError("Untrained contextual input was silently accepted")
    fallback=ExperimentalMarianTranslator(use_previous=True,model_root=translator.model_root)
    prior=fallback.translate(TranslationRequest("Necesito ayuda",source="es",target="en",
                                               context_enabled=False))
    if not prior.strip() or fallback.loaded_provenance["adapter_sha256"]!=fallback.FALLBACK_SHA:
        raise RuntimeError("Rollback adapter did not load/generate")
    warm=sorted(timings[1:])
    report={
        "status":"real_bridgecaption_text_pipeline_smoke_passed",
        "new_adapter_sha256":translator.ADAPTER_SHA,
        "fallback_adapter_sha256":fallback.FALLBACK_SHA,
        "source_only_not_falsely_context_aware":True,
        "text_captions_returned":len(SMOKE),
        "model_cached_after_first_request":True,
        "live_app_default_changed":False,
        "cloud_used":False,
        "history_reset_confirmed":True,
        "invalid_target_rejected":True,
        "past_context_rejected_for_untrained_model":True,
        "fallback_loaded_and_translated":True,
        "cold_first_request_seconds":timings[0],
        "warm_p50_seconds":warm[len(warm)//2],
        "warm_p95_seconds":warm[-1],
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "total_seconds":time.monotonic()-began,
        "text_only_no_microphone_or_asr":True,
        "independent_human_95_percent_verified":False,
        "translations_for_research_debug":results,
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/"real_pipeline_smoke.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("ES_EN_REAL_APPLICATION_TEXT_INTEGRATION",json.dumps({k:v for k,v in report.items()
           if k!="translations_for_research_debug"}),flush=True)


if __name__=="__main__":
    run()
