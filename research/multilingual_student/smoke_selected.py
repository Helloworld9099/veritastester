#!/usr/bin/env python3
"""Real BridgeCaption Pipeline smoke for the *selected* 38-update one-file model."""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from bridgecaption.provider import TranslationRequest
from bridgecaption.pipeline import Pipeline
from bridgecaption.providers import PROVIDERS
from research.multilingual_student.standalone_runtime import (
    OneWeightMultilingualResearchTranslator
)
from research.multilingual_student.export_selected import SAMPLE,OUTPUT_BASE

EXPERIMENT=ROOT/"research/multilingual_student/experiment_002"
MODEL=Path(os.environ["RUNNER_TEMP"])/OUTPUT_BASE
REPORT=EXPERIMENT/"selected_standalone_smoke.json"


def run():
    import torch
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    if REPORT.exists():
        raise FileExistsError("Never overwrite real integration evidence")
    if "single_m2m100" in PROVIDERS:
        raise RuntimeError("Experimental student must not replace working BridgeCaption provider")
    start=time.monotonic()
    tr=OneWeightMultilingualResearchTranslator(
        MODEL,receipt_path=EXPERIMENT/"selected_standalone_export.json")
    outputs={}
    for direction,text in SAMPLE.items():
        src,tgt=direction.split("-")
        before=time.monotonic()
        translated=tr.translate(TranslationRequest(text,source=src,target=tgt,
                                                    context_enabled=False))
        outputs[direction]={"translation":translated,
                            "seconds":time.monotonic()-before}
        if not translated.strip():
            raise RuntimeError("Selected full model returned an empty output")
    if tr.status()["model_loads"]!=1 or tr.status()["loaded_model_count"]!=1:
        raise RuntimeError("Single model was reloaded when switching languages")
    pipe=Pipeline(translator=tr,context_mode="all")
    for src,tgt,text in (("es","en","Necesito revisar mi horario."),
                         ("fr","en","Je n'ai toujours pas reçu le billet."),
                         ("en","es","Please repeat the last sentence.")):
        output=pipe.translate(text,src,tgt)
        if not output.translated.strip() or "disabled" not in output.context_info:
            raise RuntimeError("App Pipeline incorrectly enabled untrained context")
    if pipe.history.snapshot():
        raise RuntimeError("Local text model should not retain history")
    if tr.status()["model_loads"]!=1:
        raise RuntimeError("Text Pipeline reloaded more than one model")
    report={
        "status":"real_selected_one_multilingual_weight_pipeline_smoke_passed",
        "trained_selected_adapter":"a00c7965211853df07ce3fa032a76283dfc984dcdaeac85e03263445029ffb8e",
        "routes":list(SAMPLE),
        "total_network_loads":1,
        "six_actual_translation_outputs":outputs,
        "additional_BridgeCaption_Pipeline_translations":3,
        "default_provider_unchanged":True,
        "speech_recognition_verified":False,
        "prior_turns_conditioning_verified":False,
        "human_95_percent_adequacy_verified":False,
        "final_test_or_reviewed_500_used":False,
        "max_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds":time.monotonic()-start,
    }
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("SELECTED_ONE_WEIGHT_REAL_APP_INFERENCE_VERIFIED",json.dumps({
        "routes":len(SAMPLE),"loads":1,
        "rss_kib":report["max_rss_kib_linux"],
        "elapsed_seconds":report["elapsed_seconds"],
        "real_human_accuracy_claim":False}),flush=True)


if __name__=="__main__":
    run()
