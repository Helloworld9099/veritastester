#!/usr/bin/env python3
"""Fair, author-VALIDATION-only 12-case comparison of SELECTED shared student
against the original checkpoint specialists. Never a new final human test.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.multilingual_student.data import prepare
from research.unified_models.registry import ModelRegistry
from research.unified_models.runtime import SharedMarianResearchTranslator
from bridgecaption.provider import TranslationRequest

EXP=ROOT/"research/multilingual_student/experiment_002"
OUTPUT=EXP/"selected_legacy_comparison.json"
COMPARE=("es-en","fr-en","en-es")


def run():
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite selected model regression comparison")
    import torch
    from sacrebleu.metrics import CHRF,BLEU
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    report=json.loads((EXP/"checkpoint_selection.json").read_text())
    if (report["status"]!="actual_saved_shared_multilingual_checkpoint_sweep_completed"
        or report["selection_reason"]!="development_selected_continuation_not_production"):
        raise RuntimeError("No qualifying shared student selected")
    chosen=report["selected_research_checkpoint_relative"]
    name=Path(chosen).name
    scores=report["evaluations"][name]["scores"]
    _,dev,manifest=prepare()
    first=json.loads((ROOT/"research/multilingual_student/experiment_001/report.json").read_text())
    if manifest["dev_sha256"]!=first["dev_dataset_sha256"]:
        raise RuntimeError("Selection and legacy comparison use different publisher DEV cases")
    old=ModelRegistry()
    proofs=old.verify_sources()
    translator=SharedMarianResearchTranslator(registry=old,retry_repetition=False)
    results={}
    began=time.monotonic()
    for direction in COMPARE:
        src,tgt=direction.split("-")
        selected=dev[direction][:12]
        translations=[translator.translate(
            TranslationRequest(x["src"],source=src,target=tgt,context_enabled=False)
        ) for x in selected]
        gold=[x["tgt"] for x in selected]
        chrf=CHRF().corpus_score(translations,[gold]).score
        bleu=BLEU().corpus_score(translations,[gold]).score
        cand=scores[direction]
        results[direction]={
            "n":12,
            "old_model_revision_or_adapter_sha256":proofs[direction]["active_sha256"] or proofs[direction]["revision"],
            "old_trained_model_chrF_not_accuracy":round(chrf,4),
            "old_trained_model_BLEU_not_accuracy":round(bleu,4),
            "one_model_selected_chrF_not_accuracy":cand["chrF_not_accuracy"],
            "one_model_selected_BLEU_not_accuracy":cand["BLEU_not_accuracy"],
            "one_model_minus_old_chrF":round(cand["chrF_not_accuracy"]-chrf,4),
            "one_model_minus_old_BLEU":round(cand["BLEU_not_accuracy"]-bleu,4),
            "same_development_case_ids":True,
            "independent_bilingual_meaning_labels":False,
        }
        print("ACTUAL_SELECTED_UNIFIED_VS_OLD_"+direction.upper().replace("-","_"),
              json.dumps(results[direction]),flush=True)
    evidence={
        "status":"actual_selected_single_model_vs_old_specialists_on_identical_DEV",
        "selected_research_adapter_sha256":report["selected_research_sha256"],
        "verified_publisher_development_sha256":manifest["dev_sha256"],
        "selected_model_is_one_shared_neural_network":True,
        "results":results,
        "production_promoted":False,
        "old_language_specialists_preserved":True,
        "independent_95_percent_meaning_accuracy_verified":False,
        "reviewed_500_or_final_holdout_used_for_selection":False,
        "no_new_training_executed":True,
        "paid_spend_usd":0,
        "elapsed_seconds":time.monotonic()-began,
        "rss_peak_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "warning":"Only 12 publisher intent DEV examples/direction. This automatic "
                  "chrF/BLEU comparison is not an independent semantic accuracy "
                  "study. Old models remain unchanged even if the selected "
                  "shared model has higher text similarity.",
    }
    OUTPUT.write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+"\n")
    print("FAIR_SELECTED_ONE_MODEL_VS_PREVIOUS_TEACHERS_VERIFIED",json.dumps({
        "directions":list(results),"selected_student":report["selected_research_sha256"],
        "human_95_claim":False}),flush=True)


if __name__=="__main__":
    run()
