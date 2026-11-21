#!/usr/bin/env python3
"""Fair development comparison against the actually trained old Marian models.

The shared M2M100 and old language-specialist translators are evaluated on
the exact same reconstructed publisher validation utterances, independently
of final test or the 500 reviewed cases. Nothing is promoted from chrF alone.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from bridgecaption.provider import TranslationRequest
from research.multilingual_student.model import sha256
from research.multilingual_student.data import prepare
from research.unified_models.registry import ModelRegistry
from research.unified_models.runtime import SharedMarianResearchTranslator

OUT=ROOT/"research/multilingual_student/experiment_001/legacy_comparison.json"
ROUTES=("es-en","fr-en","en-es")


def run():
    if OUT.exists():
        raise FileExistsError("Existing peer-review comparison cannot be replaced")
    from sacrebleu.metrics import CHRF,BLEU
    import torch
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    folder=OUT.parent
    report=json.loads((folder/"report.json").read_text())
    if report["status"]!="completed_actual_single_multilingual_student_training":
        raise RuntimeError("No saved actual trained shared student")
    _,dev,manifest=prepare()
    if manifest["dev_sha256"]!=report["dev_dataset_sha256"]:
        raise RuntimeError("Legacy comparison must use same frozen developer cases")
    start=time.monotonic()
    reg=ModelRegistry()
    old=reg.verify_sources()
    old_engine=SharedMarianResearchTranslator(registry=reg,retry_repetition=False)
    results={}
    for direction in ROUTES:
        src,tgt=direction.split("-")
        chosen=dev[direction][:6]
        predictions=[old_engine.translate(
            TranslationRequest(item["src"],source=src,target=tgt,context_enabled=False)
        ) for item in chosen]
        gold=[row["tgt"] for row in chosen]
        old_chrf=CHRF().corpus_score(predictions,[gold]).score
        old_bleu=BLEU().corpus_score(predictions,[gold]).score
        new=report["automated_development_after_training"][direction]
        results[direction]={
            "n":len(chosen),
            "old_original_trained_model":old[direction]["active_sha256"]
                or old[direction]["revision"],
            "old_chrf_not_accuracy":round(old_chrf,4),
            "old_bleu_not_accuracy":round(old_bleu,4),
            "one_shared_student_chrf_not_accuracy":new["chrF_not_accuracy"],
            "one_shared_student_bleu_not_accuracy":new["BLEU_not_accuracy"],
            "difference_student_minus_legacy_chrF":round(
                new["chrF_not_accuracy"]-old_chrf,4),
            "difference_student_minus_legacy_BLEU":round(
                new["BLEU_not_accuracy"]-old_bleu,4),
            "same_frozen_development_inputs":True,
            "human_meaning_accuracy_evaluated":False,
        }
        print("MARIAN_VS_ONE_MULTILINGUAL_"+direction.upper().replace("-","_"),
              json.dumps(results[direction]),flush=True)
    # Strict no-claim release block even when automated similarity scores win.
    no_regress_by_metric=all(row["difference_student_minus_legacy_chrF"] >= -1.0
                              for row in results.values())
    evidence={
        "status":"real_fair_old_models_vs_single_multilingual_student_dev_comparison",
        "frozen_dev_sha256":manifest["dev_sha256"],
        "results":results,
        "language_specialist_models_retained":True,
        "automated_chrf_regression_gate":no_regress_by_metric,
        "production_candidate_approved":False,
        "independent_bilingual_pass_rates_available":False,
        "source_development_only":True,
        "reviewed_500_used_for_training_or_selection":False,
        "publisher_test_or_final_holdout_accessed":False,
        "elapsed_seconds":time.monotonic()-start,
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note":"A six-pair publisher-intent development comparison per direction is "
               "too small to infer real-world accuracy, slang adequacy or robust "
               "non-regression. Keep legacy champions regardless of chrF result."
    }
    OUT.write_text(json.dumps(evidence,indent=2,ensure_ascii=False)+"\n")
    print("REAL_NO_REGRESSION_LEGACY_COMPARISON_DONE",json.dumps({
        "can_promote":False,"automatic_dev_chrF_gate":no_regress_by_metric,
        "directions":list(results),"human_95_verified":False}),flush=True)


if __name__=="__main__":
    run()
