#!/usr/bin/env python3
"""Try real Spanish clause-preserving translation on already-reviewed DEV.

This is an ablation, not a production fallback or human accuracy test. No
training, paid APIs, noncommercial weights, uninspected final test data.
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from research.unified_models.runtime import SharedMarianResearchTranslator
from research.unified_models.segmented_recovery import (
    separate_current_clauses,translate_clauses,
)
from research.es_en_v2.review_feedback_500 import load_review
from research.es_en_v2.inference_safety import has_generation_loop

OUT=ROOT/"research/unified_models/receipts/spanish_segment_ablation"
ROWS=(311,314,416,33,35,79,219,220)
DECODE={
    "num_beams":4,
    "max_new_tokens":96,
    "no_repeat_ngram_size":2,
    "repetition_penalty":1.12,
    "length_penalty":1.2,
}


def run():
    if OUT.exists():
        raise FileExistsError("Never overwrite one completed experimental comparison")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start=time.monotonic()
    reviewer=load_review()
    model=SharedMarianResearchTranslator(retry_repetition=False)
    spec=model.registry.select("es","en")
    loaded=model._get(spec,use_previous=False)
    evidence=[]
    for row in ROWS:
        case=reviewer[row-2]
        if case["Direction"]!="es-en":
            raise RuntimeError("Wrong direction for Spanish developer experiment")
        clauses=separate_current_clauses(case["Original spoken text"])
        segment_attempts=[]
        def generate_part(src):
            output=model._generate(loaded,src,override=DECODE)
            segment_attempts.append({"source":src,"prediction":output,
                                     "loop":has_generation_loop(output)})
            if has_generation_loop(output):
                raise RuntimeError("Still looping inside a clause")
            return output
        try:
            text=translate_clauses(clauses,generate_part)
            status="all_source_clauses_translated_nonempty"
        except (RuntimeError,ValueError) as exc:
            text=None
            status=type(exc).__name__+": "+str(exc)
        evidence.append({
            "excel_row":row,
            "case_id":case["Case ID"],
            "source":case["Original spoken text"],
            "historical_dev_prediction":case["Actual model translation"],
            "source_current_clause_count":len(clauses),
            "clause_predictions":segment_attempts,
            "combined_segmented_output":text,
            "all_clauses_received_nonempty_translation":text is not None,
            "obvious_loop_on_final_output":has_generation_loop(text) if text else None,
            "translation_semantics_human_verified":False,
            "status":status,
        })
    report={
        "status":"actual_spanish_clause_preservation_development_ablation",
        "model_sha256":spec.active_adapter["sha256"],
        "dataset":"previously reviewed 500-source development export",
        "reviewer_rows":list(ROWS),
        "examples":evidence,
        "all_clauses_generated":sum(x["all_clauses_received_nonempty_translation"] for x in evidence),
        "final_outputs_without_long_repetition":sum(x["obvious_loop_on_final_output"] is False for x in evidence),
        "source_or_target_references_used_for_training":False,
        "final_holdout_used":False,
        "human_adequacy_verified":False,
        "production_enabled":False,
        "decoder_config":DECODE,
        "elapsed_seconds":time.monotonic()-start,
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "warning":"Current-clause segmentation can change pronouns, context and register. "
                  "Generating nonempty text per clause is NOT equivalent to retaining "
                  "each clause's meaning. Independent bilingual review is necessary."
    }
    OUT.mkdir(parents=True)
    (OUT/"source_clauses_actual_model_outputs.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("ACTUAL_SPANISH_DEV_SEGMENTATION",json.dumps({
        "examples":len(evidence),
        "all_clauses_generated":report["all_clauses_generated"],
        "nonloop_outputs":report["final_outputs_without_long_repetition"],
        "human_adequacy_verified":False,
        "elapsed_seconds":report["elapsed_seconds"],
    }),flush=True)


if __name__=="__main__":
    run()
