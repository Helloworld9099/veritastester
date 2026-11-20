#!/usr/bin/env python3
"""Actual one-retry ablation on the three observed ES→EN generation loops.

Consumes the *previously reviewed development outputs* only. This cannot
establish accuracy. Unchanged output for normal requests is a design property:
the regeneration path is called only on loop detection, never on all captions.
No training, no final holdout, no default-provider change.
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research.es_en_v2.review_feedback_500 import load_review
from research.es_en_v2.inference_safety import (
    RETRY_GENERATION_OPTIONS, RepetitionLoopError, has_generation_loop,
    retry_if_loop,
)
from research.es_en_v2.integration import ExperimentalMarianTranslator

LOOP_ROWS = (311, 314, 416)
OUT = ROOT / "research/es_en_v2/reviewer_feedback_20261120/loop_recovery"


def run():
    if OUT.exists():
        raise FileExistsError("Do not overwrite a prior experimental run")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    rows = load_review()
    original = ROOT / "research/es_en_v2/reviewer_examples_500/BridgeCaption_500_Real_Model_Translations_Review.csv"
    previous_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    translator = ExperimentalMarianTranslator()
    records = []
    start = time.monotonic()
    total_tokens = 0
    for row_number in LOOP_ROWS:
        item = rows[row_number - 2]
        if item["Direction"] != "es-en":
            raise RuntimeError("Observed loop does not belong to trained ES->EN model")
        before = item["Actual model translation"]
        if not has_generation_loop(before):
            raise RuntimeError("Expected original 500-response repetition at row " + str(row_number))

        def alternate():
            translator._load_once()
            assert translator._model is not None and translator._tokenizer is not None
            x = translator._tokenizer(
                item["Original spoken text"], return_tensors="pt",
                truncation=True, max_length=128,
            )
            with torch.no_grad():
                ids = translator._model.generate(**x, **RETRY_GENERATION_OPTIONS)
            return translator._tokenizer.decode(ids[0], skip_special_tokens=True)

        began = time.monotonic()
        try:
            state = retry_if_loop(before, alternate)
            after = state.text
            rescued = state.rescued
            status = "no_long_repetition_detected_after_retry"
            attempts = state.attempts
        except RepetitionLoopError:
            after = None
            rescued = False
            status = "repeated_or_blank_even_after_retry_needs_fallback"
            attempts = 2
        records.append({
            "excel_row":row_number, "case_id":item["Case ID"], "direction":"es-en",
            "source":item["Original spoken text"],
            "prior_context_visible_to_reviewer_not_model":item["Prior context (reviewer only)"],
            "previous_real_bot_output":before,
            "constrained_real_bot_output":after,
            "original_repetition_detected":True,
            "retry_recovers_from_repeat":rescued,
            "status":status,
            "attempts":attempts,
            "retry_only_seconds":time.monotonic()-began,
            "human_meaning_preserved_verified":False,
        })

    # All other 497 output texts remain identical; the retry callable isn't
    # invoked unless the output has an excessive repetition loop.
    normal = rows[318-2]["Actual model translation"]
    def should_not_redecode():
        raise AssertionError("Normal unchanged translation unexpectedly redecoded")
    unchanged = retry_if_loop(normal, should_not_redecode)
    if unchanged.text != normal or unchanged.attempts != 1:
        raise RuntimeError("Ordinary translator output was changed unnecessarily")

    results = {
        "status":"real_es_en_one_retry_development_ablation",
        "origin_reviewed_500_csv_sha256":previous_sha,
        "pinned_es_en_adapter_sha256":ExperimentalMarianTranslator.ADAPTER_SHA,
        "original_repetition_loops_checked":len(records),
        "loops_no_longer_detected_after_retry":sum(x["retry_recovers_from_repeat"] for x in records),
        "still_unresolved_loops":sum(not x["retry_recovers_from_repeat"] for x in records),
        "unchanged_nonloop_control":True,
        "generation_options":RETRY_GENERATION_OPTIONS,
        "tested_cases":records,
        "elapsed_seconds":time.monotonic()-start,
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "automatic_metric_or_human_accuracy_claim":False,
        "final_holdout_accessed":False,
        "trained_on_reviewer_examples":False,
        "production_provider_modified":False,
        "limitation":"This fixes only detectable repetition. It does not solve idioms, "
                     "context, negation, names, source errors, or incomplete clauses. "
                     "A human must still assess whether nonrepetitive output has correct meaning."
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "actual_loop_retry_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("ACTUAL_REVIEWED_SPANISH_LOOPS_REDECODED",json.dumps({
        "cases":len(records),
        "resolved":results["loops_no_longer_detected_after_retry"],
        "unresolved":results["still_unresolved_loops"],
        "elapsed_seconds":results["elapsed_seconds"],
        "human_semantic_accuracy_verified":False,
    }),flush=True)


if __name__ == "__main__":
    run()
