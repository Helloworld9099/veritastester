#!/usr/bin/env python3
"""Real DEV-only candidate comparison for Spanish repetition + content loss.

Does NOT pick a decoding strategy from only 3 reviewed examples. It compares
prior allowed-license stage3, original pretrained base, and several decoding
options on the actual reviewed sources. No final holdout. Human meaning must
still be checked, especially for omissions; count of punctuation segments is
ONLY a conservative flag, never a meaning accuracy metric.
"""
from __future__ import annotations

import gc
import hashlib
import json
import re
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/"src"), str(ROOT/"research/es_en_v2")]
from research.es_en_v2.review_feedback_500 import load_review
from research.es_en_v2.inference_safety import has_generation_loop
from research.unified_models.runtime import SharedMarianResearchTranslator
from research.unified_models.registry import ModelRegistry

OUT = ROOT / "research/unified_models/receipts/spanish_review_decoder_ablation"
LOOPS = (311, 314, 416)
OTHER_OMISSIONS = (33, 35, 79, 219, 220)
TRIALS = {
    "existing_bounded_retry_ngram3": {
        "num_beams":4, "max_new_tokens":96, "no_repeat_ngram_size":3,
        "repetition_penalty":1.12,
    },
    "ngram2_length1_2": {
        "num_beams":4, "max_new_tokens":96, "no_repeat_ngram_size":2,
        "repetition_penalty":1.12, "length_penalty":1.2,
    },
    "ngram2_min6": {
        "num_beams":4, "max_new_tokens":96, "no_repeat_ngram_size":2,
        "repetition_penalty":1.2, "min_new_tokens":6,
    },
    "greedy_ngram2": {
        "num_beams":1, "max_new_tokens":96, "do_sample":False,
        "no_repeat_ngram_size":2, "repetition_penalty":1.1,
    },
}
SENTENCE_END = re.compile(r"[.!?¡¿]+(?=\s|$)")
TOKENS = re.compile(r"[^\W_]+",re.UNICODE)


def warn_for_possible_omission(source: str, translated: str):
    """Only a review signal; never itself a human judgment."""
    source_clauses = len(SENTENCE_END.findall(source))
    output_clauses = len(SENTENCE_END.findall(translated))
    src_tokens = len(TOKENS.findall(source))
    out_tokens = len(TOKENS.findall(translated))
    return {
        "source_end_punctuation_count":source_clauses,
        "translated_end_punctuation_count":output_clauses,
        "possible_missing_source_clause": source_clauses>=2 and output_clauses<source_clauses,
        "possible_short_output": out_tokens*2 < src_tokens if src_tokens>=5 else False,
        "obvious_long_generation_loop":has_generation_loop(translated),
    }


def run():
    if OUT.exists():
        raise FileExistsError("Never overwrite completed review ablation")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start=time.monotonic()
    rows=load_review()
    wanted=LOOPS+OTHER_OMISSIONS
    for row in wanted:
        if rows[row-2]["Direction"]!="es-en":
            raise RuntimeError("Only Spanish developer examples may be used")
    prepared=[(n,rows[n-2]["Original spoken text"],rows[n-2]["Actual model translation"]) for n in wanted]
    # Original Marathi? No: the pinned Spanish base from the historical
    # integration code is preserved separately from both trained adapters.
    from research.es_en_v2.stress_eval import load as load_previous_base
    predict, attach = load_previous_base(True)
    base_outputs=predict([s for n,s,_ in prepared])
    del predict,attach
    gc.collect()

    engine=SharedMarianResearchTranslator(retry_repetition=False)
    spec=engine.registry.select("es","en")
    loaded=engine._get(spec,use_previous=False)
    variants={}
    for n,text,original in prepared:
        entry={"row":n,"source":text,
               "reviewed_previous_model_output":original,
               "archived_previous_repeated":has_generation_loop(original),
               "variants":{}}
        original_base=base_outputs[wanted.index(n)]
        entry["variants"]["untouched_original_pretrained_base"]={
            "text":original_base,
            **warn_for_possible_omission(text,original_base)}
        for name,options in TRIALS.items():
            with engine._lock:
                output=engine._generate(loaded,text,override=options)
            entry["variants"][name]={"text":output,**warn_for_possible_omission(text,output)}
        variants[n]=entry
    # Do NOT keep a second base and adapter loaded simultaneously.
    engine.unload()
    previous=engine._get(spec,use_previous=True)
    for n,text,_ in prepared:
        output=engine._generate(previous,text)
        variants[n]["variants"]["previous_stage3_trained_adapter"]={
            "text":output,**warn_for_possible_omission(text,output)}
    results=[variants[n] for n in wanted]
    summary={label:{
        "nonloop_outputs":sum(not x["variants"][label]["obvious_long_generation_loop"] for x in results),
        "possible_missing_source_clause_flags":sum(x["variants"][label]["possible_missing_source_clause"] for x in results),
        "possibly_short_outputs":sum(x["variants"][label]["possible_short_output"] for x in results),
    } for label in results[0]["variants"]}
    manifest={
        "status":"real_development_only_decode_comparison",
        "tested_es_en_dev_rows":list(wanted),
        "reviewed_long_loops":list(LOOPS),
        "tested_cases_n":len(results),
        "actual_model_predictions":results,
        "diagnostic_counts_not_quality_scores":summary,
        "trained_model_sha256":spec.active_adapter["sha256"],
        "previous_permitted_adapter_sha256":spec.rollback_adapter["sha256"],
        "no_nc_model_or_data_involved":True,
        "human_translation_quality_verified":False,
        "default_app_provider_changed":False,
        "no_new_optimizer_steps":True,
        "test_or_final_holdout_not_accessed":True,
        "elapsed_seconds":time.monotonic()-start,
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations":"Bilingual meaning is not inferred from punctuation count or output length. "
                      "Never promote a decoding variant from these already-reviewed "
                      "development examples without independent validation."
    }
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"actual_decode_candidates.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print("REAL_SPANISH_REVIEW_DECODING_CANDIDATES",json.dumps({
        "tested_rows":list(wanted),
        "diagnostics":summary,
        "seconds":manifest["elapsed_seconds"],
        "quality_verified":False
    }),flush=True)


if __name__=="__main__":
    run()
