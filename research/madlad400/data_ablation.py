#!/usr/bin/env python3
"""Actual DATA experiment: hard-example coverage vs random Tatoeba sampling.

No model weights loaded. Do not keep training on 350 arbitrary clean sentences
when real errors are slang, wrong pronouns, negation, multi-clause omissions
and time/number swaps. Compare TWO permitted source selection strategies on
the SAME raw CC BY 2.0 FR Tatoeba corpus, exclude all publisher validation
and previously reviewed 500 examples, and flag risky targets for bilingual
verification before they can be considered gold.

A diversity increase is not a translation-quality improvement!
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.conversation_growth.data import (
    settings,download,pairs_from_zip,sample_pairs,robust_filter,normalize
)
from research.multilingual_student.data import get_reviewed_sources,load_groups
from research.madlad400.model import file_sha256

OUT=ROOT/"research/madlad400/data_ablation_001"
REVIEW_DIR=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"madlad400-hard-examples"
SAMPLE=350

PATTERNS={
    "idiom_or_regional_phrase":re.compile(
        r"\b(?:i(?:'| a)m down|left on read|got your back|"
        r"no big deal|pull(?:ing)? (?:my|your|his|her) leg|"
        r"call it a day|stay on top of|"
        r"qué onda|que onda|echar (?:una|me) mano|"
        r"me dejó en visto|no manches|ponerse las pilas|"
        r"posé un lapin|poser un lapin|j'en ai marre|"
        r"c'est ouf|c'est pas grave)\b",re.I),
    "explicit_negation":re.compile(
        r"\b(?:not|never|no|don't|didn't|won't|can't|"
        r"ningún|nunca|jamás|no|ni|"
        r"ne|n'|jamais|pas|plus|personne)\b",re.I),
    "time_number":re.compile(
        r"\b(?:[0-9]+|a\.?m\.?|p\.?m\.?|tonight|tomorrow|"
        r"yesterday|mañana|ayer|hoy|demain|hier|aujourd'hui)\b",re.I),
    "pronoun_reference":re.compile(
        r"\b(?:he|she|they|him|her|them|it|this one|that one|"
        r"ella|ellos|ellas|él|lo|la|le|les|esto|eso|"
        r"elle|il|elles|ils|lui|leur|celle|celui)\b",re.I),
    "informal_register":re.compile(
        r"\b(?:dude|bro|hey|yo|tú|usted|ustedes|vos|"
        r"chico|tío|vale|tu|vous|mec|salut)\b",re.I),
    "multi_clause":re.compile(r"[,;:][^,;:]{5,}[,;:?!.\u00bf\u00a1]",re.I),
}
CATEGORIES=tuple(PATTERNS)


def categories(a:str,b:str)->list[str]:
    joint=a+" "+b
    return [key for key,pattern in PATTERNS.items() if pattern.search(joint)]


def deterministic_rank(seed:int,pair:str,position:int)->str:
    return hashlib.sha256(f"{seed}:{pair}:{position}".encode()).hexdigest()


def quality_candidates(raw:list[tuple[str,str]],pair:str,banned:set[str],seed:int):
    """Select reproducible real rows with hard-case diversity without gold claims."""
    pool=[]
    seen=set()
    for position,(first,second) in enumerate(raw):
        a,b=first.strip(),second.strip()
        if not robust_filter(a,b):
            continue
        na,nb=normalize(a),normalize(b)
        if na in banned or nb in banned or na in seen or nb in seen:
            continue
        # Avoid duplicate rows and trivial punctuation/case changes.
        seen.update((na,nb))
        found=categories(a,b)
        if not found:
            continue
        pool.append({"pair":pair,"row_id":str(position),
                     "source":a,"target":b,"categories":found,
                     "data_source":"Tatoeba via OPUS v2023-04-12",
                     "license":"CC-BY-2.0-FR",
                     "independent_bilingual_adequacy_verified":False,
                     "eligible_for_unreviewed_context_or_slang_training":False,
                     "priority_hash":deterministic_rank(seed,pair,position)})
    pool.sort(key=lambda row:(-len(row["categories"]),row["priority_hash"]))
    # Encourage each category to receive a quota before taking top coverage.
    by_tag={key:[] for key in CATEGORIES}
    for row in pool:
        for key in row["categories"]:
            by_tag[key].append(row)
    selected=[]
    used=set()
    for tag in CATEGORIES:
        for row in by_tag[tag][:min(35,SAMPLE//len(CATEGORIES))]:
            if row["row_id"] not in used:
                selected.append(row)
                used.add(row["row_id"])
    for row in pool:
        if len(selected)>=SAMPLE:
            break
        if row["row_id"] not in used:
            selected.append(row)
            used.add(row["row_id"])
    if len(selected)<max(90,SAMPLE//3):
        raise RuntimeError("Insufficient high-quality hard examples to compare selection methods")
    return selected[:SAMPLE],pool


def stats(rows)->dict:
    tags=Counter()
    for row in rows:
        tags.update(row["categories"])
    return {
        "count":len(rows),
        "category_counts":dict(sorted(tags.items())),
        "fraction_with_any_flag":round(
            sum(bool(row["categories"]) for row in rows)/max(1,len(rows)),4),
        "fraction_with_two_or_more_flags":round(
            sum(len(row["categories"])>=2 for row in rows)/max(1,len(rows)),4),
    }


def run():
    if OUT.exists():
        raise FileExistsError("Data-ablation research evidence exists already")
    cfg=settings()
    known_ids={
        x["pair"]:x for x in cfg["parallel_sources"]
    }
    manifest=ROOT/"research/conversation_growth/experiment_001/data_manifest.json"
    old=json.loads(manifest.read_text())
    if old.get("record_count")!=1050:
        raise RuntimeError("Historical exact licensed parallel data receipt was modified")
    reviewed=get_reviewed_sources()
    validation=load_groups("validation")
    banned=reviewed | {
        normalize(record[lang]) for record in validation for lang in ("en","es","fr")
    }
    outputs={}
    candidates_for_review=[]
    began=time.monotonic()
    for pair in ("en-es","en-fr","es-fr"):
        source=known_ids[pair]
        expected=old["sources"][pair]["archive_sha256"]
        raw_zip=None
        for url in source["download_urls"]:
            try:
                candidate=download(url)
                if hashlib.sha256(candidate).hexdigest()!=expected:
                    continue
                raw_zip=candidate
                break
            except (OSError,TimeoutError):
                continue
        if raw_zip is None:
            raise RuntimeError("No exact license-verified Tatoeba corpus archive for "+pair)
        raw=pairs_from_zip(raw_zip,pair)
        baseline=sample_pairs(raw,pair,SAMPLE,banned,cfg["training"]["seed"])
        adapted,found_pool=quality_candidates(raw,pair,banned,cfg["training"]["seed"])
        base_rows=[{**row,"categories":categories(row["source"],row["target"])}
                   for row in baseline]
        # This is a source-corpus strategy comparison, NOT ML training or
        # a semantic evaluation; hard-case gold targets require human review.
        b,a=stats(base_rows),stats(adapted)
        if len(base_rows)!=SAMPLE:
            raise RuntimeError("Baseline could not retrieve exact agreed comparison sample")
        outputs[pair]={
            "source_zip_sha256":expected,
            "raw_parallel_pairs":len(raw),
            "filtered_hard_case_pool":len(found_pool),
            "random_baseline":b,"category_stratified_candidate":a,
            "hard_category_coverage_delta":{
                cat:a["category_counts"].get(cat,0)-b["category_counts"].get(cat,0)
                for cat in CATEGORIES},
            "selected_id_hash":hashlib.sha256(
                ",".join(sorted(row["row_id"] for row in adapted)).encode()).hexdigest(),
            "train_or_production_promoted":False,
            "human_translation_quality_unverified":True,
        }
        candidates_for_review.extend(
            {k:row[k] for k in ("pair","row_id","source","target",
                                "categories","license","data_source")}
            for row in adapted[:30])
        print("ACTUAL_LICENSED_DATA_SELECTION_EXPERIMENT",json.dumps({
            "pair":pair,"baseline":b,"stratified":a,
            "candidate_needs_bilingual_review":True
        }),flush=True)
    result={
        "status":"actual_three_corpus_parallel_data_selection_ablation_completed",
        "source_license":"Tatoeba CC BY 2.0 FR (attribution required)",
        "source_url":"https://tatoeba.org/en/downloads",
        "comparisons":outputs,
        "strategies_tested":["deterministic_random","stratified_hard_phenomena"],
        "publisher_validation_and_500_reviewed_excluded":True,
        "final_holdout_accessed":False,
        "new_optimizer_updates":0,
        "all_rows_authentic_parallel_not_synthetic":True,
        "hard_case_correctness_independently_verified":False,
        "slang_rows_must_be_reviewed_before_supervised_training":True,
        "deployment_approved":False,
        "paid_spend_usd":0,
        "elapsed_seconds":round(time.monotonic()-began,3),
        "conclusion":"A gain in hard-case data coverage does not imply a gain in "
                     "translation accuracy; test with independently verified "
                     "references before selecting a future training pool."
    }
    OUT.mkdir(parents=True)
    (OUT/"report.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    (OUT/"report.sha256").write_text(file_sha256(OUT/"report.json")+"  report.json\n")
    REVIEW_DIR.mkdir(parents=True,exist_ok=True)
    (REVIEW_DIR/"potential_hard_examples.jsonl").write_text(
        "\n".join(json.dumps(r,ensure_ascii=False) for r in candidates_for_review)+"\n")
    print("ACTUAL_NO_GPU_DATA_QUALITY_HARD_CASE_ABLATION_DONE",json.dumps({
        "parallel_corpora":len(outputs),
        "selected_per_corpus":SAMPLE,
        "needs_bilingual_quality_review":True,
        "actual_model_training_claim":False
    }),flush=True)


if __name__=="__main__":
    run()
