#!/usr/bin/env python3
"""Mine actual Apache-2.0 multilingual dialogue WITHOUT fake translations.

OpenAssistant OASST1 is authentic cross-lingual conversation *text*, not
sentence-aligned bilingual reference translations. Extract past-only slang/
pronoun/context candidates for future independent bilingual rating. Never
train on, fake references for, or publish user-supplied private transcripts.
The limited candidate bundle is an ephemeral CI artifact; only aggregate
provenance, counts, hashes are committed to the repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"research/conversation_growth/openassistant_mining_v2"
PII=re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|https?://\S+|(?:\+?[0-9][ \-()]*){9,}",re.I)
SLANG={
    "en":("i'm down","im down","slaps","ghosted","left on read",
          "spill the tea","no cap","cooked","dead","bailed"),
    "es":("guagua","chamba","me dejó en visto","me dejo en visto",
          "me echo la mano","me echó la mano","lana","qué onda",
          "que onda","ni de broma","qué fuerte"),
    "fr":("c'est ouf","trop ouf","poser un lapin","posé un lapin",
          "chelou","j'suis","j'sais","bail","grave","frérot"),
}
CONTEXT={
    "en":("he ","she ","they ","that one","which one","actually","wait,",
          "no,","i meant","you mean","the other"),
    "es":("ella ","él ","ellos ","esa ","eso ","me refiero",
          "espera","quería decir","digo","en realidad"),
    "fr":("elle ","il ","eux ","celle ","celui ","je voulais dire",
          "attends","en fait","pardon","l'autre"),
}


def hashlib_records(rows):
    return hashlib.sha256(
        "\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def analyze_rows(rows,*,source_sha:str="offline-test"):
    keep={}
    for row in rows:
        lang=str(row.get("lang") or "").lower().split("-")[0]
        if lang not in SLANG or row.get("deleted") is True:
            continue
        ident=str(row.get("message_id") or "")
        tree=str(row.get("message_tree_id") or "")
        parent=row.get("parent_id")
        content=str(row.get("text") or "").strip()
        if not ident or not tree or not (8<=len(content)<=380):
            continue
        if PII.search(content) or "\x00" in content:
            continue
        keep[ident]={"id":ident,"tree":tree,"parent":str(parent) if parent else None,
                     "text":content,"lang":lang,"role":str(row.get("role",""))}
    candidates=[]
    for msg in keep.values():
        previous=[]
        seen={msg["id"]}
        cursor=msg["parent"]
        while cursor and cursor in keep and len(previous)<4:
            prior=keep[cursor]
            if prior["tree"]!=msg["tree"] or prior["lang"]!=msg["lang"]:
                break
            if prior["id"] in seen:
                raise RuntimeError("Cycle in conversation parent chain")
            seen.add(prior["id"])
            previous.append(prior["text"])
            cursor=prior["parent"]
        if not previous:
            continue
        previous=list(reversed(previous))
        lang=msg["lang"]
        case=msg["text"].casefold()
        tags=[]
        if any(term in case for term in SLANG[lang]):
            tags.append("slang_or_regional")
        if any(term in case for term in CONTEXT[lang]):
            tags.append("reference_or_correction")
        if len(previous)>=2:
            tags.append("multi_turn")
        if not tags:
            continue
        if PII.search(" ".join(previous)):
            continue
        candidates.append({
            "conversation_id":msg["tree"],
            "turn_id":msg["id"],
            "source_language":lang,
            "speaker":msg["role"],
            "previous_utterances":previous,
            "current_text":msg["text"],
            "tags":tags,
            "reference_translation":None,
            "review_status":"unverified_no_bilingual_target",
            "license":"Apache-2.0",
            "dataset":"OpenAssistant/oasst1",
            "eligible_for_training":False,
        })
    # 2026 WMT work on targeted contextual slang highlights rarity.
    # Allocate bounded review slots across all three languages and
    # prefer slang-rich cases, then reference/correction and multi-turn.
    # Keep ONE example per conversation so evaluations do not leak turns.
    unique={}
    for item in sorted(candidates,key=lambda it:(
        0 if "slang_or_regional" in it["tags"] else 1,
        0 if "reference_or_correction" in it["tags"] else 1,
        hashlib.sha256(it["turn_id"].encode()).hexdigest()
    )):
        unique.setdefault(item["conversation_id"],item)
    available=Counter(item["source_language"] for item in unique.values())
    bins={lang:[] for lang in ("en","es","fr")}
    for item in unique.values():
        bins[item["source_language"]].append(item)
    examples=[]
    selected_ids=set()
    for lang in ("en","es","fr"):
        for item in bins[lang][:30]:
            examples.append(item)
            selected_ids.add(item["turn_id"])
    # Fill any deficit without discarding rare languages or slang.
    for item in unique.values():
        if len(examples)>=90:
            break
        if item["turn_id"] not in selected_ids:
            examples.append(item)
            selected_ids.add(item["turn_id"])
    stats=Counter()
    for row in examples:
        stats[row["source_language"]]+=1
        for tag in row["tags"]:
            stats[tag]+=1
    report={
        "status":"real_openassistant_context_candidate_mining_no_parallel_gold",
        "dataset":"OpenAssistant/oasst1",
        "source_dataset_sha":source_sha,
        "license":"Apache-2.0",
        "original_message_count":len(rows),
        "valid_nonprivate_message_count":len(keep),
        "candidate_unique_conversations":len(unique),
        "eligible_candidates_before_sampling_by_language":dict(sorted(available.items())),
        "sampling_policy":"prioritize_slang_then_reference_and_equalize_en_es_fr_review_slots",
        "candidate_for_human_bilingual_review":len(examples),
        "language_and_focus_counts":dict(sorted(stats.items())),
        "sample_ids_sha256":hashlib_records([x["turn_id"] for x in examples]),
        "future_dialogue_in_context":False,
        "bilingual_targets_available":False,
        "new_verified_parallel_training_examples":0,
        "training_updates_executed":0,
        "the_500_reviewed_utterances_used_as_training":False,
        "human_95_percent_meaning_accuracy_verified":False,
        "license_and_review_required_before_any_training":True,
    }
    return examples,report


def run():
    if OUT.exists():
        raise FileExistsError("Do not overwrite context-mining evidence")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    start=time.monotonic()
    info=HfApi().dataset_info("OpenAssistant/oasst1")
    lic=(info.cardData or {}).get("license","")
    if lic!="apache-2.0" or len(info.sha)!=40:
        raise RuntimeError("OASST1 publisher revision/license could not be verified")
    records=list(load_dataset("OpenAssistant/oasst1",split="train",
                              revision=info.sha,trust_remote_code=False))
    examples,summary=analyze_rows(records,source_sha=info.sha)
    summary["elapsed_seconds"]=time.monotonic()-start
    if summary["valid_nonprivate_message_count"]<1000:
        raise RuntimeError("Too few source dialogue messages; check dataset schema")
    OUT.mkdir(parents=True)
    (OUT/"manifest.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    # Only these candidate texts go to *ephemeral Actions artifact* for
    # qualified bilingual reviewer access; never to the git working tree.
    temp=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"bridgecaption-oasst1-review"
    temp.mkdir(parents=True,exist_ok=True)
    (temp/"bilingual_review_candidates.jsonl").write_text(
        "\n".join(json.dumps(x,ensure_ascii=False) for x in examples)+"\n")
    print("REALLY_MINED_LICENSED_MULTILINGUAL_DIALOGUE",
          json.dumps(summary),flush=True)


if __name__=="__main__":
    run()
