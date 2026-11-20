#!/usr/bin/env python3
"""Three CPU-only incremental ES->EN experiments on LICENSE-COMPATIBLE weights.

Unlike the previously retained 15 parallel NC research rounds, these are
serial training continuations: accepted development-only winners become the
warm-start model for the next round. Failed dev gates retain the parent.
The final human-adequacy gate is NOT evaluated by this script.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import resource
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments" / p) for p in (
    "cpu_continuation_20261117", "massive_20261117",
)]
import train as core
import train_massive as massive

START = "experiments/massive_stage3_20261117/saved/spanish/adapter_selected.safetensors"
START_SHA = "14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b"
LANGUAGE = "spanish"
SOURCE = "AmazonScience/massive"
REV = "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
UPDATES = 64
COUNT = UPDATES * 16
DEV = 128
EXPERIMENTS = (
    {"name": "quality_screen", "seed": 2026112001, "lr": 2e-5,
     "max_ratio": 2.5, "noise_fraction": 0.0},
    {"name": "strict_alignment", "seed": 2026112002, "lr": 1.5e-5,
     "max_ratio": 1.65, "noise_fraction": 0.0},
    {"name": "asr_punctuation", "seed": 2026112003, "lr": 1e-5,
     "max_ratio": 2.0, "noise_fraction": 0.33},
)
NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
DIGITS = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")


def norm(t):
    return massive.canonical(t)


def sha_pairs(rows):
    return hashlib.sha256("\n".join(norm(a)+"\0"+norm(b) for a,b in rows).encode()).hexdigest()


def no_punctuation(text):
    return " ".join(NON_WORD.sub(" ", text).split())


def aligned(locale: str, split: str):
    """Read only the official author TRAIN or VALIDATION split, NEVER TEST."""
    if split not in {"train", "validation"} or locale not in {"en-US", "es-ES"}:
        raise ValueError("Invalid source split or locale")
    from datasets import load_dataset
    return [dict(x) for x in load_dataset(
        SOURCE, locale, revision=REV, split=split, trust_remote_code=True)]


def vetted_rows(es, en, max_ratio: float):
    """Publisher-ID align; conservative heuristics, not proof of semantic match."""
    a = {str(r["id"]): r for r in es}
    b = {str(r["id"]): r for r in en}
    if len(a) != len(es) or len(b) != len(en) or set(a) != set(b):
        raise RuntimeError("Publisher language rows differ in ID")
    kept = []
    stats = {"total":len(a), "missing":0,"intent_mismatch":0, "language_or_ratio":0,
             "quantity_mismatch":0, "duplicate":0}
    seen_s,seen_t = set(),set()
    for key in sorted(a):
        src,tgt = a[key],b[key]
        if src.get("intent") != tgt.get("intent"):
            stats["intent_mismatch"]+=1
            continue
        s,t = str(src.get("utt","")).strip(),str(tgt.get("utt","")).strip()
        if not s or not t:
            stats["missing"]+=1
            continue
        if (not 2<=len(s.split())<=35 or not 2<=len(t.split())<=36
            or not 1/max_ratio <= len(s)/len(t) <= max_ratio
            or norm(s)==norm(t)):
            stats["language_or_ratio"]+=1
            continue
        src_nums,tgt_nums=DIGITS.findall(s),DIGITS.findall(t)
        if src_nums and tgt_nums and src_nums!=tgt_nums:
            stats["quantity_mismatch"]+=1
            continue
        if norm(s) in seen_s or norm(t) in seen_t:
            stats["duplicate"]+=1
            continue
        seen_s.add(norm(s));seen_t.add(norm(t))
        kept.append((key,s,t))
    stats["kept"]=len(kept)
    return kept,stats


def build_data(experiment, en_train,es_train,en_dev,es_dev):
    # Use the SAME frozen dev corpus in all three rounds; never reuse it in train.
    dev_rows,_=vetted_rows(es_dev,en_dev,2.5)
    dev_rows=sorted(dev_rows,key=lambda x:hashlib.sha256(("v2dev"+x[0]).encode()).hexdigest())[:DEV]
    if len(dev_rows)<DEV:
        raise RuntimeError("Insufficient official validation examples")
    dev=[(s,t) for _,s,t in dev_rows]
    train_rows,stats=vetted_rows(es_train,en_train,experiment["max_ratio"])
    forbidden_s={norm(s) for s,t in dev}
    forbidden_t={norm(t) for s,t in dev}
    train_rows=[r for r in train_rows if norm(r[1]) not in forbidden_s and
                norm(r[2]) not in forbidden_t]
    rng=random.Random(experiment["seed"])
    rng.shuffle(train_rows)
    selected=train_rows[:COUNT]
    if len(selected)<COUNT:
        raise RuntimeError("Not enough train-only examples after filtering")
    output=[]
    for i,(_,s,t) in enumerate(selected):
        # Punctuation and casing reduction only; never alter reference meaning.
        if rng.random()<experiment["noise_fraction"]:
            s=no_punctuation(s)
        output.append((s,t))
    rng.shuffle(output)
    if {norm(s) for s,_ in output}&forbidden_s or {norm(t) for _,t in output}&forbidden_t:
        raise RuntimeError("Training and development text collision")
    if len({norm(s) for s,_ in output})!=COUNT or len({norm(t) for _,t in output})!=COUNT:
        raise RuntimeError("Duplicate train source/target after preprocessing")
    return output,dev,{"train_filter":stats,"train_count":len(output),
                       "dev_count":len(dev),"training_sha256":sha_pairs(output),
                       "dev_sha256":sha_pairs(dev),
                       "test_split_never_opened":True,
                       "publisher_alignment_not_independently_human_validated":True,
                       "source":SOURCE,"revision":REV,"license":"CC-BY-4.0",
                       "max_char_ratio":experiment["max_ratio"],
                       "noise_fraction":experiment["noise_fraction"],
                       "augmentation":"punctuation-only no target mutation",
                       "no_slang_or_multi_turn_ground_truth_claim":True}


def run():
    warm_path=ROOT/START
    if core.sha256(warm_path)!=START_SHA:
        raise RuntimeError("License-compatible stage-three starting adapter changed")
    output_root=ROOT/"experiments/es_en_v2_runs"
    output_root.mkdir(parents=True,exist_ok=False)
    en_train=aligned("en-US","train")
    es_train=aligned("es-ES","train")
    en_dev=aligned("en-US","validation")
    es_dev=aligned("es-ES","validation")
    winner=START
    winner_sha=START_SHA
    overall=[]
    begin=time.monotonic()
    for number,options in enumerate(EXPERIMENTS,1):
        train,dev,manifest=build_data(options,en_train,es_train,en_dev,es_dev)
        # Source data is in memory, never written to the public repository.
        # A temporary local JSON file is removed after each train call.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix="bridge-v2-allowed-corpus-") as scratch:
            data=Path(scratch)/"train.json"
            data.write_text(json.dumps({"train":train,"dev":dev},ensure_ascii=False))
            core.LANGUAGES[LANGUAGE]={
                **core.LANGUAGES[LANGUAGE],
                "archive_name":"commercial-candidate-MASSIVE-train-only",
                "data_url":data.as_uri(),
                "warm_adapter_override":{"path":winner,"sha256":winner_sha},
            }
            core.SEED=options["seed"]
            core.TRAIN_COUNT=COUNT
            core.DEV_COUNT=DEV
            core.TOTAL_UPDATES=UPDATES
            core.CHECKPOINTS=(32,64)
            core.LEARNING_RATE=options["lr"]
            core.LENGTH_BUCKET_SIZE=64

            def parser(blob: bytes,*,archive_name: str):
                if archive_name!="commercial-candidate-MASSIVE-train-only":
                    raise RuntimeError("Unknown dataset")
                r=json.loads(blob)
                if (sha_pairs(r["train"])!=manifest["training_sha256"] or
                    sha_pairs(r["dev"])!=manifest["dev_sha256"]):
                    raise RuntimeError("Prepared data integrity mismatch")
                return r["train"],r["dev"],COUNT+DEV

            core.parse_pairs=parser
            out=output_root/f"round-{number}"
            result=core.train_experiment(ROOT,out,LANGUAGE)
            if result["completed_updates"]!=UPDATES or result["warm_adapter_sha256"]!=winner_sha:
                raise RuntimeError("Training failed or parent weights were reset")
            # No FINAL human semantic evaluation in this workflow.
            dev_accepted=result["selected"]!="previous"
            result.update({
                "round":number,"experimental_method":options,
                "dataset_manifest":manifest,
                "human_semantic_pass_rate":None,
                "production_model_promoted":False,
                "research_development_gate":dev_accepted,
                "parent_checkpoint_sha256":winner_sha,
                "cpu_user_time_seconds":resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                "max_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "paid_spend_usd":0.0,
            })
            (out/"report.json").write_text(json.dumps(result,indent=2))
            if dev_accepted:
                winner=str((out/"candidate_selected.safetensors").relative_to(ROOT))
                winner_sha=core.sha256(ROOT/winner)
            overall.append({
                "round":number,"hypothesis":options["name"],
                "parent":result["parent_checkpoint_sha256"],
                "accepted_on_dev_only":dev_accepted,
                "selected":result["selected"],
                "dev_scores":result["scores"],
                "elapsed_seconds":result["elapsed_seconds"],
                "optimizer_updates":result["completed_updates"],
                "result_checkpoint_sha256":winner_sha,
                "production_approval":"blocked_pending_independent_human_evaluation",
            })
            print("V2_COMPLETED_REAL_ROUND",json.dumps(overall[-1]),flush=True)
    final={
        "status":"completed_three_serial_real_training_rounds",
        "origin_adapter_sha256":START_SHA,
        "final_dev_candidate_sha256":winner_sha,
        "total_new_optimizer_updates":UPDATES*len(EXPERIMENTS),
        "total_elapsed_seconds":time.monotonic()-begin,
        "successful_rounds":len(overall),"results":overall,
        "license":"CC-BY-4.0 MASSIVE / Apache-2.0 Marian + historical attributed Tatoeba",
        "human_95_percent_accuracy_verified":False,
        "commercial_default_activation":False,
        "warning":"MASSIVE is intent-oriented short speech, NOT realistic multi-turn slang. "
                  "These runs do not meet real-world contextual accuracy gate.",
    }
    (output_root/"summary.json").write_text(json.dumps(final,indent=2))
    print("V2_THREE_SERIAL_TRAINING_RESULTS",json.dumps(final),flush=True)


if __name__=="__main__":
    run()
