"""Train/dev allocation for genuine licensed EN/ES/FR parallel sentences.

Only the authenticated publisher TRAIN-equivalent open OPUS Tatoeba corpus is
used. The new validation group is outside ALL earlier fine-tuning prefixes
(0:350), and is removed from training across ALL language pairs by both
normalized source and normalized target strings. It remains development-only;
the pretrained foundation may still have encountered Tatoeba before release.
No final holdout, unlicensed dialogues, or user-reviewed original 500 used.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from research.conversation_growth.data import (
    CONFIG as OLD_CONFIG, download, pairs_from_zip, sample_pairs,
    get_reviewed_sources, load_groups, normalize, ROOT
)

CFG = ROOT / "research/conversation_growth/config_round3.json"
ORIGINAL_RECEIPT = ROOT / "research/conversation_growth/experiment_001/data_manifest.json"
PAIRS = ("en-es", "en-fr", "es-fr")


def settings():
    spec = json.loads(CFG.read_text())
    old = json.loads(OLD_CONFIG.read_text())
    dataset = spec["data"]
    training = spec["train"]
    policy = spec["restrictions"]
    if (spec["schema_version"] != 1 or
        spec["model"]["id"] != "facebook/m2m100_418M" or
        spec["parent"]["rank"] != 4 or spec["parent"]["alpha"] != 8 or
        dataset["license"] != "CC-BY-2.0-FR" or
        dataset["source"] != "OPUS-Tatoeba-v2023-04-12" or
        dataset["download_pairs"] != 1500 or
        dataset["past_training_prefix"] != 350 or
        dataset["training_upper_index"] != 1200 or
        dataset["dev_per_corpus"] != 20 or
        dataset["training_sample_use_limit_per_corpus"] != 800 or
        dataset["frozen_massive_validation_per_route"] != 36 or
        training["updates_per_direction"] * 6 != training["total_updates"] or
        training["batch_size"] != 2 or
        not all(policy.get(x) is True for x in (
            "no_paid_compute","no_final_holdout",
            "no_reviewed_500_training_or_validation",
            "no_MASSIVE_intent_as_literal_gold",
            "do_not_modify_original_or_round2_checkpoints",
            "no_production_changes","no_independent_human_adequacy_claim"
        )) or not dataset["no_multiturn_claim"] or
        old["training"]["seed"] != 2026112143):
        raise RuntimeError("Unsafe or unapproved round 3 configuration")
    return spec, old


def normalized_keys(text: str) -> tuple[str,str]:
    value = normalize(text)
    return value, re.sub(r"[\W_]+", "", value)


def identities(row) -> set[str]:
    return {x for field in ("source","target")
            for x in normalized_keys(row[field]) if x}


def split_samples(corpora: dict, *, prefix=350, upper=1200, target_train=800, target_dev=20):
    """Globally disjoint across pair boundaries, forward AND reverse paths."""
    if set(corpora) != set(PAIRS):
        raise RuntimeError("Expected three authentic parallel source sets")
    seen = set()
    training = {}
    for pair in PAIRS:
        rows = corpora[pair]
        if len(rows) < upper + target_dev:
            raise RuntimeError("Insufficient publisher corpus after legal quality filter")
        eligible = []
        for row in rows[:upper]:
            keys = identities(row)
            if keys & seen:
                continue
            seen.update(keys)
            eligible.append(row)
            if len(eligible) >= target_train:
                break
        if len(eligible) < target_train:
            raise RuntimeError("Insufficient globally distinct train examples: "+pair)
        training[pair] = eligible
        # Reserve ALL earlier model fine-tuning examples from validation.
        seen.update(k for row in rows[:prefix] for k in identities(row))
    dev = {}
    for pair in PAIRS:
        picked = []
        # New source data is disjoint from earlier 350-item fine-tuning input.
        for row in corpora[pair][upper:]:
            keys = identities(row)
            if keys & seen:
                continue
            seen.update(keys)
            picked.append(row)
            if len(picked) == target_dev:
                break
        if len(picked) != target_dev:
            raise RuntimeError("Insufficient globally distinct newly held-out DEV: "+pair)
        dev[pair] = picked

    all_training = {k for rows in training.values() for row in rows for k in identities(row)}
    all_dev = {k for rows in dev.values() for row in rows for k in identities(row)}
    previous = {k for pair in PAIRS for row in corpora[pair][:prefix] for k in identities(row)}
    if all_training & all_dev or all_dev & previous:
        raise RuntimeError("DEV leakage from training or previous rounds")
    return training,dev


def directions(pairs: dict) -> dict:
    result={}
    for pair in PAIRS:
        a,b=pair.split("-")
        samples=pairs[pair]
        result[pair]=[{"src":row["source"],"tgt":row["target"],
                       "group_id":pair+":"+str(row["row_id"])} for row in samples]
        result[b+"-"+a]=[{"src":row["target"],"tgt":row["source"],
                          "group_id":pair+":"+str(row["row_id"])} for row in samples]
    if len(result)!=6:
        raise RuntimeError("Invalid six-route parallel development set")
    return result


def build_data(*, downloader=download):
    spec,oldcfg=settings()
    previous=json.loads(ORIGINAL_RECEIPT.read_text())
    excluded=get_reviewed_sources()
    official_dev=load_groups("validation")
    for group in official_dev:
        for language in ("en","es","fr"):
            excluded.add(normalize(group[language]))
    # Near-copy variants must be excluded across all languages as well.
    forbidden={key for value in excluded for key in normalized_keys(value)}
    result={}
    provenance={}
    for source in oldcfg["parallel_sources"]:
        pair=source["pair"]
        if pair not in PAIRS or source["license"]!="CC-BY-2.0-FR":
            raise RuntimeError("Unapproved corpus or license")
        failure=[]
        for url in source["download_urls"]:
            try:
                archive=downloader(url)
                digest=hashlib.sha256(archive).hexdigest()
                if digest != previous["sources"][pair]["archive_sha256"]:
                    raise RuntimeError("Pinned publisher archive digest mismatch")
                raw=pairs_from_zip(archive,pair)
                sample=sample_pairs(raw,pair,spec["data"]["download_pairs"],
                                    forbidden,oldcfg["training"]["seed"])
                if len(sample) != spec["data"]["download_pairs"]:
                    raise RuntimeError("Insufficient new parallel pairs")
                result[pair]=sample
                provenance[pair]={
                    "publisher":"Tatoeba via OPUS v2023-04-12",
                    "archive_sha256":digest,
                    "url":url,
                    "license":"CC-BY-2.0-FR",
                    "accepted":len(sample),
                    "first_350_are_already_seen":True,
                    "no_new_publisher_test_split_exists":True
                }
                break
            except (RuntimeError,ValueError,OSError,TimeoutError) as exc:
                failure.append(type(exc).__name__+": "+str(exc)[:180])
        if pair not in result:
            raise RuntimeError("Cannot verify pinned OPUS Tatoeba "+pair+"; "+str(failure))
    train,dev=split_samples(
        result,prefix=spec["data"]["past_training_prefix"],
        upper=spec["data"]["training_upper_index"],
        target_train=spec["data"]["training_sample_use_limit_per_corpus"],
        target_dev=spec["data"]["dev_per_corpus"])
    # Exclude reviewed originals and all publisher MASSIVE validation by string.
    for corpus in (train,dev):
        for group in corpus.values():
            for row in group:
                if identities(row) & forbidden:
                    raise RuntimeError("Forbidden reviewed / publisher-validation string leaked")
    manifest={
        "sources":provenance,"license":"CC-BY-2.0-FR",
        "train_samples_by_pair":{p:len(v) for p,v in train.items()},
        "dev_samples_by_pair":{p:len(v) for p,v in dev.items()},
        "new_dev_outside_original_350":True,
        "train_dev_cross_pair_disjoint_by_text":True,
        "old_500_reviewed_not_used":True,
        "no_MASSIVE_as_parallel_gold":True,
        "no_multiturn_data":True,
        "no_final_holdout":True,
        "pretraining_overlap_with_tatoeba_not_ruled_out":True,
        "train_ids_digest_sha256":hashlib.sha256(
            json.dumps({p:[r["row_id"] for r in rows] for p,rows in train.items()},sort_keys=True).encode()
        ).hexdigest(),
        "dev_ids_digest_sha256":hashlib.sha256(
            json.dumps({p:[r["row_id"] for r in rows] for p,rows in dev.items()},sort_keys=True).encode()
        ).hexdigest()
    }
    return train,directions(dev),manifest
