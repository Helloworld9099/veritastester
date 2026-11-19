#!/usr/bin/env python3
"""ACTUAL research-only French LoRA training with genuine chat turn context.

WMT24 CHAT is licensed CC BY-NC 4.0. Research weights are NONCOMMERCIAL and
MUST NOT be integrated into the production application or used commercially.

Only official train + validation are downloaded here. The official test and
its human references are protected by FROZEN_TEST_MANIFEST.json and NEVER
opened for optimizer updates, selection, or diagnostic scoring in this run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[
    str(ROOT/"experiments/cpu_continuation_20261117"),
    str(ROOT/"experiments/opus_books_20261117"),
    str(ROOT/"experiments/balanced_stage5_20261118"),
    str(ROOT/"experiments/massive_20261117"),
]
import train as core
import train_stage5 as balanced
import train_massive as massive
from train_books import evaluate_frozen_test

LANGUAGE="french"
SOURCE_REV="12a34149267dd9ae98e60e0ec2e415707fad5618"
BASE_URL="https://raw.githubusercontent.com/WMT-Chat-task/chat-task-2024-data/"+SOURCE_REV
SOURCE_FILES={
    "train":("train/en-fr.csv","f5b10236f9aa4f235239b1004e380aec2abc4f37"),
    "dev":("valid/en-fr.csv","8c9d0a051de662fd43a50365ee11c4d968507a47"),
}
INPUT_FIELDS=("source_language","target_language","source","reference",
              "doc_id","client_id","sender","tags")
SEED=2026111901
UPDATES=240
TRAIN_CHAT=3072
TRAIN_MASSIVE_REPLAY=768
TRAIN_N=TRAIN_CHAT+TRAIN_MASSIVE_REPLAY
DEV_N=256
WARM_REL="experiments/tatoeba_stage7_20261118/saved/adapter_selected.safetensors"
WARM_SHA="5b9b136997fcc80c961542831a1dd0578018de81f1e1a069df686616a4f5d80f"
SELECTION_STEPS=(60,120,180,240)


def canonical(text:str)->str:
    return " ".join(text.casefold().strip().split())


def file_digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def parse_chat(content:bytes):
    reader=csv.DictReader(io.StringIO(content.decode("utf-8-sig")),
                          delimiter=",",quotechar='"')
    fields=tuple(reader.fieldnames or ())
    # The official TRAIN file has 7 columns; VALID adds an eighth "tags"
    # field with discourse-phenomena metadata. Both versions are pinned
    # to their exact Git blob SHAs above, so no arbitrary schema is allowed.
    if fields not in (INPUT_FIELDS, INPUT_FIELDS[:-1]):
        raise ValueError("Unexpected original WMT24 chat train/validation column schema")
    parsed=[]
    for row in reader:
        if not isinstance(row,dict) or None in row:
            raise ValueError("Malformed chat CSV record")
        if (row.get("source_language"),row.get("target_language")) not in (
                ("fr","en"),("en","fr")):
            raise ValueError("Unexpected original language direction")
        if (not row.get("doc_id") or not row.get("source") or
                not row.get("reference") or row.get("sender") not in ("customer","agent")):
            raise ValueError("Missing WMT24 speaker, source, reference or conversation ID")
        parsed.append(row)
    if len(parsed)<2500:
        raise ValueError("WMT24 chat corpus unexpectedly small")
    return parsed


def add_context(rows, *, topic="customer support"):
    """Current French text FIRST to survive truncation; two genuine prior turns."""
    history={}
    output=[]
    for row in rows:
        dialogue_id=row["doc_id"]
        previous=history.setdefault(dialogue_id,[])
        if row["source_language"]=="fr":
            fr,english=row["source"].strip(),row["reference"].strip()
            sw,tw=len(fr.split()),len(english.split())
            if (2<=sw<=45 and 2<=tw<=56 and
                    0.45<=len(fr)/max(len(english),1)<=2.3 and
                    canonical(fr)!=canonical(english)):
                recent=[r[:95] for r in previous[-2:] if r.strip()]
                context=(" | contexte : "+" / ".join(recent)) if recent else ""
                output.append({
                    "doc_id":dialogue_id,"source":fr,"reference":english,
                    "input_with_context":fr+context,
                    "prior_turns":len(recent),
                    "tags":row.get("tags","[]"),
                })
        # No gold reference or future utterance is used as conversation history.
        # The actual previous source, in whichever participant's original
        # language, is what was available before this turn.
        previous.append(row["source"].strip())
    return output


def select_group_aware(rows:list[dict], count:int, seed:int):
    """Choose one turn per conversation first, then extra turns if needed."""
    by_doc={}
    for r in rows:by_doc.setdefault(r["doc_id"],[]).append(r)
    docs=sorted(by_doc,key=lambda doc:hashlib.sha256(
        (str(seed)+"|"+doc).encode()).hexdigest())
    rng=random.Random(seed)
    for doc in docs:rng.shuffle(by_doc[doc])
    selected=[]
    seen_source,seen_english=set(),set()
    for offset in range(max(map(len,by_doc.values()),default=0)):
        for doc in docs:
            entries=by_doc[doc]
            if offset>=len(entries):continue
            row=entries[offset]
            src,ref=canonical(row["source"]),canonical(row["reference"])
            if src in seen_source or ref in seen_english:continue
            seen_source.add(src);seen_english.add(ref)
            selected.append(row)
            if len(selected)>=count:return selected
    raise ValueError(f"Only {len(selected)} distinct WMT24 source/ref pairs, need {count}")


def download_frozen(split:str):
    name,blob_sha=SOURCE_FILES[split]
    with urlopen(Request(BASE_URL+"/"+name,
                         headers={"User-Agent":"BridgeCaption-research-noncommercial/1.0"}),
                 timeout=60) as stream:
        raw=stream.read(4_000_000+1)
    if len(raw)>4_000_000 or not raw:
        raise RuntimeError("WMT24 source unexpectedly changed size")
    actual_git_blob=hashlib.sha1(
        ("blob "+str(len(raw))).encode()+b"\0"+raw).hexdigest()
    if actual_git_blob!=blob_sha:
        raise RuntimeError("Publisher WMT24 dataset Git blob changed: "+name)
    return raw


def make_training_cohorts():
    raw_train=download_frozen("train")
    raw_dev=download_frozen("dev")
    original_train=parse_chat(raw_train)
    original_dev=parse_chat(raw_dev)
    train_groups={r["doc_id"] for r in original_train}
    dev_groups={r["doc_id"] for r in original_dev}
    if train_groups & dev_groups:
        raise RuntimeError("WMT24 official conversation groups leak between train and validation")
    train_items=add_context(original_train)
    dev_items=add_context(original_dev)
    validation=select_group_aware(dev_items,DEV_N,SEED+1)
    dev_source={canonical(row["source"]) for row in validation}
    dev_target={canonical(row["reference"]) for row in validation}
    filtered_train=[
        row for row in train_items
        if canonical(row["source"]) not in dev_source and
           canonical(row["reference"]) not in dev_target
    ]
    selected_train=select_group_aware(filtered_train,TRAIN_CHAT,SEED)
    # This copy is a historic official MASSIVE TRAIN ONLY, never a test or dev.
    english=massive.fetch_multipart("en-US")
    french=massive.fetch_multipart("fr-FR")
    prior=massive.select_pairs(english,french,include_ids=True)
    original=json.loads((ROOT/"experiments/massive_20261117/saved/french/"
                         "massive_evaluation.json").read_text())
    if prior["ids_sha256"]!=original["split_ids_sha256"]:
        raise RuntimeError("MASSIVE consumed historical cohort hash mismatch")
    forbidden=[(row["source"],row["reference"])
               for row in selected_train+validation]
    forbidden+=prior["test"]+prior["dev"]
    rehearsal=balanced.pick_replay(prior["train"],forbidden,
                                   count=TRAIN_MASSIVE_REPLAY,seed=SEED)
    pairs=[(row["input_with_context"],row["reference"]) for row in selected_train]
    pairs+=rehearsal
    rng=random.Random(SEED)
    rng.shuffle(pairs)
    dev_ctx=[(row["input_with_context"],row["reference"]) for row in validation]
    dev_raw=[(row["source"],row["reference"]) for row in validation]
    if (len(pairs),len(dev_ctx))!=(TRAIN_N,DEV_N):
        raise RuntimeError("Frozen WMT chat training size changed")
    if set(train_groups)&set(dev_groups):
        raise RuntimeError("WMT split contamination")
    metadata={
        "publisher":"WMT24 Chat Translation (Unbabel)",
        "license":"CC BY-NC 4.0: research only, commercial deployment PROHIBITED",
        "source_commit":SOURCE_REV,
        "source_git_blob_SHA1":{k:v[1] for k,v in SOURCE_FILES.items()},
        "source_SHA256":{"train":file_digest(raw_train),"dev":file_digest(raw_dev)},
        "official_train_conversations":len(train_groups),
        "official_validation_conversations":len(dev_groups),
        "selected_chat_training_n":len(selected_train),
        "selected_history_dependent_training_n":sum(
            bool(r["prior_turns"]) for r in selected_train),
        "selected_chat_dev_n":len(validation),
        "selected_history_dependent_dev_n":sum(
            bool(r["prior_turns"]) for r in validation),
        "distinct_training_groups":len({r["doc_id"] for r in selected_train}),
        "distinct_dev_groups":len({r["doc_id"] for r in validation}),
        "old_MASSIVE_TRAIN_replay_n":len(rehearsal),
        "warm_stage7_adapter_SHA256":WARM_SHA,
        "strict_no_train_dev_overlap_by_conversation_id":True,
        "WMT24_TEST_source_or_human_references_downloaded_or_scored":False,
        "context_format":"current French text first, then | contexte : last up to two previous original speaker turns",
        "context_note":"Contexts may contain genuine English and French original speaker messages; gold translations were NEVER supplied as context.",
        "research_only":True
    }
    return pairs,dev_ctx,dev_raw,metadata


def run(output:Path):
    if output.exists():
        raise FileExistsError("Refusing to overwrite any older research checkpoint")
    warm=ROOT/WARM_REL
    if core.sha256(warm)!=WARM_SHA:
        raise RuntimeError("Protected Stage7 French selected adapter changed")
    train,dev,dev_plain,manifest=make_training_cohorts()
    print("REAL_CONTEXTUAL_WMT_TRAIN_MANIFEST",json.dumps(manifest),flush=True)
    with tempfile.TemporaryDirectory(prefix="BridgeCaption-WMT24-context-") as tmp:
        prep=Path(tmp)/"train-dev.json"
        prep.write_text(json.dumps({"train":train,"dev":dev},ensure_ascii=False))
        spec=core.LANGUAGES[LANGUAGE].copy()
        core.LANGUAGES[LANGUAGE]={
            **spec,"data_url":prep.as_uri(),
            "archive_name":"research-only-WMT24-chat",
            "warm_adapter_override":{"path":WARM_REL,"sha256":WARM_SHA}
        }
        core.SEED=SEED
        core.TRAIN_COUNT=TRAIN_N
        core.DEV_COUNT=DEV_N
        core.TOTAL_UPDATES=UPDATES
        core.CHECKPOINTS=SELECTION_STEPS
        def parse(payload:bytes,*,archive_name:str):
            if archive_name!="research-only-WMT24-chat":
                raise RuntimeError("Wrong corpus archive")
            sample=json.loads(payload)
            if len(sample["train"])!=TRAIN_N or len(sample["dev"])!=DEV_N:
                raise RuntimeError("Frozen data size differs")
            return sample["train"],sample["dev"],TRAIN_N+DEV_N
        core.parse_pairs=parse
        report=core.train_experiment(ROOT,output,LANGUAGE)
        report.update({
            "stage":"context-v1","source":"WMT24 official train/valid conversations",
            "research_license":"CC-BY-NC-4.0, not for commercial model use",
            "cohort_manifest":manifest,
            "WMT24_test_still_completely_sealed":True,
            "prior_speaker_turns_passed_to_model_during_training":True,
        })
        (output/"report.json").write_text(json.dumps(report,indent=2))

    # The following uses official DEVELOPMENT examples, not official TEST.
    # It is an ablation diagnostic, never a final unbiased accuracy rating.
    ablation=evaluate_frozen_test(
        LANGUAGE,output,dev_plain,report["selected"],previous_adapter=warm)
    evidence={
        "language":LANGUAGE,"completed_optimizer_updates":report["completed_updates"],
        "selected_based_on_contextual_DEVELOPMENT_only":report["selected"],
        "cohorts":manifest,
        "contextual_dev_scores":report["scores"],
        "plain_input_dev_ablation":ablation,
        "official_publisher_TEST_was_never_scored":True,
        "commercial_deployment_allowed":False,
        "90_percent_independent_human_adequacy_confirmed":False,
        "caveat":"Research-only WMT24 Chat training, genuine dialogue context in input. "
                 "No prior DiaBLa, TICO or WMT24 TEST rows were included. "
                 "No bilingual-human ratings of BridgeCaption output are provided."
    }
    (output/"context_research_evaluation.json").write_text(
        json.dumps(evidence,indent=2))
    print("REAL_WMT_CHAT_CONTEXT_TRAINING_RESULTS",json.dumps(evidence),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    options=p.parse_args()
    run(options.output_dir.resolve())
