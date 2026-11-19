#!/usr/bin/env python3
"""Run REAL free-CPU French LoRA updates on conversational Tatoeba + MASSIVE train.

All 144 public DiaBLa dialogues and original publisher Tatoeba test components
are excluded from optimizer input, development selection and test reporting.
A fresh heldout sample from previously unused Tatoeba TRAIN components is
reserved before fitting. No claim of independent human-rated accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"experiments"/name) for name in (
    "cpu_continuation_20261117", "balanced_stage5_20261118",
    "massive_20261117", "massive_stage2_20261117",
    "massive_stage3_20261117", "opus_books_20261117",
    "globalvoices_stage4_20261118")]
sys.path.insert(0,str(ROOT/"research/data_preparation/scripts"))

import prepare_data as prep
import train as core
import train_massive as massive
import train_stage5 as balanced
from train_books import evaluate_frozen_test

SEED=202611187
TRAIN_TATOEBA=2304
TRAIN_REPLAY=768
DEV_TATOEBA=256
RESERVED_TEST=256
UPDATES=192
WARM_REL="experiments/balanced_stage5_20261118/saved/french/adapter_selected.safetensors"
WARM_SHA="fba2bd78958bf0bf8a4c14dc55e601e08663633718c4c9c17f91ef9b52dcfa1c"

def norm(s):
    return " ".join(s.casefold().strip().split())

def sha_pairs(rows):
    return hashlib.sha256("\n".join(
        hashlib.sha256((a+"\0"+b).encode()).hexdigest() for a,b in rows
    ).encode()).hexdigest()

def download_pinned_sources():
    manifest=json.loads((ROOT/"research/data_preparation/sources.json").read_text())
    wanted={"fra-eng.zip","diabla_all_dialogues.json"}
    for row in manifest:
        if row["file"] not in wanted: continue
        path=prep.RAW/row["file"]
        if not path.is_file():
            with urlopen(Request(row["url"],
                        headers={"User-Agent":"BridgeCaption-research/1.0"}),
                        timeout=90) as response:
                raw=response.read(row["bytes"]+1)
            if len(raw)!=row["bytes"] or hashlib.sha256(raw).hexdigest()!=row["sha256"]:
                raise RuntimeError("Pinned dataset bytes changed: "+row["file"])
            path.write_bytes(raw)
        if path.stat().st_size!=row["bytes"] or core.sha256(path)!=row["sha256"]:
            raise RuntimeError("Pinned dataset SHA-256 mismatch: "+row["file"])
        wanted.remove(row["file"])
    if wanted: raise RuntimeError("Missing licensed pinned source files")

def choose_train_and_test(train_rows, prior_quick, protected,
                          train_n=TRAIN_TATOEBA, test_n=RESERVED_TEST):
    """Strict translation-connected COMPONENT separation incl prior 12k used."""
    old_groups={row["group_id"] for row in prior_quick}
    groups={}
    for row in train_rows:
        group=row["group_id"]
        if group in old_groups: continue
        a,b=row["utterance"].strip(),row["reference_translation"].strip()
        if not 2<=len(a.split())<=36 or not 2<=len(b.split())<=36:
            continue
        if not .45<=len(a)/max(1,len(b))<=2.3: continue
        if norm(a) in protected or norm(b) in protected:
            raise RuntimeError("DiaBLa final test sentence leaked into training pool")
        # Connected components often repeat the exact same source with
        # alternative targets. Keep ONE deterministic representative per
        # group rather than teaching repeated/contradictory translations.
        candidate=(a,b)
        if group not in groups or hashlib.sha256(str(candidate).encode()).hexdigest() < hashlib.sha256(str(groups[group][0]).encode()).hexdigest():
            groups[group]=[candidate]
    order=sorted(groups,key=lambda x:hashlib.sha256((str(SEED)+":"+x).encode()).hexdigest())
    result={}
    used=set()
    for name,needed in (("reserved_test",test_n),("train",train_n)):
        sample,keys=[],[]
        for group in order:
            if group in used or len(sample)+len(groups[group])>needed:continue
            keys.append(group)
            sample.extend(groups[group])
            used.add(group)
            if len(sample)==needed: break
        if len(sample)!=needed:
            raise ValueError("Not enough component-disjoint "+name+" examples")
        result[name]=sample
        result[name+"_manifest"]={
            "count":len(sample),"groups":len(keys),
            "group_sha256":hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(),
            "pairs_sha256":sha_pairs(sample),
        }
    balanced.assert_no_overlap(("train",result["train"]),
                               ("reserved_test",result["reserved_test"]))
    return result

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def run(output:Path):
    if output.exists():raise FileExistsError("Never overwrite a research experiment")
    warm=ROOT/WARM_REL
    if core.sha256(warm)!=WARM_SHA:
        raise RuntimeError("Saved French stage-five model changed")
    download_pinned_sources()
    _, protected=prep.prepare_diabla()
    source_audit=prep.prepare_tatoeba(protected)
    private=prep.OUT
    train_all=read_rows(private/"tatoeba.fr-en.train.jsonl")
    dev_all=read_rows(private/"tatoeba.fr-en.dev.jsonl")
    old_quick=read_rows(private/"tatoeba.fr-en.train.quick12000.jsonl")
    chosen=choose_train_and_test(train_all,old_quick,protected)
    dev=[]
    dev_groups=set()
    for r in sorted(dev_all,key=lambda v:hashlib.sha256(v["case_id"].encode()).hexdigest()):
        if r["group_id"] in dev_groups:
            continue
        a,b=r["utterance"].strip(),r["reference_translation"].strip()
        if (norm(a) in protected or norm(b) in protected
                or not 2<=len(a.split())<=36 or not 2<=len(b.split())<=36):
            continue
        dev.append((a,b))
        dev_groups.add(r["group_id"])
        if len(dev)==DEV_TATOEBA:break
    if len(dev)!=DEV_TATOEBA:raise ValueError("Missing real Tatoeba development pairs")
    en=massive.fetch_multipart("en-US")
    fr=massive.fetch_multipart("fr-FR")
    first=massive.select_pairs(en,fr,include_ids=True)
    old=json.loads((ROOT/"experiments/massive_20261117/saved/french/massive_evaluation.json").read_text())
    if first["ids_sha256"]!=old["split_ids_sha256"]:
        raise RuntimeError("MASSIVE historical split digest mismatch")
    rehearsal=balanced.pick_replay(
        first["train"],
        forbidden=(chosen["train"]+chosen["reserved_test"]+dev+
                   first["dev"]+first["test"]+[(text,text) for text in protected]),
        count=TRAIN_REPLAY,seed=SEED)
    balanced.assert_no_overlap(
        ("Tatoeba train",chosen["train"]),
        ("Tatoeba reserved",chosen["reserved_test"]),
        ("Tatoeba development",dev),
        ("prior MASSIVE TRAIN rehearsal",rehearsal))
    train=chosen["train"]+rehearsal
    random.Random(SEED).shuffle(train)
    info={
        "original_Tatoeba_component_train_manifest_sha256":source_audit["files"][0]["sha256"],
        "Tatoeba_train":chosen["train_manifest"],
        "Tatoeba_reserved_not_used_for_selection":chosen["reserved_test_manifest"],
        "Tatoeba_dev_n":len(dev),"MASSIVE_OLD_TRAIN_replay_n":len(rehearsal),
        "historical_Tatoeba_quick12000_entire_components_excluded":True,
        "DiaBLa_144_dialogues_protected_as_FINAL_EXTERNAL_TEST":True,
        "warm_model_sha256":WARM_SHA,
        "credits":"Tatoeba contributors, CC BY 2.0 FR; MASSIVE CC BY 4.0; DiaBLa CC BY-SA 4.0 (test only)",
        "limits":"Original Marian pretraining overlap unknown; not context-aware.",
    }
    print("STAGE7_COHORTS_BEFORE_TRAIN",json.dumps(info),flush=True)
    with tempfile.TemporaryDirectory(prefix="tatoeba7-") as d:
        trainfile=Path(d)/"train_dev.json"
        trainfile.write_text(json.dumps({"train":train,"dev":dev},ensure_ascii=False))
        original=core.LANGUAGES["french"].copy()
        core.LANGUAGES["french"]={
            **original,"data_url":trainfile.as_uri(),
            "archive_name":"tatoeba-conversation7",
            "warm_adapter_override":{"path":WARM_REL,"sha256":WARM_SHA}
        }
        core.SEED=SEED
        core.TRAIN_COUNT=len(train)
        core.DEV_COUNT=len(dev)
        core.TOTAL_UPDATES=UPDATES
        core.CHECKPOINTS=(48,96,144,192)
        def parse(payload:bytes,*,archive_name:str):
            if archive_name!="tatoeba-conversation7":
                raise RuntimeError("Unexpected candidate data")
            obj=json.loads(payload)
            if len(obj["train"])!=len(train) or len(obj["dev"])!=len(dev):
                raise RuntimeError("Prepared training data modified")
            return obj["train"],obj["dev"],len(train)+len(dev)
        core.parse_pairs=parse
        report=core.train_experiment(ROOT,output,"french")
        report["stage"]=7
        report["data_integrity"]=info
        report["research_only"]=True
        (output/"report.json").write_text(json.dumps(report,indent=2))
    # This test was selected BEFORE optimizer steps, from unused Tatoeba
    # TRAIN-partition connected components; it is not the FINAL DiaBLa test.
    test_scores=evaluate_frozen_test(
        "french",output,chosen["reserved_test"],report["selected"],
        previous_adapter=warm)
    summary={
        "language":"french","completed_updates":report["completed_updates"],
        "selected_using_dev_only":report["selected"],
        "frozen_source_splits":info,
        "fresh_component_disjoint_Tatoeba_diagnostic":test_scores,
        "untouched_final_DiaBLa_test_evaluated":False,
        "human_90_percent_adequacy_verified":False,
        "notes":"New Tatoeba test is disjoint from adaptation data but derives "
                "from the publisher TRAIN partition; not a final independent "
                "human adequacy assessment. Never use these consumed cases to tune.",
    }
    (output/"stage7_evaluation.json").write_text(json.dumps(summary,indent=2))
    print("STAGE7_RESULTS",json.dumps(summary),flush=True)

if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir",required=True,type=Path)
    args=cli.parse_args()
    run(args.output_dir.resolve())
