#!/usr/bin/env python3
"""Actual 512-update French LoRA continuation on NEW Tatoeba groups + MASSIVE train.

Rebuild the historical Stage7 cohort and exclude every Stage7 training + test
connected component and the original quick12k components before choosing any
new data. No prior Tatoeba publisher test, DiaBLa, or MASSIVE test is fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"experiments"/name) for name in (
    "cpu_continuation_20261117","tatoeba_stage7_20261118",
    "balanced_stage5_20261118","massive_20261117",
    "massive_stage2_20261117","massive_stage3_20261117",
    "opus_books_20261117","globalvoices_stage4_20261118")]
sys.path.insert(0,str(ROOT/"research/data_preparation/scripts"))
import train as core
import train_stage7 as stage7
import train_stage5 as balanced
import train_massive as massive
import prepare_data as prep
from train_books import evaluate_frozen_test

SEED=202611188
UPDATES=512
TATOEBA_TRAIN=6144
MASSIVE_REPLAY=2048
TATOEBA_DEV=128
MASSIVE_DEV=128
TEST_SIZE=256
WARM_REL="experiments/tatoeba_stage7_20261118/saved/adapter_selected.safetensors"
WARM_SHA="5b9b136997fcc80c961542831a1dd0578018de81f1e1a069df686616a4f5d80f"

def norm(x):
    return prep.norm(x)

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def source_cohorts():
    stage7.download_pinned_sources()
    _,protected=prep.prepare_diabla()
    audit=prep.prepare_tatoeba(protected)
    root=prep.OUT
    all_train=read_jsonl(root/"tatoeba.fr-en.train.jsonl")
    all_dev=read_jsonl(root/"tatoeba.fr-en.dev.jsonl")
    old_quick=read_jsonl(root/"tatoeba.fr-en.train.quick12000.jsonl")

    stage7.SEED=202611187
    first=stage7.choose_train_and_test(all_train,old_quick,protected,
                                       train_n=2304,test_n=256)
    saved7=json.loads((ROOT/"experiments/tatoeba_stage7_20261118/saved/"
                       "stage7_evaluation.json").read_text())
    older=saved7["frozen_source_splits"]
    for kind in ("train","reserved_test"):
        if first[kind+"_manifest"]!=older["Tatoeba_"+(
                "train" if kind=="train" else "reserved_not_used_for_selection")]:
            raise RuntimeError("Stage7 published cohort digest changed: "+kind)

    # Build exact source/reference->component map; a group must be kept whole.
    bypair={(norm(r["utterance"]),norm(r["reference_translation"])):r["group_id"]
            for r in all_train}
    prior_groups=set()
    for kind in ("train","reserved_test"):
        for s,e in first[kind]:
            key=(norm(s),norm(e))
            if key not in bypair: raise RuntimeError("Unreproducible Stage7 source")
            prior_groups.add(bypair[key])
    protected_groups=old_quick+[{"group_id":gid} for gid in sorted(prior_groups)]
    stage7.SEED=SEED
    current=stage7.choose_train_and_test(all_train,protected_groups,protected,
                                         train_n=TATOEBA_TRAIN,test_n=TEST_SIZE)
    if current["train_manifest"]["group_sha256"] in (
        first["train_manifest"]["group_sha256"],
        first["reserved_test_manifest"]["group_sha256"]):
        raise RuntimeError("Reused former training or evaluation components")

    # Development examples only; the first 256 official Tatoeba-dev groups were
    # used for Stage7 model selection and will not be used here.
    groups=set()
    freshdev=[]
    for row in sorted(all_dev,
                      key=lambda r:hashlib.sha256(r["case_id"].encode()).hexdigest()):
        gid=row["group_id"]
        if gid in groups:continue
        s,e=row["utterance"].strip(),row["reference_translation"].strip()
        if (norm(s) in protected or norm(e) in protected or
                not 2<=len(s.split())<=36 or not 2<=len(e.split())<=36):
            continue
        groups.add(gid)
        if len(groups)<=256:continue
        freshdev.append((s,e))
        if len(freshdev)==TATOEBA_DEV:break
    if len(freshdev)!=TATOEBA_DEV:
        raise RuntimeError("Insufficient unused official Tatoeba dev groups")

    english=massive.fetch_multipart("en-US")
    french=massive.fetch_multipart("fr-FR")
    old=massive.select_pairs(english,french,include_ids=True)
    prior=json.loads((ROOT/"experiments/massive_20261117/saved/french/"
                      "massive_evaluation.json").read_text())
    if old["ids_sha256"]!=prior["split_ids_sha256"]:
        raise RuntimeError("MASSIVE historical author split changed")
    dev_mass=old["dev"][:MASSIVE_DEV]  # legitimate development, NEVER trained
    used=current["train"]+current["reserved_test"]+freshdev+dev_mass
    rehearsal=balanced.pick_replay(
        old["train"],
        forbidden=(used+old["dev"]+old["test"]+
                   first["train"]+first["reserved_test"]+
                   [(x,x) for x in protected]),
        count=MASSIVE_REPLAY,seed=SEED)
    train=current["train"]+rehearsal
    random.Random(SEED).shuffle(train)
    dev=freshdev+dev_mass
    random.Random(SEED+1).shuffle(dev)
    balanced.assert_no_overlap(
        ("new Tatoeba test",current["reserved_test"]),
        ("new Tatoeba train",current["train"]),
        ("new Tatoeba dev",freshdev),
        ("MASSIVE development",dev_mass),
        ("historical MASSIVE train rehearsal",rehearsal))
    return train,dev,current["reserved_test"],{
        "stage7_train_and_test_groups_excluded":len(prior_groups),
        "historic_Tatoeba_quick12000_components_excluded":True,
        "new_tatoeba_train":current["train_manifest"],
        "new_tatoeba_frozen_test":current["reserved_test_manifest"],
        "new_tatoeba_dev_n":len(freshdev),
        "massive_prior_dev_n":len(dev_mass),
        "massive_old_TRAIN_only_replay":len(rehearsal),
        "all_144_previously_evaluated_DiaBLa_dialogues_excluded_from_training":True,
        "source_rights":"Tatoeba contributors CC BY 2.0 FR; MASSIVE CC BY 4.0",
        "source_audit":audit["accepted"],
        "warm_stage7_adapter_sha256":WARM_SHA,
    }

def run(output:Path):
    if output.exists():raise FileExistsError("Never overwrite historical trained models")
    warm=ROOT/WARM_REL
    if core.sha256(warm)!=WARM_SHA:raise RuntimeError("Stage7 selected model SHA mismatch")
    train,dev,final_test,manifest=source_cohorts()
    if (len(train),len(dev),len(final_test))!=(TATOEBA_TRAIN+MASSIVE_REPLAY,
                                                TATOEBA_DEV+MASSIVE_DEV,TEST_SIZE):
        raise RuntimeError("Stage8 frozen cohort size altered")
    print("STAGE8_FROZEN_NEW_COHORTS_BEFORE_OPTIMIZER",json.dumps(manifest),flush=True)
    with tempfile.TemporaryDirectory(prefix="bridgecaption-tatoeba8-") as tmp:
        path=Path(tmp)/"train-dev.json"
        path.write_text(json.dumps({"train":train,"dev":dev},ensure_ascii=False))
        config=core.LANGUAGES["french"].copy()
        core.LANGUAGES["french"]={
            **config,"data_url":path.as_uri(),"archive_name":"tatoeba-stage8-fr",
            "warm_adapter_override":{"path":WARM_REL,"sha256":WARM_SHA}}
        core.SEED=SEED
        core.TRAIN_COUNT=len(train)
        core.DEV_COUNT=len(dev)
        core.TOTAL_UPDATES=UPDATES
        core.CHECKPOINTS=(128,256,384,512)
        def parse(raw:bytes,*,archive_name:str):
            if archive_name!="tatoeba-stage8-fr":raise ValueError("Unexpected source")
            obj=json.loads(raw)
            if (len(obj["train"]),len(obj["dev"]))!=(len(train),len(dev)):
                raise RuntimeError("Frozen training cohort mutated")
            return obj["train"],obj["dev"],len(train)+len(dev)
        core.parse_pairs=parse
        report=core.train_experiment(ROOT,output,"french")
        report.update({"stage":8,"real_source_component_manifest":manifest,
                       "updates_planned":UPDATES,
                       "validity":"Research only; human semantic adequacy unverified."})
        (output/"report.json").write_text(json.dumps(report,indent=2))
    # These Tatoeba TRAIN-group cases were reserved BEFORE training, NOT used
    # for fitting or choosing; they become CONSUMED after this single score.
    scores=evaluate_frozen_test("french",output,final_test,report["selected"],
                                 previous_adapter=warm)
    summary={
        "language":"french","selected_on_development_only":report["selected"],
        "completed_optimizer_steps":report["completed_updates"],
        "cohort_manifest":manifest,
        "first_and_only_new_component_disjoint_diagnostic_scores":scores,
        "new_test_not_used_for_fitting_or_selection":True,
        "no_final_human_90_percent_accuracy_claim":True,
        "limitations":"Publisher TRAIN-component holdout, not blinded real-dialogue "
                      "human adequacy. Never reuse these exact groups for tuning. "
                      "Context and slang remain unverified; prior Marian pretraining "
                      "overlap cannot be ruled out."
    }
    (output/"stage8_evaluation.json").write_text(json.dumps(summary,indent=2))
    print("STAGE8_REAL_TRAINING_RESULTS",json.dumps(summary),flush=True)

if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--output-dir",type=Path,required=True)
    run(arg.parse_args().output_dir.resolve())
