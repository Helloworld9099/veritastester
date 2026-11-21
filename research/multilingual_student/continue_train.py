#!/usr/bin/env python3
"""A REAL second multilingual training round, continuing learned student weights.

Starts from the committed single shared LoRA after 18 optimizer updates, not
from base/random or old incompatible French/Spanish Marian tensors. New
publisher TRAIN group IDs exclude every first-round sampled group per route.
Uses a conservative learning rate, balanced replay, and extra priority for
ES->EN and FR->EN. Keeps the parent if any route regresses on DEV.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.multilingual_student.data import prepare,make_step_schedule
from research.multilingual_student.model import (
    read_config,load_shared_model,attach_shared_lora,load_adapter,only_adapter_state,
    sha256
)
from research.multilingual_student.train import (
    evaluate,encode_for_loss,ensure_old_winners_untouched
)

PARENT=ROOT/"research/multilingual_student/experiment_001"
OUTPUT=ROOT/"research/multilingual_student/experiment_002"
PARENT_SHA="97c3a1b23cf7d290cecac038ead9fd8a6f810905944b6c441184c48418258f4b"
SEED=2026112102
STEPS=60
LR=0.000065
BATCH=2
SAVE_AT=(20,40,60)
TARGET_UPDATES={
    "fr-en":16,"es-en":16,"en-es":10,
    "en-fr":6,"es-fr":6,"fr-es":6,
}


def schedule_fresh(train):
    cfg=read_config()
    first=make_step_schedule(
        train,seed=cfg["training"]["seed"],steps=cfg["training"]["updates"],
        batch_size=cfg["training"]["batch_size"])
    previously_used={
        (round["route"],str(item["group_id"]))
        for round in first for item in round["examples"]
    }
    routes=[]
    for route,n in TARGET_UPDATES.items():
        routes.extend([route]*n)
    if len(routes)!=STEPS or set(routes)!=set(cfg["directions"]):
        raise RuntimeError("Expected all six routes and 60 additional updates")
    rng=random.Random(SEED)
    rng.shuffle(routes)
    by_route={}
    for route,items in train.items():
        eligible=[item for item in items if
                  (route,str(item["group_id"])) not in previously_used]
        random.Random(SEED+sum(map(ord,route))).shuffle(eligible)
        if len(eligible)<TARGET_UPDATES[route]*BATCH:
            raise RuntimeError("Not enough unseen train utterances to continue")
        by_route[route]=eligible
    used=Counter()
    steps=[]
    for i,route in enumerate(routes):
        offset=used[route]*BATCH
        examples=by_route[route][offset:offset+BATCH]
        if len(examples)!=BATCH:
            raise RuntimeError("Training sampler wrapped into repeated old data")
        used[route]+=1
        if any((route,str(row["group_id"])) in previously_used for row in examples):
            raise RuntimeError("Already-trained route/group repeated")
        steps.append({"step":i+1,"route":route,"examples":examples})
    return steps,{
        "old_round1_route_group_examples_excluded":True,
        "new_optimizer_updates":STEPS,
        "sampling":dict(used),
        "new_source_target_pairs":STEPS*BATCH,
        "schedule_sha256":hashlib.sha256(
            json.dumps([{"route":item["route"],
                         "ids":[r["group_id"] for r in item["examples"]]}
                        for item in steps],sort_keys=True).encode()).hexdigest(),
    }


def run():
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite any student checkpoint")
    cfg=read_config()
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(cfg["compute"]["max_omp_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    start=time.monotonic()
    first=json.loads((PARENT/"report.json").read_text())
    if first["status"]!="completed_actual_single_multilingual_student_training":
        raise RuntimeError("First actual trained shared checkpoint not available")
    adapter=PARENT/"shared_lora_step18.safetensors"
    if sha256(adapter)!=first["checkpoint_sha256"] or first["checkpoint_sha256"]!=PARENT_SHA:
        raise RuntimeError("Failed to pin already-learned multilingual weights")
    original=ensure_old_winners_untouched()
    train,dev,receipt=prepare()
    if receipt["train_sha256"]!=first["training_dataset_sha256"]:
        raise RuntimeError("Publisher training identities changed")
    if receipt["dev_sha256"]!=first["dev_dataset_sha256"]:
        raise RuntimeError("Original development split was modified")
    steps,more_data=schedule_fresh(train)
    print("CONTINUING_REAL_STUDENT_FROM_PREVIOUS_LEARNED_WEIGHTS",json.dumps({
        "parent":PARENT_SHA,"train":more_data,
        "old_specialists":original}),flush=True)
    model,tok=load_shared_model()
    meta=attach_shared_lora(model,
                            rank=cfg["training"]["lora_rank"],
                            alpha=cfg["training"]["lora_alpha"],
                            encoder_last=cfg["training"]["last_encoder_layers"],
                            decoder_last=cfg["training"]["last_decoder_layers"])
    load_adapter(model,adapter,sha=PARENT_SHA)
    parent_score=evaluate(model,tok,dev,n_per_route=12)
    trainable=[p for p in model.parameters() if p.requires_grad]
    anchor=[p.detach().clone() for p in trainable]
    optimizer=torch.optim.AdamW(trainable,lr=LR,weight_decay=0.0)
    loss_records=[]
    for step in steps:
        route=step["route"]
        src,tgt=route.split("-")
        batch,labels=encode_for_loss(
            tok,step["examples"],src=src,tgt=tgt,max_tokens=96)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predicted=model(**batch,labels=labels)
        # Mild penalty limits drift away from the already learned 6-route
        # checkpoint. It is NOT a proven solution to catastrophic forgetting.
        stability=sum(torch.mean((p-v)**2) for p,v in zip(trainable,anchor))/len(trainable)
        loss=predicted.loss + 0.01*stability
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss in continuation")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(trainable,max_norm=1.0)
        if not math.isfinite(float(norm)):
            raise RuntimeError("Bad gradient in continuation")
        optimizer.step()
        record={"step":step["step"],"route":route,
                "total_loss":round(float(loss),6),
                "grad_norm":round(float(norm),6)}
        loss_records.append(record)
        print("CONTINUED_MULTILINGUAL_OPTIMIZER_STEP",json.dumps(record),flush=True)
        if step["step"] in SAVE_AT:
            OUTPUT.mkdir(parents=True,exist_ok=True)
            file=OUTPUT/f"shared_lora_extra{step['step']:02d}.safetensors"
            if file.exists():
                raise FileExistsError("Do not overwrite student optimizer candidate")
            save_file(only_adapter_state(model),str(file),metadata={
                "one_shared_multilingual_student":"true",
                "parent_sha256":PARENT_SHA,
                "added_optimizer_steps":str(step["step"])})
            print("SAVED_STUDENT_CONTINUATION_CANDIDATE",json.dumps({
                "update":step["step"],"sha256":sha256(file),
                "bytes":file.stat().st_size}),flush=True)
    if len(loss_records)!=STEPS:
        raise RuntimeError("Some optimizer updates were skipped")
    new_score=evaluate(model,tok,dev,n_per_route=12)
    change={k:round(new_score[k]["chrF_not_accuracy"]-
                    parent_score[k]["chrF_not_accuracy"],4)
            for k in cfg["directions"]}
    bleu_change={k:round(new_score[k]["BLEU_not_accuracy"]-
                         parent_score[k]["BLEU_not_accuracy"],4)
                 for k in cfg["directions"]}
    pass_automatic_gate=(
        all(v>=-0.75 for v in change.values())
        and all(v>=-0.75 for v in bleu_change.values())
        and (sum(change.values())/len(change))>=0.25
    )
    result_file=OUTPUT/"shared_lora_extra60.safetensors"
    candidate_sha=sha256(result_file)
    approved_dev_only=(candidate_sha if pass_automatic_gate else PARENT_SHA)
    report={
        "status":"completed_real_shared_multilingual_student_continuation",
        "model_id":cfg["model"]["id"],
        "same_single_base_and_single_shared_adapter":True,
        "parent_checkpoint_sha256":PARENT_SHA,
        "parent_weights_file_not_modified":sha256(adapter)==PARENT_SHA,
        "candidate_checkpoint_sha256":candidate_sha,
        "selected_research_checkpoint_sha256":approved_dev_only,
        "selected_from":"continued_candidate" if pass_automatic_gate else "preserved_parent",
        "updated_steps":STEPS,
        "total_student_updates_in_lineage":18+STEPS,
        "directions":cfg["directions"],
        "each_direction_actual_update_count":dict(Counter(x["route"] for x in loss_records)),
        "optimizer_state_reset_for_new_round":True,
        "underlying_model_and_adapter_weights_continued":True,
        "parent_development_12_per_direction":parent_score,
        "candidate_development_12_per_direction":new_score,
        "development_chrF_deltas_not_accuracy":change,
        "development_BLEU_deltas_not_accuracy":bleu_change,
        "passes_automatic_no_regression_gate":pass_automatic_gate,
        "old_french_and_spanish_champions_unchanged":ensure_old_winners_untouched(),
        "reviewed_500_used_for_training":False,
        "final_holdout_or_publisher_test_accessed":False,
        "new_training_corpus":more_data,
        "no_nc_corpora_or_teacher_output_used":True,
        "training_loss":loss_records,
        "elapsed_seconds":time.monotonic()-start,
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "paid_cost_usd":0,
        "human_95_percent_meaning_accuracy_verified":False,
        "production_activated":False,
        "notice":"This is a real continuation from learned weights, but small publisher "
                 "dev chrF is NOT semantic adequacy or a meaningful final holdout. "
                 "Keep the historical specialist models. If any automatic subgroup "
                 "regresses, retain the exact previous shared checkpoint."
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (OUTPUT/"SHA256SUMS.txt").write_text("".join(
        sha256(f)+"  "+f.name+"\n" for f in sorted(OUTPUT.iterdir()) if f.is_file()))
    print("REAL_ONE_MULTILINGUAL_STUDENT_CONTINUED_FROM_PARENT",json.dumps({
        "updates":STEPS,
        "total_lineage_steps":18+STEPS,
        "selected_sha256":approved_dev_only,
        "parent":PARENT_SHA,
        "dev_deltas_not_accuracy":change,
        "bleu_deltas_not_accuracy":bleu_change,
        "automatic_gate":pass_automatic_gate,
        "human_accuracy_95_verified":False
    }),flush=True)


if __name__=="__main__":
    run()
