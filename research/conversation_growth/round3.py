#!/usr/bin/env python3
"""Actual M2M100 round 3: shared rank-4 continuation, licensed parallel text.

Corrects two weaknesses of rounds 1/2: too little corpus diversity, and no
separate genuine parallel-text DEV. Train on new Tatoeba sentences with
warmup/cosine optimization; replay some prior *authentic* parallel sentences.
Select against BOTH frozen original MASSIVE validation and a new Tatoeba
development slice, with per-route regression limits. Keep all parents frozen
on disk; this experiment never updates the live provider or makes a semantic
accuracy claim.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from research.conversation_growth.round3_data import settings,build_data
from research.conversation_growth.data import sampled_dev
from research.multilingual_student.model import (
    load_shared_model,attach_shared_lora,load_adapter,
    sha256,only_adapter_state,read_config
)
from research.multilingual_student.train import (
    evaluate,encode_for_loss,ensure_old_winners_untouched
)

HERE = ROOT / "research/conversation_growth"
OUT = HERE / "experiment_003"
ROUTES = ("en-es","es-en","en-fr","fr-en","es-fr","fr-es")


def make_schedule(train: dict,cfg:dict) -> list:
    """Each bilingual Tatoeba pair is used in at most one optimizer step."""
    steps_per_route = cfg["train"]["updates_per_direction"]
    batch_size = cfg["train"]["batch_size"]
    seed = cfg["train"]["seed"]
    schedule = []
    all_seen = set()
    for pair in ("en-es","en-fr","es-fr"):
        pool = list(train[pair])
        random.Random(seed + sum(map(ord,pair))).shuffle(pool)
        first,second = pair.split("-")
        route_specs=((first+"-"+second,False),(second+"-"+first,True))
        for offset,(route,reverse) in enumerate(route_specs):
            selected = pool[offset*steps_per_route*batch_size:
                            (offset+1)*steps_per_route*batch_size]
            if len(selected) != steps_per_route*batch_size:
                raise RuntimeError("Insufficient genuine aligned pairs for "+route)
            for i in range(steps_per_route):
                examples=[]
                for row in selected[i*batch_size:(i+1)*batch_size]:
                    key=pair+":"+str(row["row_id"])
                    if key in all_seen:
                        raise RuntimeError("A training pair was reused or reversed")
                    all_seen.add(key)
                    a,b = (row["target"],row["source"]) if reverse else (row["source"],row["target"])
                    examples.append({"src":a,"tgt":b,"group_id":key,
                                     "license":"CC-BY-2.0-FR","provenance":row["provenance"]})
                schedule.append({"route":route,"examples":examples,
                                 "origin":"licensed_genuine_Tatoeba_parallel"})
    if len(schedule)!=cfg["train"]["total_updates"] or len(all_seen)!=(len(schedule)*batch_size):
        raise RuntimeError("Round3 schedule is not the approved budget")
    random.Random(seed).shuffle(schedule)
    for i,item in enumerate(schedule,1):
        item["optimizer_step"]=i
    if set(x["route"] for x in schedule)!=set(ROUTES):
        raise RuntimeError("Schedule did not cover all six language directions")
    return schedule


def learning_rate(step:int,cfg:dict)->float:
    train=cfg["train"]
    maxlr=float(train["max_learning_rate"])
    minlr=float(train["min_learning_rate"])
    warmup=int(train["warmup_updates"])
    updates=int(train["total_updates"])
    if not 1 <= step <= updates or not 0 < minlr < maxlr <= 0.001:
        raise ValueError("Unapproved update or learning rate")
    if not 0 < warmup < updates:
        raise ValueError("Invalid warmup schedule")
    if step <= warmup:
        return maxlr * step/warmup
    progress=(step-warmup)/(updates-warmup)
    return minlr+(maxlr-minlr)*0.5*(1+math.cos(math.pi*progress))


def compare_scores(baseline:dict,other:dict,*,label:str,cfg:dict) -> dict:
    gates=cfg["quality_gates"]
    if set(baseline)!=set(ROUTES) or set(other)!=set(ROUTES):
        raise RuntimeError("Incomplete development route set")
    chrf={}
    bleu={}
    for route in ROUTES:
        if other[route]["n"]!=baseline[route]["n"] or other[route]["n"]<20:
            raise RuntimeError("Incompatible DEV sample counts")
        chrf[route]=round(other[route]["chrF_not_accuracy"]-
                          baseline[route]["chrF_not_accuracy"],4)
        bleu[route]=round(other[route]["BLEU_not_accuracy"]-
                          baseline[route]["BLEU_not_accuracy"],4)
    mean=sum(chrf.values())/len(chrf)
    if label=="massive":
        gain=gates["massive_mean_chrF_min_gain"]
        chrf_loss=gates["massive_max_per_direction_chrF_loss"]
        bleu_loss=gates["massive_max_per_direction_BLEU_loss"]
    elif label=="tatoeba":
        gain=gates["tatoeba_mean_chrF_min_gain"]
        chrf_loss=gates["tatoeba_max_per_direction_chrF_loss"]
        bleu_loss=gates["tatoeba_max_per_direction_BLEU_loss"]
    else:
        raise ValueError("Unexpected validation set")
    failed=[]
    if mean<gain:
        failed.append("insufficient_mean_"+label+"_chrF_gain")
    failed += [x+":chrF_regression" for x in ROUTES if chrf[x]<-chrf_loss]
    failed += [x+":BLEU_regression" for x in ROUTES if bleu[x]<-bleu_loss]
    return {"eligible":not failed,
            "chrF_delta_not_human_accuracy":chrf,
            "BLEU_delta_not_human_accuracy":bleu,
            "mean_chrF_delta":round(mean,4),
            "blocked_by":failed}


def acceptance(baseline_massive,after_massive,baseline_tatoeba,after_tatoeba,cfg):
    a=compare_scores(baseline_massive,after_massive,label="massive",cfg=cfg)
    b=compare_scores(baseline_tatoeba,after_tatoeba,label="tatoeba",cfg=cfg)
    # No result is eligible for release without bilingual human review.
    return a["eligible"] and b["eligible"],{"massive":a,"tatoeba":b,
        "automated_research_selection_only":True,
        "numeric_and_slang_context_meaning_not_independently_verified":True}


def run():
    import torch
    from safetensors.torch import save_file
    cfg,_=settings()
    parent=ROOT/cfg["parent"]["path"]
    parent_sha=cfg["parent"]["sha256"]
    if OUT.exists():
        raise FileExistsError("Round3 experiment already exists; previous records immutable")
    if sha256(parent)!=parent_sha:
        raise RuntimeError("Selected old shared checkpoint was changed or missing")
    chosen=json.loads((ROOT/"research/multilingual_student/experiment_002/checkpoint_selection.json").read_text())
    if chosen["selected_research_sha256"] != parent_sha:
        raise RuntimeError("Parent no longer selected by earlier research gates")
    previous=json.loads((HERE/"experiment_002/report.json").read_text())
    if previous["selected_sha256"]!=parent_sha or previous["actual_optimizer_updates"]!=48:
        raise RuntimeError("Previous round's selection and lineage must be retained")
    specialists=ensure_old_winners_untouched()
    torch.manual_seed(cfg["train"]["seed"])
    random.seed(cfg["train"]["seed"])
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    start=time.monotonic()

    train,dev_new,manifest=build_data()
    dev_original,groups=sampled_dev(cfg["data"]["frozen_massive_validation_per_route"])
    digest=hashlib.sha256("|".join(sorted(str(g["id"]) for g in groups)).encode()).hexdigest()
    if digest!=cfg["reference_receipts"]["frozen_massive_dev_sha256"]:
        raise RuntimeError("Frozen MASSIVE development group changed")
    schedule=make_schedule(train,cfg)
    manifest["training_schedule_sha256"]=hashlib.sha256(json.dumps([
        {"route":r["route"],"ids":[v["group_id"] for v in r["examples"]]} for r in schedule
    ],sort_keys=True).encode()).hexdigest()

    model,tokenizer=load_shared_model()
    base=read_config()["training"]
    architecture=attach_shared_lora(
        model,rank=base["lora_rank"],alpha=base["lora_alpha"],
        encoder_last=base["last_encoder_layers"],decoder_last=base["last_decoder_layers"])
    if architecture["rank"]!=cfg["parent"]["rank"] or architecture["alpha"]!=cfg["parent"]["alpha"]:
        raise RuntimeError("Changed shared adapter architecture")
    load_adapter(model,parent,sha=parent_sha)
    baseline_massive=evaluate(model,tokenizer,dev_original,n_per_route=len(groups))
    baseline_tatoeba=evaluate(model,tokenizer,dev_new,
                              n_per_route=cfg["data"]["dev_per_corpus"])
    if not all(v["n"]==36 for v in baseline_massive.values()):
        raise RuntimeError("Frozen old development cohort incomplete")
    if not all(v["n"]==20 for v in baseline_tatoeba.values()):
        raise RuntimeError("New parallel development cohort incomplete")
    old_scores=json.loads((HERE/"experiment_001/report.json").read_text())
    for direction in ROUTES:
        for metric in ("chrF_not_accuracy","BLEU_not_accuracy"):
            expected=old_scores["baseline_dev_not_meaning_accuracy"][direction][metric]
            if abs(baseline_massive[direction][metric]-expected)>0.05:
                raise RuntimeError("Pinned parent has different baseline score: "+direction)
    print("SAFE_TWO_DATASET_BASELINES_READY",json.dumps({
        "Tatoeba_DEV_n_per_direction":20,
        "original_frozen_MASSIVE_DEV_n_per_direction":36,
        "no_sealed_or_independent_human_test_access":True,
        "unchanged_parent_sha256":parent_sha
    }),flush=True)

    params=[p for p in model.parameters() if p.requires_grad]
    anchors=[p.detach().clone() for p in params]
    if len(params)!=24:
        raise RuntimeError("Expected 24 shared LoRA trainable parameter tensors")
    optimizer=torch.optim.AdamW(params,lr=cfg["train"]["max_learning_rate"],weight_decay=0.0)
    snapshots={}
    history=[]
    for item in schedule:
        i=item["optimizer_step"]
        src,tgt=item["route"].split("-")
        lr=learning_rate(i,cfg)
        for group in optimizer.param_groups:
            group["lr"]=lr
        inputs,labels=encode_for_loss(
            tokenizer,item["examples"],src=src,tgt=tgt,
            max_tokens=cfg["train"]["max_token_length"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        base_loss=model(**inputs,labels=labels).loss
        # A soft weight anchor protects the initialization on optimizer updates.
        # Actual non-regression is decided by frozen validation, not this term.
        penalty=sum(torch.mean((p-v)**2) for p,v in zip(params,anchors))/len(params)
        loss=base_loss+cfg["train"]["anchor_penalty"]*penalty
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,cfg["train"]["clip_grad_norm"])
        if not math.isfinite(float(norm)):
            raise RuntimeError("Invalid gradient norm")
        optimizer.step()
        observation={"update":i,"route":item["route"],
                     "learning_rate":round(lr,8),
                     "cross_entropy":round(float(base_loss),6),
                     "gradient_norm":round(float(norm),5)}
        history.append(observation)
        print("REAL_SHARED_MODEL_ROUND3_OPTIMIZER_UPDATE",
              json.dumps(observation),flush=True)
        if i not in cfg["train"]["saved_updates"]:
            continue
        OUT.mkdir(parents=True,exist_ok=True)
        name=f"shared_lora_parallel_extra{i:03d}.safetensors"
        saved=OUT/name
        if saved.exists():
            raise FileExistsError("Never overwrite experimental weights")
        save_file(only_adapter_state(model),str(saved),metadata={
            "model":"facebook/m2m100_418M","parent_sha256":parent_sha,
            "reproducible_origin":"licensed_Tatoeba_parallel_only",
            "approved_for_production":"false",
            "independent_human_semantic_quality_verified":"false"
        })
        massive=evaluate(model,tokenizer,dev_original,n_per_route=len(groups))
        tatoeba=evaluate(model,tokenizer,dev_new,
                         n_per_route=cfg["data"]["dev_per_corpus"])
        eligible,gate=acceptance(baseline_massive,massive,baseline_tatoeba,tatoeba,cfg)
        snapshots[name]={
            "sha256":sha256(saved),"bytes":saved.stat().st_size,
            "massive_dev":massive,"tatoeba_dev":tatoeba,
            "automatic_gates":gate,"eligible_for_research_only":eligible
        }
        print("REAL_ROUND3_CHECKPOINT_EVALUATED",json.dumps({
            "checkpoint":name,"passed_both_dev_gates":eligible,
            "massive_mean_chrF_delta":gate["massive"]["mean_chrF_delta"],
            "tatoeba_mean_chrF_delta":gate["tatoeba"]["mean_chrF_delta"],
            "blocked_massive":gate["massive"]["blocked_by"],
            "blocked_tatoeba":gate["tatoeba"]["blocked_by"]
        }),flush=True)

    if len(history)!=cfg["train"]["total_updates"] or len(snapshots)!=len(cfg["train"]["saved_updates"]):
        raise RuntimeError("Incomplete training run")
    parent_ok=sha256(parent)==parent_sha
    specialists_ok=ensure_old_winners_untouched()==specialists
    if not parent_ok or not specialists_ok:
        raise RuntimeError("Frozen historical weights changed unexpectedly")
    eligible=[(v["automatic_gates"]["massive"]["mean_chrF_delta"]+
               v["automatic_gates"]["tatoeba"]["mean_chrF_delta"],name)
              for name,v in snapshots.items() if v["eligible_for_research_only"]]
    if eligible:
        _,winning=max(eligible)
        selected=OUT/winning
        decision="candidate_research_dev_pass_only_not_production"
    else:
        selected=parent
        decision="parent_preserved_no_candidate_passed_both_gates"
    report={
        "status":"actual_shared_rank4_parallel_round3_completed",
        "model_id":cfg["model"]["id"],
        "actual_optimizer_updates":len(history),
        "routes":dict(Counter(x["route"] for x in history)),
        "parent_sha256":parent_sha,
        "parent_file_unchanged":parent_ok,
        "original_specialists_unchanged":specialists_ok,
        "architecture":architecture,
        "train_manifest":manifest,
        "new_parallel_dev_per_direction":20,
        "frozen_massive_dev_per_direction":36,
        "frozen_massive_group_digest":digest,
        "baseline_massive_dev_not_human_accuracy":baseline_massive,
        "baseline_tatoeba_dev_not_human_accuracy":baseline_tatoeba,
        "checkpoints":snapshots,
        "checkpoint_decision":decision,
        "selected_path":str(selected.relative_to(ROOT)),
        "selected_sha256":sha256(selected),
        "never_downgrade_if_gates_fail":True,
        "previous_models_and_reports_preserved":True,
        "train_log":history,
        "human_semantic_accuracy_verified":False,
        "true_multiturn_context_or_slang_supervision":False,
        "new_tatoeba_dev_is_not_independent_of_foundation_pretraining":True,
        "final_holdout_accessed":False,
        "user_reviewed_500_used":False,
        "noncommercial_data_used":False,
        "production_provider_or_ASR_modified":False,
        "release_approved":False,
        "paid_spend_usd":0,
        "wall_seconds":round(time.monotonic()-start,3),
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "warning":"chrF/BLEU are overlap scores, NOT meaning accuracy. "
                  "Development data is limited, and authentic multi-turn "
                  "bilingual supervision and independent human QA are still missing."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (OUT/"data_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    (OUT/"SHA256SUMS.txt").write_text("".join(
        sha256(p)+"  "+p.name+"\n" for p in sorted(OUT.iterdir()) if p.is_file()
    ))
    print("ONE_MODEL_ROUND3_TRAINED",json.dumps({
        "updates":len(history),"decision":decision,
        "selected_sha256":report["selected_sha256"],
        "production_changed":False,"human_meaning_accuracy_verified":False
    }),flush=True)


if __name__=="__main__":
    run()
