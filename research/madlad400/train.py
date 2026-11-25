#!/usr/bin/env python3
"""ACTUAL CPU/GPU pilot training of pinned MADLAD-400 3B (not a fake tiny model).

The 3B pretrained T5 weights are downloaded exactly once from the pinned
Apache-2.0 upstream model. Train only a shared final-decoder LoRA on publisher
MASSIVE TRAIN voice commands. Save real tensors, gradient norms, base SHA,
group-disjoint DEVELOPMENT-only comparison, resource costs and rollback.

An incompatible M2M100 LoRA cannot be loaded into MADLAD! The old 38-step
trained M2M100 checkpoint stays SHA-pinned and never overwritten. This starts
from MADLAD *translation-pretrained* weights, not random initialization.
"""
from __future__ import annotations

import json
import math
import os
import random
import resource
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from research.madlad400.model import (
    EXPERIMENT,ROOT,adapter_state,attach_shared_lora,config,
    file_sha256,load_backbone,supported_target_markers,
    translate_short,verify_model_card,language_input
)
from research.multilingual_student.data import prepare
from research.multilingual_student.train import ensure_old_winners_untouched
from research.conversation_growth.data import settings as previous_exp_settings

PARENT=ROOT/"research/multilingual_student/experiment_002/shared_lora_extra20.safetensors"


def _memory_available_mib()->int:
    # Linux MemAvailable is more useful than MemFree when evaluating whether
    # a large HF checkpoint can be loaded without pushing runner into OOM.
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])//1024
    except (FileNotFoundError,OSError,ValueError):
        return 0
    return 0


def _check_free_runner_limits()->dict:
    cfg=config()
    old=ensure_old_winners_untouched()
    if file_sha256(PARENT)!=cfg["prior_model"]["selected_lora_sha256"]:
        raise RuntimeError("Preserved 38-update learned M2M100 model changed")
    earlier=previous_exp_settings()
    if earlier["parent"]["selected_adapter_sha256"]!=cfg["prior_model"]["selected_lora_sha256"]:
        raise RuntimeError("Historical multilingual checkpoint ancestry was changed")
    available=_memory_available_mib()
    if available and available<8500:
        raise RuntimeError(f"Insufficient free RAM ({available} MiB) for 3B CPU BF16")
    disk=shutil.disk_usage(os.environ.get("RUNNER_TEMP","/tmp"))
    free_gb=disk.free/1000**3
    if free_gb<19:
        raise RuntimeError(f"Insufficient disk for pinned MADLAD weights/cache: {free_gb:.2f} GB")
    if os.environ.get("BRIDGE_NO_PAID_APIS","true").lower()!="true":
        raise RuntimeError("Paid API integrations are forbidden")
    return {"old_specialists":old,"old_shared_checkpoint_sha256":file_sha256(PARENT),
            "ram_available_mib_before_training":available,
            "runner_disk_free_gb_before_training":round(free_gb,2),
            "paid_compute_usd":0}


def sample_unseen_training():
    cfg=config()
    train,dev,manifest=prepare()  # Only publisher TRAIN + VALIDATION; no TEST
    schedule=[]
    for route in cfg["pilot"]["training_routes"]:
        if route not in train:
            raise RuntimeError("Requested train pair absent from source pool")
        chosen=next(iter(train[route]))
        schedule.append({"route":route,"src":chosen["src"],"tgt":chosen["tgt"],
                         "group_id":str(chosen["group_id"])})
    if len(schedule)!=cfg["pilot"]["optimizer_updates"]:
        raise RuntimeError("Expected one actual optimizer update per selected route")
    if any(r["group_id"] in {str(x["group_id"]) for x in dev[r["route"]]} for r in schedule):
        raise RuntimeError("Publisher development leakage")
    return schedule,dev,manifest


def supervised_loss(model,tokenizer,row,markers,*,max_source,max_target):
    target=row["route"].split("-")[1]
    source=language_input(row["src"],target,markers)
    src=tokenizer(source,return_tensors="pt",truncation=False)
    gold=tokenizer(text_target=row["tgt"],return_tensors="pt",truncation=False)
    if (int(src["input_ids"].shape[1])>max_source or
        int(gold["input_ids"].shape[1])>max_target):
        raise ValueError("Current translation example would be silently truncated")
    labels=gold["input_ids"].clone()
    labels[labels==tokenizer.pad_token_id]=-100
    return model(**src,labels=labels).loss


def development(model,tokenizer,dev,markers):
    """A resource-bounded smoke on exactly 1 pub-VALIDATION prompt per route.

    NOT a semantic accuracy metric and too small for checkpoint selection.
    """
    from sacrebleu.metrics import CHRF,BLEU
    scores={}
    for route,rows in sorted(dev.items()):
        source,target=route.split("-")
        prompt=rows[0]
        generated=translate_short(
            model,tokenizer,prompt["src"],source,target,markers,
            max_source_tokens=config()["pilot"]["max_source_tokens"])
        scores[route]={
            "n":1,"prediction":generated["text"],
            "chrF_not_accuracy":round(CHRF().sentence_score(
                generated["text"],[prompt["tgt"]]).score,3),
            "BLEU_not_accuracy":round(BLEU().sentence_score(
                generated["text"],[prompt["tgt"]]).score,3),
            "inference_seconds":round(generated["seconds"],3),
            "human_meaning_correctness_unverified":True,
        }
        print("MADLAD_3B_DEVELOPMENT_SMOKE",json.dumps({
            "route":route,"chrF_not_accuracy":scores[route]["chrF_not_accuracy"],
            "seconds":scores[route]["inference_seconds"],"n":1}),flush=True)
    return scores


def run():
    cfg=config()
    if EXPERIMENT.exists():
        raise FileExistsError("Refusing to destroy or retrain the first MADLAD checkpoint")
    EXPERIMENT.mkdir(parents=True)
    begin=time.monotonic()
    policy=_check_free_runner_limits()
    rights=verify_model_card()
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(cfg["pilot"]["number_of_free_cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(cfg["pilot"]["seed"])
    random.seed(cfg["pilot"]["seed"])
    schedule,dev,data_receipt=sample_unseen_training()
    (EXPERIMENT/"data_manifest.json").write_text(
        json.dumps(data_receipt,sort_keys=True,indent=2)+"\n")
    print("MADLAD_PINNED_TRAINING_INPUTS",json.dumps({
        "model_license":rights,"train_groups":data_receipt["train_groups"],
        "train_routes":[x["route"] for x in schedule],
        "dev_groups":data_receipt["development_groups"],
        "old_learned_model_SHA":policy["old_shared_checkpoint_sha256"],
        "free_mem_MiB":policy["ram_available_mib_before_training"]
    }),flush=True)
    model,tokenizer,markers=load_backbone(check_rights=False,dtype=torch.bfloat16)
    model.config.use_cache=False
    print("ACTUAL_PRETRAINED_MADLAD_LOADED",json.dumps({
        "model_type":model.config.model_type,
        "native_target_tags":len(markers),
        "parameters":sum(x.numel() for x in model.parameters()),
        "residency":"one_model_CPU_bfloat16",
        "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    }),flush=True)
    base_before=development(model,tokenizer,dev,markers)
    adapter=attach_shared_lora(model,rank=cfg["pilot"]["LoRA"]["rank"],
                               alpha=cfg["pilot"]["LoRA"]["alpha"])
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=cfg["pilot"]["lr"],weight_decay=0.0)
    updates=[]
    for idx,row in enumerate(schedule,1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=supervised_loss(
            model,tokenizer,row,markers,
            max_source=cfg["pilot"]["max_source_tokens"],
            max_target=cfg["pilot"]["max_target_tokens"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite MADLAD actual training loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(trainable,max_norm=1)
        if not math.isfinite(float(norm)):
            raise RuntimeError("Nonfinite MADLAD low-rank gradient")
        opt.step()
        trace={"update":idx,"direction":row["route"],
               "loss":round(float(loss.detach()),5),
               "gradient_norm":round(float(norm),5),
               "source":"MASSIVE_CC-BY-4.0_publisher_TRAIN"}
        updates.append(trace)
        print("REAL_MADLAD_3B_OPTIMIZER_UPDATE",json.dumps(trace),flush=True)
        file=EXPERIMENT/f"madlad400_shared_lora_step{idx:02d}.safetensors"
        save_file(adapter_state(model),str(file),metadata={
            "model_id":cfg["model"]["repo_id"],
            "model_revision":cfg["model"]["revision"],
            "optimizer_updates":str(idx),"one_shared_adapter":"true",
            "production_approved":"false"})
        print("SAVED_MADLAD_ACTUAL_LEARNED_TENSORS",json.dumps({
            "step":idx,"sha256":file_sha256(file),"bytes":file.stat().st_size}),flush=True)
    after=development(model,tokenizer,dev,markers)
    if len(updates)!=cfg["pilot"]["optimizer_updates"]:
        raise RuntimeError("MADLAD optimizer did not complete all configured updates")
    chosen=EXPERIMENT/f"madlad400_shared_lora_step{len(updates):02d}.safetensors"
    report={
        "status":"completed_real_MADLAD_3B_CPU_finetuning_pilot",
        "model_id":cfg["model"]["repo_id"],
        "model_commit_SHA":cfg["model"]["revision"],
        "model_license":rights["license"],
        "model_family":"T5_pretrained_machine_translation",
        "native_target_codes":len(markers),
        "real_optimizer_updates":len(updates),
        "training_examples_seen":len(updates),
        "real_trainable_adapter":adapter,
        "trained_shared_model_adapter_sha256":file_sha256(chosen),
        "all_previous_adapters_unchanged":
            file_sha256(PARENT)==cfg["prior_model"]["selected_lora_sha256"],
        "parent_pretrained_weights_randomly_initialized":False,
        "reused_incompatible_old_M2M100_weights":False,
        "original_legacy_Marian_models":ensure_old_winners_untouched(),
        "licensed_training_split":"MASSIVE publisher TRAIN CC BY 4.0",
        "publisher_development_only":True,
        "publisher_final_test_accessed":False,
        "previous_500_reviewed_cases_used_as_training":False,
        "development_baseline":base_before,
        "development_after":after,
        "automatic_dev_chrF_deltas_not_semantic_accuracy":{
            route:round(after[route]["chrF_not_accuracy"]-
                        base_before[route]["chrF_not_accuracy"],3)
            for route in base_before},
        "optimizer_trace":updates,
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds":round(time.monotonic()-begin,2),
        "paid_spend_usd":0,
        "commercial_licensed_model_verified":True,
        "production_activated":False,
        "human_95_meaning_accuracy_verified":False,
        "truly_context_slanged_trained":False,
        "full_ASR_Whisper_app_integration_verified":False,
        "quality_gate":"research-only, THREE steps are feasibility testing, no accuracy promotion",
        "warning":"Tiny baseline vs candidate evaluation has only ONE development "
                  "example/direction. This verifies real training and weights, "
                  "not meaningful improvement or production adequacy. "
                  "Do not replace legacy translators on these results."
    }
    (EXPERIMENT/"report.json").write_text(
        json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n")
    (EXPERIMENT/"SHA256SUMS.txt").write_text("".join(
        file_sha256(f)+"  "+f.name+"\n" for f in sorted(EXPERIMENT.iterdir())
        if f.is_file()))
    print("ACTUAL_MADLAD_400_MULTILINGUAL_TRAINING_REPORT",json.dumps({
        "status":report["status"],"steps":len(updates),
        "trained_sha256":report["trained_shared_model_adapter_sha256"],
        "native_language_codes":len(markers),
        "human_accuracy_claim":False,"paid_spend_usd":0
    }),flush=True)


if __name__=="__main__":
    run()
