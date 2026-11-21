#!/usr/bin/env python3
"""Actually train ONE shared six-direction M2M100 LoRA and compare to base.

This is a bounded, reproducible CPU research pilot; NOT a 95% human-adequacy
claim or a production replacement. Old French/Spanish Marian checkpoints are
not overwritten, merged incorrectly, or trained on NC corpora.

One base + ONE shared adapter = one neural network handling en/fr/es in all
directions. The merge/export utility can turn those weights into a standalone
M2M100 checkpoint, subject to the ~1.94 GB base distribution size.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.multilingual_student.data import prepare, make_step_schedule
from research.multilingual_student.model import (
    CONFIG, load_shared_model, attach_shared_lora, generate,
    only_adapter_state, read_config, sha256, verify_model_rights
)

OUTPUT = ROOT / "research/multilingual_student/experiment_001"
PARENT_LEGACY_SHA = {
    "es-en": "3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71",
    "fr-en": "fba2bd78958bf0bf8a4c14dc55e601e08663633718c4c9c17f91ef9b52dcfa1c",
}


def ensure_old_winners_untouched():
    old_paths={
        "es-en":ROOT/"experiments/es_en_v2_runs/candidate_after_three.safetensors",
        "fr-en":ROOT/"experiments/balanced_stage5_20261118/saved/french/adapter_selected.safetensors",
    }
    for lang,path in old_paths.items():
        if not path.is_file() or sha256(path)!=PARENT_LEGACY_SHA[lang]:
            raise RuntimeError("Cannot start: historical trained "+lang+" champion changed")
    return {lang:{"path":str(old_paths[lang].relative_to(ROOT)),
                  "sha256":PARENT_LEGACY_SHA[lang]} for lang in old_paths}


def encode_for_loss(tokenizer,rows,*,src:str,tgt:str,max_tokens:int):
    tokenizer.src_lang=src
    tokenizer.tgt_lang=tgt
    inputs=tokenizer([r["src"] for r in rows],return_tensors="pt",
                     padding=True,truncation=False)
    if inputs["input_ids"].shape[1]>max_tokens:
        raise ValueError("Training input exceeds source budget; never silently truncate")
    labels=tokenizer(text_target=[r["tgt"] for r in rows],return_tensors="pt",
                     padding=True,truncation=False)
    if labels["input_ids"].shape[1]>max_tokens:
        raise ValueError("Training reference exceeds target budget")
    target=labels["input_ids"]
    target[target==tokenizer.pad_token_id]=-100
    return inputs,target


def evaluate(model,tokenizer,dev,*,n_per_route:int):
    from sacrebleu.metrics import CHRF, BLEU
    import torch
    model.eval()
    results={}
    with torch.inference_mode():
        for route,rows in dev.items():
            source,target=route.split("-")
            selected=rows[:n_per_route]
            preds=[generate(model,tokenizer,r["src"],source,target,max_length=72)
                   for r in selected]
            gold=[r["tgt"] for r in selected]
            results[route]={
                "n":len(selected),
                "chrF_not_accuracy":round(CHRF().corpus_score(preds,[gold]).score,4),
                "BLEU_not_accuracy":round(BLEU().corpus_score(preds,[gold]).score,4),
                "empty_predictions":sum(not x for x in preds),
                "example_output":preds[0],
                "source_and_reference_not_in_report":True,
            }
            print("MULTILINGUAL_DEV_"+route.upper().replace("-","_"),
                  json.dumps({k:v for k,v in results[route].items()
                              if k!="example_output"}),flush=True)
    return results


def run():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite or restart a previous training experiment")
    cfg=read_config()
    import torch
    from safetensors.torch import save_file
    start=time.monotonic()
    tcfg=cfg["training"]
    seed=tcfg["seed"]
    torch.set_num_threads(cfg["compute"]["max_omp_threads"])
    torch.set_num_interop_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    old=ensure_old_winners_untouched()
    print("HISTORICAL_FRENCH_AND_SPANISH_CHECKPOINTS_UNTOUCHED",json.dumps(old),flush=True)
    rights=verify_model_rights()
    print("PINNED_SINGLE_MODEL_LICENSE",json.dumps(rights),flush=True)
    data_start=time.monotonic()
    train,dev,data_receipt=prepare()
    schedule=make_step_schedule(
        train,seed=seed,steps=tcfg["updates"],batch_size=tcfg["batch_size"])
    data_receipt["optimizer_schedule_sha256"]=hashlib.sha256(
        json.dumps([{"step":x["step"],"route":x["route"],
                     "ids":[a["group_id"] for a in x["examples"]]}
                    for x in schedule],sort_keys=True).encode()).hexdigest()
    print("TRILINGUAL_LICENSED_CORPUS_READY",json.dumps(data_receipt),flush=True)

    model,tokenizer=load_shared_model(verify_license=False)
    from huggingface_hub import hf_hub_download
    pinned_base=Path(hf_hub_download(repo_id=cfg["model"]["id"],
                                    revision=cfg["model"]["revision"],
                                    filename="pytorch_model.bin"))
    base_weight_sha=sha256(pinned_base)
    base_weight_bytes=pinned_base.stat().st_size
    if base_weight_bytes<1000000000:
        raise RuntimeError("Downloaded multilingual base weights unexpectedly incomplete")
    print("EXACT_PRETRAINED_MULTILINGUAL_BASE_SHA256",base_weight_sha,flush=True)
    model.config.use_cache=False
    model.generation_config.use_cache=True
    # No LoRA is attached yet. This measures actual multilingual pretrained
    # model behavior, rather than an untrained random student.
    base_before=evaluate(model,tokenizer,dev,n_per_route=6)
    lora=attach_shared_lora(
        model,rank=tcfg["lora_rank"],alpha=tcfg["lora_alpha"],
        encoder_last=tcfg["last_encoder_layers"],
        decoder_last=tcfg["last_decoder_layers"])
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=tcfg["learning_rate"],
                                weight_decay=tcfg["weight_decay"])
    losses=[]
    base_debug={
        "architectural_model_type":model.config.model_type,
        "vocab_size":model.config.vocab_size,
        "shared_tensor_shapes":{k:list(v.shape) for k,v in
                                only_adapter_state(model).items()},
        "single_shared_adapter_trainable_parameters":lora["trainable_parameters"],
        "old_marian_checkpoints":old,
    }
    # Never delete or reset anyone's historical model. New student starts
    # from multilingual translation-pretrained M2M100 weights and one adapter.
    for index,stage in enumerate(schedule):
        route=stage["route"]
        source,target=route.split("-")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batches,labels=encode_for_loss(
            tokenizer,stage["examples"],src=source,tgt=target,
            max_tokens=tcfg["token_max_length"])
        prediction=model(**batches,labels=labels)
        loss=prediction.loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss; refuse to save/publish bad adapter")
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(trainable,max_norm=1.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError("Nonfinite gradients")
        optimizer.step()
        loss_entry={"step":index+1,"route":route,"cross_entropy_loss":round(float(loss),6),
                    "gradient_norm":round(float(grad_norm),6)}
        losses.append(loss_entry)
        print("ACTUAL_MULTILINGUAL_OPTIMIZER_STEP",json.dumps(loss_entry),flush=True)
        step=index+1
        if step in tcfg["checkpoint_steps"]:
            from safetensors.torch import save_file
            OUTPUT.mkdir(parents=True,exist_ok=True)
            checkpoint=OUTPUT/f"shared_lora_step{step:02d}.safetensors"
            if checkpoint.exists():
                raise FileExistsError("Cannot overwrite existing student weights")
            save_file(only_adapter_state(model),str(checkpoint),metadata={
                "model":"facebook/m2m100_418M",
                "source_revision":cfg["model"]["revision"],
                "single_shared_adapter":"true",
                "step":str(step),"production_approved":"false"})
            print("SAVED_ACTUAL_SINGLE_MULTILINGUAL_LORA",json.dumps({
                "step":step,"sha256":sha256(checkpoint),
                "bytes":checkpoint.stat().st_size}),flush=True)
    if len(losses)!=tcfg["updates"]:
        raise RuntimeError("Optimizer did not complete configured number of updates")
    after=evaluate(model,tokenizer,dev,n_per_route=6)

    delta={
        route:round(after[route]["chrF_not_accuracy"]-
                    base_before[route]["chrF_not_accuracy"],4)
        for route in cfg["directions"]
    }
    # This automatic gate is necessary but NOT sufficient. Production gate is
    # NEVER met without paired independent bilingual meaning review.
    routes_pass_similarity=all(value>=-1.0 for value in delta.values())
    candidate=OUTPUT/f"shared_lora_step{tcfg['updates']:02d}.safetensors"
    candidate_sha=sha256(candidate)
    overall_status=("experimental_candidate_preserved_no_human_adequacy"
                    if routes_pass_similarity else
                    "experimental_regression_detected_checkpoint_not_promoted")
    # Deliberately keep the older actual production-provider runtime untouched.
    report={
        "status":"completed_actual_single_multilingual_student_training",
        "provenance":rights,
        "model_format":"one M2M100 base plus one shared learned LoRA for all 6 routes",
        "base_pretrained_revision":cfg["model"]["revision"],
        "base_pretrained_weights_sha256":base_weight_sha,
        "base_pretrained_weights_bytes":base_weight_bytes,
        "single_shared_weight_set":True,
        "trained_languages":["en","es","fr"],
        "trained_directions":cfg["directions"],
        "legacy_champions_unchanged":ensure_old_winners_untouched(),
        "optimizer_updates_completed":len(losses),
        "each_direction_trained":{route:sum(row["route"]==route for row in losses)
                                  for route in cfg["directions"]},
        "optimizer_seed":seed,
        "optimizer":"AdamW, new student optimizer; legacy Marian optimizers not reused",
        "loss_records":losses,
        "train_groups":data_receipt["train_groups"],
        "training_dataset_sha256":data_receipt["train_sha256"],
        "dev_dataset_sha256":data_receipt["dev_sha256"],
        "source_reviewed_500_used_for_training":False,
        "dataset_test_or_final_holdout_accessed":False,
        "automated_development_baseline":base_before,
        "automated_development_after_training":after,
        "dev_chrf_change_not_meaning_accuracy":delta,
        "all_directions_within_one_chrf_of_base":routes_pass_similarity,
        "research_checkpoint_status":overall_status,
        "checkpoint_relative_path":str(candidate.relative_to(ROOT)),
        "checkpoint_sha256":candidate_sha,
        "model_config_sha256":sha256(CONFIG),
        "lora_architecture":lora,
        "cpu_threads":torch.get_num_threads(),
        "torch_version":torch.__version__,
        "time_data_preparation_seconds":data_start-start,
        "training_and_evaluation_wall_time_seconds":time.monotonic()-start,
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "paid_spend_usd":0,
        "independently_verified_95_percent_meaning_accuracy":False,
        "production_activation_authorized":False,
        "old_translators_or_update_system_modified":False,
        "not_a_french_or_spanish_marian_weight_merge":True,
        "caveat":"This is a real learned shared multilingual neural network. "
                 "It does not prove no regressions against historical Marian models, "
                 "or independently validated slang, dialogue, or 95% meaning adequacy. "
                 "No production switch is permitted.",
    }
    data_receipt["model_base"] = rights
    data_receipt["trained_adapter_sha256"]=candidate_sha
    data_receipt["raw_source_text_published"]=False
    (OUTPUT/"data_manifest.json").write_text(
        json.dumps(data_receipt,indent=2,sort_keys=True)+"\n")
    (OUTPUT/"report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    list_files=sorted(p for p in OUTPUT.iterdir() if p.is_file())
    (OUTPUT/"SHA256SUMS.txt").write_text(
        "".join(sha256(p)+"  "+p.name+"\n" for p in list_files))
    print("REAL_ONE_MULTILINGUAL_MODEL_TRAINED",json.dumps({
        "status":report["status"],
        "optimizer_steps":len(losses),
        "trained_routes":report["each_direction_trained"],
        "research_checkpoint_status":overall_status,
        "checkpoint_sha256":candidate_sha,
        "dev_deltas_chrF_not_accuracy":delta,
        "any_final_human_95_claim":False,
        "spend_usd":0,
    }),flush=True)


if __name__=="__main__":
    run()
