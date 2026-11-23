#!/usr/bin/env python3
"""Actually continue the existing *ONE* neural EN/ES/FR model with new data.

Warm start: SHA-pinned best 38-step M2M100 shared adapter. Never restart from
scratch, mix in unverifiable WMT24 Chat/NC corpora, or train on user-reviewed
500 examples. Add OPUS Tatoeba CC BY 2.0 FR bilingual authentic pairs plus a
small replay of previously licensed MASSIVE TRAIN voice commands. The initial
round is 72 CPU optimizer updates with a FROZEN, fresh publisher-VALIDATION
cohort. The model remains research-only absent bilingual human evaluation.
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
from research.conversation_growth.data import (
    settings,build_new_parallel_training,sampled_dev,normalize
)
from research.multilingual_student.data import prepare
from research.multilingual_student.model import (
    load_shared_model,attach_shared_lora,load_adapter,sha256,read_config,
    only_adapter_state,generate
)
from research.multilingual_student.train import (
    evaluate,encode_for_loss,ensure_old_winners_untouched
)

OUT=ROOT/"research/conversation_growth/experiment_001"
PARENT_SHA="a00c7965211853df07ce3fa032a76283dfc984dcdaeac85e03263445029ffb8e"
PARENT=ROOT/"research/multilingual_student/experiment_002/shared_lora_extra20.safetensors"


def train_batches(corpora,*,seed:int):
    cfg=settings()
    counts=cfg["training"]["sampling"]
    route_to_base={
        "en-es":"en-es","es-en":"en-es",
        "en-fr":"en-fr","fr-en":"en-fr",
        "es-fr":"es-fr","fr-es":"es-fr",
    }
    old_training,_,_=prepare()
    schedule=[]
    for route,count in counts.items():
        base=route_to_base[route]
        reverse=route!=base
        candidates=[]
        for row in corpora[base]:
            a,b=row["source"],row["target"]
            candidates.append({
                "src":b if reverse else a,
                "tgt":a if reverse else b,
                "group_id":base+":"+row["row_id"],
                "license":row["license"],"provenance":row["provenance"],
            })
        random.Random(seed+sum(map(ord,route))).shuffle(candidates)
        replay=list(old_training[route])
        random.Random(seed+100+sum(map(ord,route))).shuffle(replay)
        idx=0
        for i in range(count):
            # One in five steps replays permitted publisher TRAIN speech
            # commands to slow catastrophic forgetting of common requests.
            take_replay=(i%5==4)
            if take_replay:
                offset=i//5*2
                selected=replay[offset:offset+2]
                origin="MASSIVE_CC-BY-4.0_replay"
            else:
                selected=candidates[idx:idx+2]
                idx+=2
                origin="Tatoeba_CC-BY-2.0-FR_parallel"
            if len(selected)!=2:
                raise RuntimeError("Training attempted to reuse exhausted sample pool")
            schedule.append({"route":route,"items":selected,"origin":origin})
    rng=random.Random(seed)
    rng.shuffle(schedule)
    for i,step in enumerate(schedule,1):
        step["optimizer_step"]=i
    if len(schedule)!=cfg["training"]["updates"]:
        raise RuntimeError("Unbalanced training schedule")
    if set(row["route"] for row in schedule)!=set(counts):
        raise RuntimeError("No training for some direction")
    return schedule


def number_swap_proxy(model,tokenizer,dev):
    """High-confidence digit mismatch proxy; NOT full semantic error rate."""
    import re
    from collections import Counter
    digits=re.compile(r"(?<!\w)[0-9]+(?:[.,][0-9]+)?")
    probes=Counter()
    for direction,items in dev.items():
        src,tgt=direction.split("-")
        for sample in items:
            expected=digits.findall(sample["tgt"])
            original=digits.findall(sample["src"])
            if not (expected and original):
                continue
            predicted=generate(model,tokenizer,sample["src"],src,tgt,max_length=72)
            got=digits.findall(predicted)
            probes["eligible_paired_digit_cases"]+=1
            if got and got!=expected:
                probes["mismatched_digits_in_outputs"]+=1
            if not got:
                probes["unresolved_digit_word_or_omission_cases"]+=1
    return dict(probes)


def acceptance(parent,candidate,parent_proxy,candidate_proxy):
    cfg=settings()["gates"]
    delta={}
    bleu={}
    for route in sorted(parent):
        delta[route]=round(candidate[route]["chrF_not_accuracy"]-
                           parent[route]["chrF_not_accuracy"],4)
        bleu[route]=round(candidate[route]["BLEU_not_accuracy"]-
                          parent[route]["BLEU_not_accuracy"],4)
    fail=[route+":chrF" for route,value in delta.items()
          if value < -cfg["max_any_chrF_loss"]]
    fail += [route+":BLEU" for route,value in bleu.items()
             if value < -cfg["max_any_BLEU_loss"]]
    mean=sum(delta.values())/len(delta)
    if mean < cfg["overall_chrF_gain_min"]:
        fail.append("mean_chrF_gain_below_fixed_minimum")
    if (candidate_proxy.get("mismatched_digits_in_outputs",0)>
        parent_proxy.get("mismatched_digits_in_outputs",0)):
        fail.append("high_confidence_digit_swap_proxy_worse")
    return not fail,{"chrF_deltas_not_accuracy":delta,
                     "BLEU_deltas_not_accuracy":bleu,
                     "mean_chrF_delta":round(mean,4),
                     "blocked_by":fail,
                     "high_confidence_digit_proxy_only":True}


def run():
    import torch
    from safetensors.torch import save_file
    cfg=settings()
    if OUT.exists():
        raise FileExistsError("Refusing to overwrite previous research model")
    if sha256(PARENT)!=PARENT_SHA:
        raise RuntimeError("Existing preferred 38-step model weight changed")
    old=ensure_old_winners_untouched()
    selected=json.loads((ROOT/"research/multilingual_student/experiment_002/checkpoint_selection.json").read_text())
    if selected["selected_research_sha256"]!=PARENT_SHA:
        raise RuntimeError("Training must start from last actually selected research champion")
    seed=cfg["training"]["seed"]
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    began=time.monotonic()
    corpus,manifest=build_new_parallel_training()
    dev,dev_groups=sampled_dev(cfg["training"]["reserved_new_development_groups"])
    schedule=train_batches(corpus,seed=seed)
    manifest["new_frozen_development_group_ids_sha256"]=hashlib.sha256(
        "|".join(sorted(str(r["id"]) for r in dev_groups)).encode()).hexdigest()
    manifest["developer_examples_per_direction"]=len(dev_groups)
    manifest["sampled_training_schedule_sha256"]=hashlib.sha256(
        json.dumps([{"direction":r["route"],"origin":r["origin"],
                     "row_ids":[str(x["group_id"]) for x in r["items"]]}
                    for r in schedule],sort_keys=True).encode()).hexdigest()
    manifest["parent_adapter_sha256"]=PARENT_SHA
    print("NEW_COMMERCIALLY_LICENSED_PARALLEL_CORPUS_READY",json.dumps({
        "sources":manifest["sources"],
        "dev_new_groups":len(dev_groups),
        "source_final_holdout_accessed":False
    }),flush=True)
    model,tok=load_shared_model()
    mcfg=read_config()["training"]
    arch=attach_shared_lora(model,rank=mcfg["lora_rank"],alpha=mcfg["lora_alpha"],
                            encoder_last=mcfg["last_encoder_layers"],
                            decoder_last=mcfg["last_decoder_layers"])
    load_adapter(model,PARENT,sha=PARENT_SHA)
    baseline=evaluate(model,tok,dev,n_per_route=len(dev_groups))
    baseline_digits=number_swap_proxy(model,tok,dev)
    anchor=[p.detach().clone() for p in model.parameters() if p.requires_grad]
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=cfg["training"]["lr"],weight_decay=0.0)
    loss_history=[]
    snapshots={}
    for step in schedule:
        src,tgt=step["route"].split("-")
        inputs,labels=encode_for_loss(
            tok,step["items"],src=src,tgt=tgt,max_tokens=96)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output=model(**inputs,labels=labels)
        stability=sum(torch.mean((p-v)**2) for p,v in zip(trainable,anchor))/len(trainable)
        loss=output.loss+0.02*stability
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite shared model training loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
        if not math.isfinite(float(norm)):
            raise RuntimeError("Invalid gradient in multilingual training")
        optimizer.step()
        row={"update":step["optimizer_step"],"route":step["route"],
             "origin":step["origin"],"loss":round(float(loss),5),
             "grad_norm":round(float(norm),5)}
        loss_history.append(row)
        print("REAL_CONVERSATIONAL_GROWTH_OPTIMIZER_UPDATE",json.dumps(row),flush=True)
        if step["optimizer_step"] in cfg["training"]["checkpoint_updates"]:
            OUT.mkdir(parents=True,exist_ok=True)
            name=f"shared_tatoeba_extra{step['optimizer_step']:02d}.safetensors"
            ckpt=OUT/name
            if ckpt.exists():
                raise FileExistsError("No previous weights may be overwritten")
            save_file(only_adapter_state(model),str(ckpt),metadata={
                "parent":PARENT_SHA,"training_source":"Tatoeba_and_MASSIVE",
                "commercially_usable": "true","human_accuracy_verified":"false"})
            scores=evaluate(model,tok,dev,n_per_route=len(dev_groups))
            proxy=number_swap_proxy(model,tok,dev)
            ok,quality=acceptance(baseline,scores,baseline_digits,proxy)
            snapshots[name]={
                "sha256":sha256(ckpt),"bytes":ckpt.stat().st_size,
                "development":scores,"high_confidence_digit_proxy":proxy,
                "automatic_gate":quality,"eligible_for_research_only":ok}
            print("CANDIDATE_DEV_ONLY_REPLAY_BENCHMARK",json.dumps({
                "checkpoint":name,"sha":snapshots[name]["sha256"],
                "eligible_for_research":ok,
                "dev_quality":quality
            }),flush=True)
    if len(loss_history)!=cfg["training"]["updates"]:
        raise RuntimeError("Failed to finish full optimizer schedule")
    eligible=[(sum(x["automatic_gate"]["chrF_deltas_not_accuracy"].values()),name)
              for name,x in snapshots.items() if x["eligible_for_research_only"]]
    if eligible:
        _,name=max(eligible)
        selected_sha=snapshots[name]["sha256"]
        decision="new_development_candidate_only"
    else:
        name=None
        selected_sha=PARENT_SHA
        decision="parent_preserved_all_new_candidates_rejected"
    report={
        "status":"actual_additional_licensed_multilingual_training_completed",
        "model_family":"facebook/m2m100_418M",
        "parent_38_step_shared_sha256":PARENT_SHA,
        "parent_file_unchanged":sha256(PARENT)==PARENT_SHA,
        "actual_optimizer_updates":len(loss_history),
        "source_counts":dict(Counter(row["origin"] for row in loss_history)),
        "directions":dict(Counter(row["route"] for row in loss_history)),
        "new_source_dataset_sha256":{
            route:meta["archive_sha256"] for route,meta in manifest["sources"].items()},
        "new_development_uses_publisher_validation_not_test":True,
        "dev_group_count_per_route":len(dev_groups),
        "baseline_dev_not_meaning_accuracy":baseline,
        "baseline_digit_swap_proxy":baseline_digits,
        "all_candidate_evaluations":snapshots,
        "selected_relative_path":str((OUT/name).relative_to(ROOT)) if name else
                                 str(PARENT.relative_to(ROOT)),
        "selected_sha256":selected_sha,
        "checkpoint_decision":decision,
        "loss_history":loss_history,
        "old_spanish_french_specialists_preserved":ensure_old_winners_untouched(),
        "human_95_percent_meaning_accuracy_verified":False,
        "true_dialogue_context_trained":False,
        "authentic_parallel_tatoeba_sentences_used":True,
        "public_or_synthetic_only_no_private_speech_used":True,
        "noncommercial_or_unlicensed_data_used":False,
        "prior_reviewed_500_used_as_train_or_dev":False,
        "final_holdout_accessed":False,
        "default_live_app_modified":False,
        "paid_cost_usd":0,
        "wall_seconds":time.monotonic()-began,
        "max_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "warning":"Automatic chrF, BLEU and a narrow digit mismatch proxy are not "
                  "95% meaning preservation. These corpora are mostly isolated "
                  "utterances. Real conversational slang/context requires "
                  "conversation-grounded verified references and bilingual QA."
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"data_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (OUT/"SHA256SUMS.txt").write_text("".join(sha256(x)+"  "+x.name+"\n"
                                for x in sorted(OUT.iterdir()) if x.is_file()))
    print("COMPLETE_NEW_TATOEBA_PLUS_MASSIVE_SINGLE_MODEL_TRAINING",json.dumps({
        "optimizer_updates":len(loss_history),
        "selected_sha256":selected_sha,"decision":decision,
        "parent_preserved":True,
        "dev_n_per_route":len(dev_groups),"human_accuracy_95_claim":False
    }),flush=True)


if __name__=="__main__":
    run()
