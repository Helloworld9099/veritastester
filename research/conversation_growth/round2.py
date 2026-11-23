#!/usr/bin/env python3
"""One-model continuation: additive LoRA rank expansion, no overwritten history.

A rank-4 alpha-8 trained adapter is exactly embedded into a rank-8 alpha-16
adapter (both scales = 2). The old rank's gradients are masked. Only NEW rank
columns learn from authentic CC BY 2.0 FR Tatoeba bilingual pairs; MASSIVE
intent-matched but nonliteral utterances are NOT treated as aligned gold.
All candidates must beat the pinned parent on the same disjoint six-route DEV.
This is offline research, never a production deployment or human accuracy claim.
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
sys.path.insert(0, str(ROOT))
from research.conversation_growth.data import build_new_parallel_training, sampled_dev
from research.conversation_growth.train import acceptance
from research.multilingual_student.model import (
    attach_shared_lora, load_shared_model, only_adapter_state, read_config, sha256
)
from research.multilingual_student.train import (
    encode_for_loss, ensure_old_winners_untouched, evaluate
)

HERE = ROOT / "research/conversation_growth"
CFG = HERE / "config_round2.json"
OUT = HERE / "experiment_002"
ROUTES = ("es-en", "fr-en", "en-es", "en-fr", "es-fr", "fr-es")


def config():
    cfg = json.loads(CFG.read_text())
    tr = cfg["training"]
    if (cfg["schema_version"] != 1 or cfg["release_approved"] is not False
        or cfg["paid_spend_cap_usd"] != 0 or cfg["final_holdout_access"] is not False
        or cfg["parent"]["rank"] != 4 or cfg["parent"]["alpha"] != 8
        or tr["rank"] != 8 or tr["alpha"] != 16
        or tr["updates"] != sum(tr["sampling"].values())
        or set(tr["sampling"]) != set(ROUTES)
        or not tr["parent_rank_gradients_frozen"]
        or cfg["restrictions"]["forbid_retraining_on_reviewer_500"] is not True
        or cfg["restrictions"]["do_not_train_massive_intent_aligned_utterances_as_literal_gold"] is not True
        or tr["supervision"] != "licensed_real_parallel_Tatoeba_only"):
        raise ValueError("Unapproved configuration or non-additive training")
    if sorted(set(tr["checkpoints"])) != tr["checkpoints"] or tr["checkpoints"][-1] != tr["updates"]:
        raise ValueError("Invalid checkpoint schedule")
    return cfg


def embed_existing_rank4(shared_model, source_weights: dict, old_rank=4):
    """Exact LoRA contribution at initialization, with no averaging or merging."""
    import torch
    target_state = shared_model.state_dict()
    keys = {name for name in target_state if ".lora_" in name}
    if set(source_weights) != keys or len(keys) != 24:
        raise RuntimeError("Unexpected legacy adapter keys")
    with torch.no_grad():
        for name, old in source_weights.items():
            new = target_state[name]
            if name.endswith(".lora_A"):
                if old.ndim != 2 or old.shape[0] != old_rank or new.shape != (2*old_rank,old.shape[1]):
                    raise RuntimeError("Incompatible LoRA A shape")
                new[:old_rank].copy_(old)
            elif name.endswith(".lora_B"):
                if old.ndim != 2 or old.shape[1] != old_rank or new.shape != (old.shape[0],2*old_rank):
                    raise RuntimeError("Incompatible LoRA B shape")
                new[:,:old_rank].copy_(old)
                if not torch.count_nonzero(new[:,old_rank:]).item() == 0:
                    raise RuntimeError("New LoRA rank did not initialize to zero contribution")
            else:
                raise RuntimeError("Unexpected LoRA tensor key")
    return len(keys)


def lock_previously_learned_ranks(model, old_rank=4):
    """Only new ranks train; AdamW has zero decay, so old slices cannot drift."""
    import torch
    count = 0
    for module in model.modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        module.lora_A.register_hook(
            lambda grad: torch.cat([torch.zeros_like(grad[:old_rank]), grad[old_rank:]], dim=0)
        )
        module.lora_B.register_hook(
            lambda grad: torch.cat([torch.zeros_like(grad[:,:old_rank]), grad[:,old_rank:]], dim=1)
        )
        count += 1
    if count != 12:
        raise RuntimeError("Expected exactly 12 shared LoRA layer pairs")
    return count


def schedule_batches(corpora: dict, cfg: dict):
    """No previous labels reinterpreted as literal parallel gold."""
    seed = cfg["training"]["seed"]
    batches = []
    for pair in ("en-es", "en-fr", "es-fr"):
        first, second = pair.split("-")
        forward, reverse = f"{first}-{second}", f"{second}-{first}"
        pool = list(corpora[pair])
        random.Random(seed + sum(map(ord, pair))).shuffle(pool)
        used = set()
        for route in (forward, reverse):
            count = cfg["training"]["sampling"][route]
            items = []
            for row in pool:
                if row["row_id"] in used:
                    continue
                used.add(row["row_id"])
                a,b = (row["source"],row["target"]) if route == forward else (row["target"],row["source"])
                items.append({"src":a,"tgt":b,"group_id":pair+":"+row["row_id"],
                              "provenance":row["provenance"],"license":row["license"]})
                if len(items) == 2*count:
                    break
            if len(items) != 2*count:
                raise RuntimeError("Insufficient nonrepeating parallel pairs")
            for i in range(count):
                batches.append({"route":route, "items":items[i*2:i*2+2],
                                "origin":"Tatoeba_CC-BY-2.0-FR_parallel"})
    random.Random(seed).shuffle(batches)
    for idx, step in enumerate(batches,1):
        step["optimizer_step"] = idx
    if len(batches) != cfg["training"]["updates"]:
        raise RuntimeError("Bad optimizer schedule")
    return batches


def preserved_slices(model, original: dict, rank=4):
    import torch
    for key, weight in original.items():
        current = model.state_dict()[key]
        preserved = current[:rank] if key.endswith(".lora_A") else current[:,:rank]
        if not torch.equal(preserved.cpu(), weight.cpu()):
            raise RuntimeError("Previously trained LoRA slice changed: "+key)
    return True


def run():
    import torch
    from safetensors.torch import load_file, save_file
    cfg = config()
    if OUT.exists():
        raise FileExistsError("Refusing to overwrite prior experimental receipt")
    start = time.monotonic()
    p = ROOT / cfg["parent"]["path"]
    parent_sha = cfg["parent"]["sha256"]
    if sha256(p) != parent_sha:
        raise RuntimeError("Frozen 38-step adapter missing or modified")
    pinned = json.loads((ROOT/"research/multilingual_student/experiment_002/checkpoint_selection.json").read_text())
    if pinned["selected_research_sha256"] != parent_sha:
        raise RuntimeError("Not starting from selected previous best model")
    existing = ensure_old_winners_untouched()
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])
    corpora, manifest = build_new_parallel_training()
    old_manifest = json.loads((HERE/"experiment_001/data_manifest.json").read_text())
    for key, meta in manifest["sources"].items():
        if meta["archive_sha256"] != old_manifest["sources"][key]["archive_sha256"]:
            raise RuntimeError("Published upstream corpus changed")
    dev, groups = sampled_dev(cfg["frozen_dev"]["n_per_direction"])
    dev_hash = hashlib.sha256("|".join(sorted(str(g["id"]) for g in groups)).encode()).hexdigest()
    if dev_hash != cfg["frozen_dev"]["group_ids_sha256"]:
        raise RuntimeError("DEV group selection changed; cannot compare with parent")
    schedule = schedule_batches(corpora,cfg)
    model, tokenizer = load_shared_model()
    base = read_config()["training"]
    architecture = attach_shared_lora(
        model,rank=cfg["training"]["rank"],alpha=cfg["training"]["alpha"],
        encoder_last=base["last_encoder_layers"],decoder_last=base["last_decoder_layers"]
    )
    original = load_file(str(p))
    embed_existing_rank4(model,original)
    lock_previously_learned_ranks(model)
    baseline = evaluate(model,tokenizer,dev,n_per_route=len(groups))
    prior_report = json.loads((HERE/"experiment_001/report.json").read_text())
    for route in ROUTES:
        earlier = prior_report["baseline_dev_not_meaning_accuracy"][route]
        for metric in ("chrF_not_accuracy","BLEU_not_accuracy"):
            if abs(baseline[route][metric] - earlier[metric]) > 0.05:
                raise RuntimeError("Rank growth failed baseline functional equivalence: "+route)
    print("FROZEN_PARENT_FUNCTIONALLY_PRESERVED",json.dumps({
        "dev_n_per_direction":len(groups),"parent_sha256":parent_sha,
        "rank":cfg["training"]["rank"],"alpha":cfg["training"]["alpha"]
    }),flush=True)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=cfg["training"]["lr"],weight_decay=0.0)
    records = []
    checkpoints = {}
    for step in schedule:
        src,tgt = step["route"].split("-")
        inputs,labels = encode_for_loss(tokenizer,step["items"],src=src,tgt=tgt,max_tokens=96)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs,labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params,1.0)
        if not math.isfinite(float(grad_norm)):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        i = step["optimizer_step"]
        records.append({"step":i,"route":step["route"],"loss":round(float(loss),5),
                        "gradient_norm":round(float(grad_norm),5)})
        print("ADDITIVE_REAL_PARALLEL_LEARNING_UPDATE",json.dumps(records[-1]),flush=True)
        if i not in cfg["training"]["checkpoints"]:
            continue
        preserved_slices(model,original)
        OUT.mkdir(parents=True,exist_ok=True)
        name = f"shared_rank8_extra{i:02d}.safetensors"
        path = OUT/name
        if path.exists():
            raise FileExistsError("Refusing to overwrite checkpoint")
        save_file(only_adapter_state(model),str(path),metadata={
            "original_parent_sha256":parent_sha,
            "old_rank_parameters_frozen":"true","human_semantic_accuracy_verified":"false",
            "license":"CC-BY-2.0-FR_attribution_required"
        })
        scores=evaluate(model,tokenizer,dev,n_per_route=len(groups))
        eligible,gate=acceptance(baseline,scores,{}, {})
        checkpoints[name]={"sha256":sha256(path),"development":scores,
                           "gate":gate,"eligible_research_only":eligible,
                           "numeric_critical_error_proxy_untested":True}
        print("ADDITIVE_SIX_DIRECTION_DEV_GATE",json.dumps({
            "step":i,"eligible":eligible,"chrF_deltas":gate["chrF_deltas_not_accuracy"],
            "blocked_by":gate["blocked_by"]
        }),flush=True)
    preserved_slices(model,original)
    passed=[(sum(x["gate"]["chrF_deltas_not_accuracy"].values()),name)
            for name,x in checkpoints.items() if x["eligible_research_only"]]
    if passed:
        _,winner=max(passed)
        selection=f"research/conversation_growth/experiment_002/{winner}"
        selected_sha=checkpoints[winner]["sha256"]
        decision="new_rank8_dev_eligible_only_not_release"
    else:
        selection=cfg["parent"]["path"]
        selected_sha=parent_sha
        decision="previous_rank4_champion_preserved_all_candidates_rejected"
    report={
        "status":"actual_additive_rank8_training_completed",
        "actual_optimizer_updates":len(records),
        "old_learned_rank4_frozen":True,
        "parent_sha256":parent_sha,
        "parent_unchanged":sha256(p)==parent_sha,
        "previous_specialists_unchanged":ensure_old_winners_untouched()==existing,
        "model_id":cfg["parent"]["model"],
        "architecture":architecture,
        "sources":manifest["sources"],
        "source_counts":{"Tatoeba_parallel":len(records),"MASSIVE_intent_only_gold":0},
        "directions":dict(Counter(row["route"] for row in records)),
        "frozen_development_hash":dev_hash,
        "developer_groups_per_route":len(groups),
        "baseline_not_semantic_accuracy":baseline,
        "candidate_evaluations":checkpoints,
        "selected_path":selection,
        "selected_sha256":selected_sha,
        "checkpoint_decision":decision,
        "all_previous_runs_and_checkpoints_preserved":True,
        "reviewer_500_used_for_training":False,
        "human_semantic_95_percent_verified":False,
        "true_multiturn_context_gold_trained":False,
        "final_holdout_accessed":False,
        "production_translation_provider_changed":False,
        "paid_spend_usd":0,
        "wall_seconds":time.monotonic()-start,
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "training_records":records,
        "warning":"DEV chrF/BLEU are not accuracy percentages; 36 examples per direction "
                  "cannot establish nuanced slang/context or human meaning fidelity. "
                  "Numeric preservation gate had no eligible digit probes on this cohort."
    }
    if len(records)!=cfg["training"]["updates"] or len(checkpoints)!=len(cfg["training"]["checkpoints"]):
        raise RuntimeError("Incomplete genuine optimizer run")
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    manifest["parent_sha256"]=parent_sha
    manifest["real_tatoeba_only_and_no_intent_pseudogold"]=True
    manifest["dev_group_ids_sha256"]=dev_hash
    (OUT/"data_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    (OUT/"SHA256SUMS.txt").write_text("".join(
        sha256(x)+"  "+x.name+"\n" for x in sorted(OUT.iterdir()) if x.is_file()
    ))
    print("ACTUAL_ADDITIVE_SHARED_MODEL_ROUND2_COMPLETE",json.dumps({
        "steps":len(records),"decision":decision,"selected_sha256":selected_sha,
        "old_checkpoint_preserved":True,"release_approved":False
    }),flush=True)


if __name__ == "__main__":
    run()
