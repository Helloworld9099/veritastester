#!/usr/bin/env python3
"""Evaluate every saved shared multilingual continuation checkpoint on DEV.

This does not train more, overwrite any successful weights, touch final
holdout, or promote to production. Search among *saved* checkpoints from the
actual 60-step continuation after a poor final BLEU gate; retain the parent
if all intermediate candidates also regress.
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.multilingual_student.model import (
    read_config,load_shared_model,attach_shared_lora,load_adapter,sha256
)
from research.multilingual_student.data import prepare
from research.multilingual_student.train import evaluate,ensure_old_winners_untouched

EXP=ROOT/"research/multilingual_student/experiment_002"
REPORT=EXP/"checkpoint_selection.json"
PARENT_SHA="97c3a1b23cf7d290cecac038ead9fd8a6f810905944b6c441184c48418258f4b"
NAME=[f"shared_lora_extra{step:02d}.safetensors" for step in (20,40,60)]


def gate(old:dict,new:dict) -> tuple[bool,dict]:
    routes=set(old)
    if routes!=set(new) or len(routes)!=6:
        raise ValueError("All six directions must be scored together")
    chr_delta={r:round(new[r]["chrF_not_accuracy"]-
                       old[r]["chrF_not_accuracy"],4) for r in routes}
    bleu_delta={r:round(new[r]["BLEU_not_accuracy"]-
                        old[r]["BLEU_not_accuracy"],4) for r in routes}
    mean_chr=sum(chr_delta.values())/len(routes)
    blocked=[f"{route}:chrF" for route,v in chr_delta.items() if v<-.75]
    blocked += [f"{route}:BLEU" for route,v in bleu_delta.items() if v<-.75]
    if mean_chr < .25:
        blocked.append("mean_chrF_gain_less_than_.25")
    return not blocked,{
        "chrF_changes_not_accuracy":dict(sorted(chr_delta.items())),
        "BLEU_changes_not_accuracy":dict(sorted(bleu_delta.items())),
        "mean_chrF_delta_not_accuracy":round(mean_chr,4),
        "automatic_gate_blockers":sorted(blocked),
    }


def run():
    import torch
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    if REPORT.exists():
        raise FileExistsError("Never overwrite checkpoint selection result")
    latest=json.loads((EXP/"report.json").read_text())
    first=json.loads((ROOT/"research/multilingual_student/experiment_001/report.json").read_text())
    if (latest["status"]!="completed_real_shared_multilingual_student_continuation"
        or latest["parent_checkpoint_sha256"]!=PARENT_SHA
        or first["checkpoint_sha256"]!=PARENT_SHA):
        raise RuntimeError("Original trained checkpoint lineage is invalid")
    old=ensure_old_winners_untouched()
    _,dev,manifest=prepare()
    if manifest["dev_sha256"]!=first["dev_dataset_sha256"]:
        raise RuntimeError("Developer dataset changed")
    if latest["parent_weights_file_not_modified"] is not True:
        raise RuntimeError("Parent progress is not preserved")
    checks={}
    for line in (EXP/"SHA256SUMS.txt").read_text().splitlines():
        digest,name=line.split("  ")
        checks[name]=digest
    start=time.monotonic()
    model,tok=load_shared_model()
    cfg=read_config()
    attach_shared_lora(model,rank=cfg["training"]["lora_rank"],
                       alpha=cfg["training"]["lora_alpha"],
                       encoder_last=cfg["training"]["last_encoder_layers"],
                       decoder_last=cfg["training"]["last_decoder_layers"])
    baseline=latest["parent_development_12_per_direction"]
    evaluations={}
    for filename in NAME:
        path=EXP/filename
        expected=checks.get(filename)
        if not expected or sha256(path)!=expected:
            raise RuntimeError("Saved checkpoint hash mismatch: "+filename)
        load_adapter(model,path,sha=expected)
        scores=evaluate(model,tok,dev,n_per_route=12)
        approved,summary=gate(baseline,scores)
        evaluations[filename]={
            "sha256":expected,
            "no_regression_automatic_gate":approved,
            "development_summary":summary,
            "scores":scores,
            "meaning_preservation_human_verified":False,
        }
        print("EVALUATED_SAVED_ONE_MODEL_CANDIDATE",json.dumps({
            "checkpoint":filename,"sha256":expected,
            "accepted_on_automatic_dev_only":approved,
            "summary":summary}),flush=True)
    accepted=[(row["development_summary"]["mean_chrF_delta_not_accuracy"],name)
              for name,row in evaluations.items() if row["no_regression_automatic_gate"]]
    if accepted:
        _,name=max(accepted)
        selected=EXP/name
        selected_sha=evaluations[name]["sha256"]
        label="development_selected_continuation_not_production"
    else:
        selected=ROOT/"research/multilingual_student/experiment_001/shared_lora_step18.safetensors"
        selected_sha=PARENT_SHA
        label="all_continuations_rejected_parent_preserved"
    if sha256(selected)!=selected_sha:
        raise RuntimeError("Selected checkpoint not pinned")
    receipt={
        "status":"actual_saved_shared_multilingual_checkpoint_sweep_completed",
        "from_parent_sha256":PARENT_SHA,
        "selected_research_checkpoint_relative":str(selected.relative_to(ROOT)),
        "selected_research_sha256":selected_sha,
        "selection_reason":label,
        "evaluations":evaluations,
        "on_development_only":True,
        "full_six_directions_evaluated":True,
        "human_95_percent_meaning_accuracy_verified":False,
        "production_activated":False,
        "user_default_not_changed":True,
        "publisher_test_or_final_holdout_accessed":False,
        "reviewed_500_source_text_not_trained":True,
        "one_base_and_one_shared_adapter_per_candidate":True,
        "old_marian_models_untouched":ensure_old_winners_untouched(),
        "paid_cost_usd":0,
        "elapsed_seconds":time.monotonic()-start,
        "rss_peak_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitation":"Four candidate checkpoints compared on only 12 publisher development cases per "
                     "direction; scores have high variance and selection bias. Human adequacy remains "
                     "unmeasured, and old specialist language-model performance blocks deployment."
    }
    REPORT.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    print("SINGLE_MULTILINGUAL_CHECKPOINT_SWEEP_VERIFIED",json.dumps({
        "selected_sha256":selected_sha,"reason":label,
        "accepted_candidates":len(accepted),
        "mean_chrF_delta_for_selected":max(accepted)[0] if accepted else 0,
        "independent_human_accuracy_verified":False}),flush=True)


if __name__=="__main__":
    run()
