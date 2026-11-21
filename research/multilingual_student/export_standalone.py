#!/usr/bin/env python3
"""Merge the single shared M2M100 LoRA into ONE standalone ~1.94GB model.

Does not alter legacy French/Spanish or production. The enormous standalone
weight is deliberately a GitHub Actions artifact, never committed as 20+
copies or hidden inside application wheels. One file + tokenizer/config can be
loaded with M2M100.from_pretrained() without any per-language adapter/router.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.multilingual_student.model import (
    ROOT,attach_shared_lora,generate,load_shared_model,load_adapter,
    read_config,merge_into_one_model,sha256
)
from research.multilingual_student.train import ensure_old_winners_untouched

EXP=ROOT/"research/multilingual_student/experiment_001"
OUTPUT_REPORT=EXP/"standalone_export.json"
EXAMPLES={
    "es-en":"Hola, ¿puedes ayudarme con la reserva?",
    "fr-en":"Bonjour, pouvez-vous m'aider avec la réservation ?",
    "en-es":"Hello, I need your help with my booking.",
    "en-fr":"Hello, I need your help with my booking.",
    "es-fr":"¿Puedes llegar antes de las ocho?",
    "fr-es":"Je voudrais réserver une table pour demain.",
}


def run():
    import torch
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    if OUTPUT_REPORT.exists():
        raise FileExistsError("A standalone export has already been recorded")
    if not EXP.exists():
        raise FileNotFoundError("Train the actual shared student first")
    cfg=read_config()
    previous=ensure_old_winners_untouched()
    report=json.loads((EXP/"report.json").read_text())
    if report["status"]!="completed_actual_single_multilingual_student_training":
        raise RuntimeError("Not a complete trained student")
    if report["independently_verified_95_percent_meaning_accuracy"] is not False:
        raise RuntimeError("Reject falsely certified trained output")
    adapter=EXP/f"shared_lora_step{report['optimizer_updates_completed']:02d}.safetensors"
    if sha256(adapter)!=report["checkpoint_sha256"]:
        raise RuntimeError("Shared adapter checkpoint changed")
    start=time.monotonic()
    model,tok=load_shared_model()
    lora=attach_shared_lora(
        model,rank=cfg["training"]["lora_rank"],
        alpha=cfg["training"]["lora_alpha"],
        encoder_last=cfg["training"]["last_encoder_layers"],
        decoder_last=cfg["training"]["last_decoder_layers"])
    if lora["trainable_parameters"]!=report["lora_architecture"]["trainable_parameters"]:
        raise RuntimeError("Shared LoRA layout changed before merge")
    load_adapter(model,adapter,sha=report["checkpoint_sha256"])
    model.eval()
    before={}
    for route,text in EXAMPLES.items():
        s,t=route.split("-")
        before[route]=generate(model,tok,text,s,t)
    merged=merge_into_one_model(model)
    merged.eval()
    after={}
    for route,text in EXAMPLES.items():
        s,t=route.split("-")
        after[route]=generate(merged,tok,text,s,t)
    if before!=after:
        raise RuntimeError("Full shared LoRA merge changed generated text on a route")

    scratch=Path(os.environ.get("RUNNER_TEMP","/tmp"))/"bridgecaption-standalone-trilingual"
    if scratch.exists():
        raise FileExistsError("Refuse to overwrite existing exported standalone model")
    scratch.mkdir(parents=True)
    merged.save_pretrained(str(scratch),safe_serialization=True,max_shard_size="4GB")
    tok.save_pretrained(str(scratch))
    files=sorted(path for path in scratch.iterdir() if path.is_file())
    weights=[path for path in files if path.suffix==".safetensors"]
    if len(weights)!=1:
        raise RuntimeError("Expected exactly ONE standalone M2M100 neural weight file")
    if weights[0].stat().st_size<1000000000:
        raise RuntimeError("Unexpectedly incomplete 418M pretrained checkpoint")
    modelfiles={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}
                for p in files}
    export={
        "status":"full_single_standalone_multilingual_neural_model_exported",
        "architecture":"M2M100",
        "model_family":cfg["model"]["id"],
        "base_revision":cfg["model"]["revision"],
        "original_trained_shared_adapter_sha256":report["checkpoint_sha256"],
        "full_weight_file":weights[0].name,
        "full_weight_sha256":sha256(weights[0]),
        "full_weight_bytes":weights[0].stat().st_size,
        "trained_routes":cfg["directions"],
        "exact_text_generation_matches_adapter":before==after,
        "actual_pre_post_merge_texts":before,
        "files":modelfiles,
        "artifact_location":"GitHub Actions run artifact; too large for Git history",
        "production_enabled":False,
        "trained_student_meaning_95_percent_independently_verified":False,
        "old_french_spanish_champions_unchanged":previous,
        "paid_spend_usd":0,
        "elapsed_seconds":time.monotonic()-start,
        "no_final_holdout_opened":True,
    }
    (OUTPUT_REPORT).write_text(json.dumps(export,indent=2,ensure_ascii=False)+"\n")
    print("EXPORTED_REAL_SINGLE_MULTILINGUAL_NEURAL_WEIGHT_FILE",json.dumps({
        "file":export["full_weight_file"],
        "bytes":export["full_weight_bytes"],
        "sha256":export["full_weight_sha256"],
        "six_directions":export["trained_routes"],
        "all_six_prepost_merge_outputs_identical":export["exact_text_generation_matches_adapter"],
        "human_adequacy_claim":False,
        "elapsed_seconds":export["elapsed_seconds"],
    }),flush=True)


if __name__=="__main__":
    run()
