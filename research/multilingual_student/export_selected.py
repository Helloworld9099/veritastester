#!/usr/bin/env python3
"""Export only the no-regression-selected shared student to one full weight file.

The earlier 18-step standalone model remains independently reproducible and
available. This separate candidate is 18 + 20 actual optimizer updates and is
NEVER activated as BridgeCaption's default without human adequacy evidence.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.multilingual_student.model import (
    load_shared_model,attach_shared_lora,load_adapter,generate,
    read_config,merge_into_one_model,sha256
)
from research.multilingual_student.train import ensure_old_winners_untouched

EXP=ROOT/"research/multilingual_student/experiment_002"
REPORT=EXP/"selected_standalone_export.json"
SAMPLE={
    "es-en":"Hola, ¿puedes ayudarme con la reserva?",
    "fr-en":"Bonjour, pouvez-vous m'aider avec la réservation ?",
    "en-es":"Hello, I need your help with my booking.",
    "en-fr":"Hello, I need your help with my booking.",
    "es-fr":"¿Puedes llegar antes de las ocho?",
    "fr-es":"Je voudrais réserver une table pour demain.",
}
OUTPUT_BASE="bridgecaption-standalone-selected-38-updates"


def run():
    import torch
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    if REPORT.exists():
        raise FileExistsError("Do not overwrite selected standalone export evidence")
    selected=json.loads((EXP/"checkpoint_selection.json").read_text())
    config=read_config()
    trained=json.loads((EXP/"report.json").read_text())
    if (selected["status"]!="actual_saved_shared_multilingual_checkpoint_sweep_completed"
        or selected["selection_reason"]!="development_selected_continuation_not_production"
        or selected["publisher_test_or_final_holdout_accessed"] is not False):
        raise RuntimeError("No verified nonregressing research continuation selected")
    adapter=ROOT/selected["selected_research_checkpoint_relative"]
    sha=selected["selected_research_sha256"]
    if not adapter.is_file() or sha256(adapter)!=sha or adapter.parent!=EXP:
        raise RuntimeError("Selected shared-model continuation missing/corrupted")
    if "extra20" not in adapter.name:
        raise RuntimeError("Only the verified 20-extra-update research checkpoint is allowed")
    first=json.loads((ROOT/"research/multilingual_student/experiment_001/report.json").read_text())
    if trained["parent_checkpoint_sha256"]!=first["checkpoint_sha256"]:
        raise RuntimeError("Original learned parent ancestry changed")
    began=time.monotonic()
    old=ensure_old_winners_untouched()
    model,tok=load_shared_model()
    lora=attach_shared_lora(model,rank=config["training"]["lora_rank"],
                            alpha=config["training"]["lora_alpha"],
                            encoder_last=config["training"]["last_encoder_layers"],
                            decoder_last=config["training"]["last_decoder_layers"])
    load_adapter(model,adapter,sha=sha)
    model.eval()
    before={}
    for direction,text in SAMPLE.items():
        source,target=direction.split("-")
        before[direction]=generate(model,tok,text,source,target)
    merged=merge_into_one_model(model)
    merged.eval()
    after={}
    for direction,text in SAMPLE.items():
        source,target=direction.split("-")
        after[direction]=generate(merged,tok,text,source,target)
    if before!=after:
        raise RuntimeError("Selected multilingual model differs before/after LoRA folding")
    out=Path(os.environ.get("RUNNER_TEMP","/tmp"))/OUTPUT_BASE
    if out.exists():
        raise FileExistsError("Standalone selected research model output exists already")
    out.mkdir(parents=True)
    merged.save_pretrained(str(out),safe_serialization=True,max_shard_size="4GB")
    tok.save_pretrained(str(out))
    files=sorted(x for x in out.iterdir() if x.is_file())
    weights=[p for p in files if p.suffix==".safetensors"]
    if len(weights)!=1 or weights[0].stat().st_size<1000000000:
        raise RuntimeError("Selected multilingual full model was not saved as ONE large weight file")
    inventory={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in files}
    result={
        "status":"full_single_standalone_multilingual_neural_model_exported",
        "variant":"selected_38_total_updates",
        "from_actual_learned_adapter_sha256":sha,
        "selected_after_18_plus_20_updates":True,
        "base_revision":config["model"]["revision"],
        "full_weight_file":weights[0].name,
        "full_weight_sha256":sha256(weights[0]),
        "full_weight_bytes":weights[0].stat().st_size,
        "trained_routes":config["directions"],
        "exact_text_generation_matches_adapter":True,
        "actual_pre_post_merge_texts":before,
        "files":inventory,
        "artifact_location":"Downloadable Actions artifact; not committed to Git history",
        "production_enabled":False,
        "trained_student_meaning_95_percent_independently_verified":False,
        "no_final_holdout_opened":True,
        "old_french_spanish_champions_unchanged":old,
        "paid_spend_usd":0,
        "elapsed_seconds":time.monotonic()-began,
        "warning":"Development selection on 12 utterances per direction is noisy; comparison "
                 "against the historical Marian models and independent human adequacy "
                 "is still required. The app's default provider is unchanged.",
    }
    REPORT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print("EXPORTED_SELECTED_ONE_MULTILINGUAL_WEIGHT_FILE",json.dumps({
        "sha256":result["full_weight_sha256"],
        "bytes":result["full_weight_bytes"],
        "parent_learned_adapter_sha256":sha,
        "all_six_before_after_exact":True,
        "human_95_verified":False,
        "elapsed_seconds":result["elapsed_seconds"]
    }),flush=True)


if __name__=="__main__":
    run()
