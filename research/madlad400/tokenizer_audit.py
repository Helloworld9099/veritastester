#!/usr/bin/env python3
"""Inspect real pinned MADLAD tokenizer and all native target tags before 12GB download."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.madlad400.model import (
    config,verify_model_card,supported_target_markers,language_input
)

OUT=ROOT/"research/madlad400/real_tokenizer_audit.json"


def run():
    if OUT.exists():
        raise FileExistsError("Do not overwrite existing native-target audit")
    from transformers import AutoTokenizer
    cfg=config()
    meta=verify_model_card()
    m=cfg["model"]
    tokenizer=AutoTokenizer.from_pretrained(
        m["repo_id"],revision=m["revision"],
        trust_remote_code=False,use_fast=True)
    native=supported_target_markers(tokenizer)
    supported_examples={}
    for target,text in (("en","Hola, ¿puedes ayudarme?"),
                        ("es","Can you help me?"),
                        ("fr","Can you help me?"),
                        ("de","Je voudrais un café."),
                        ("pt","I want to read the book.")):
        if target not in native:
            raise RuntimeError("Unexpected missing native MADLAD "+target)
        prefix=language_input(text,target,native)
        token_ids=tokenizer(prefix,add_special_tokens=True)["input_ids"]
        marker_id=tokenizer.convert_tokens_to_ids("<2"+target+">")
        if marker_id not in token_ids or marker_id==tokenizer.unk_token_id:
            raise RuntimeError("MADLAD language prefix was not encoded as expected")
        supported_examples[target]={
            "tag":"<2"+target+">","marker_token_id":marker_id,
            "actual_encoding_starts_with_marker":token_ids[0]==marker_id,
            "encoded_token_count":len(token_ids)}
    report={
        "status":"real_upstream_MADLAD_tokenizer_and_license_verified",
        "model":m["repo_id"],
        "revision":m["revision"],
        "license":meta["license"],
        "native_target_code_count":len(native),
        "native_target_codes":sorted(native),
        "actual_tested_tag_encodings":supported_examples,
        "finetuned_by_BridgeCaption":False,
        "human_semantic_accuracy_verified":False,
        "production_activated":False,
        "weights_downloaded_by_this_audit":False,
        "paid_spend_usd":0,
        "note":"Only native target prefix tokens, not independently translated languages."
    }
    OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("REAL_MADLAD_400_MODEL_NATIVE_LANGUAGE_COVERAGE_VERIFIED",json.dumps({
        "native_target_code_count":len(native),
        "all_five_target_prefixes_real":True,
        "upstream_model_sha":m["revision"],
        "approved_for_production":False}),flush=True)


if __name__=="__main__":
    run()
