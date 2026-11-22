#!/usr/bin/env python3
"""Independent real INT8 MADLAD 3B CPU inference efficiency A/B benchmark.

Purpose: determine whether fast, quantized 425-code translation works well
enough for BridgeCaption's live captions BEFORE paying for more gradients.
Loads a pinned third-party CONVERSION, not pretrained Google model weights.
This is explicitly NOT a training experiment or a human accuracy study.
Never promote third-party weights or commercial licenses from card data alone.
"""
from __future__ import annotations

import json
import math
import os
import resource
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.madlad400.model import config,file_sha256
from research.conversation_growth.data import sampled_dev

MODEL_ID="cstr/madlad400-3b-ct2-int8"
EXP=ROOT/"research/madlad400/efficiency_001"
MAX_MODEL_BYTES=8000000000
ROUTES=("es-en","fr-en","en-es","en-fr","es-fr","fr-es")
SENTINELS={
    "en-de":"I need to reschedule the meeting.",
    "en-pt":"Don't forget your bag when you leave.",
    "en-ar":"Can you pick me up after work?",
    "en-zh":"Wait, I meant next Friday, not today.",
}


def safe_metadata(info)->dict:
    """Forbid weights with missing rights, huge sizes or unpinned revisions."""
    if str((info.cardData or {}).get("license","")).lower()!="apache-2.0":
        raise RuntimeError("INT8 conversion license is not Apache-2.0")
    sha=str(info.sha)
    if len(sha)!=40 or any(ch not in "0123456789abcdef" for ch in sha):
        raise RuntimeError("Third-party conversion requires immutable upstream revision")
    selected={}
    for row in info.siblings:
        name=str(row.rfilename)
        if name in {"model.bin","config.json","sentencepiece.model"}:
            selected[name]=int(row.size) if row.size else 0
    if set(selected)!={"model.bin","config.json","sentencepiece.model"}:
        raise RuntimeError("Expected complete CTranslate2 model binary, config and SentencePiece file")
    if any(v<=0 for v in selected.values()):
        raise RuntimeError("Could not check file sizes before download")
    if selected["model.bin"]<500000000 or sum(selected.values())>MAX_MODEL_BYTES:
        raise RuntimeError("Converted model size exceeds safe 3B RAM/disk budget")
    return {
        "converted_repo":MODEL_ID,
        "converted_commit":sha,
        "license":"apache-2.0",
        "candidate_model_size_bytes":sum(selected.values()),
        "file_sizes":selected,
        "conversion_not_verified_exact_against_google_pretrained":True,
        "source_provenance":"Third-party conversion claims to derive from google/madlad400-3b-mt; no parity assumed."
    }


def p95(items):
    if not items:return 0.0
    return sorted(items)[min(len(items)-1,math.ceil(len(items)*.95)-1)]


def evaluate_single_setting(translator,sp,dev,*,beam_width:int,per_route:int):
    from sacrebleu.metrics import CHRF,BLEU
    timings=[]
    scores={}
    for route in ROUTES:
        source,target=route.split("-")
        items=dev[route][:per_route]
        if len(items)!=per_route:
            raise RuntimeError("Not enough fresh publisher-development prompts")
        preds=[]
        for row in items:
            text=f"<2{target}> "+row["src"]
            tokens=sp.encode(text,out_type=str)
            if len(tokens)>72:
                raise ValueError("Dev source exceeds checked short ASR utterance window")
            started=time.monotonic()
            outputs=translator.translate_batch(
                [tokens],beam_size=beam_width,max_decoding_length=48,
                batch_type="tokens",max_batch_size=128,
                return_scores=False)
            took=time.monotonic()-started
            timings.append(took)
            pred=sp.decode(outputs[0].hypotheses[0]).strip()
            if not pred:
                raise RuntimeError("CT2 returned blank translation on development data")
            preds.append(pred)
        gold=[row["tgt"] for row in items]
        scores[route]={
            "n":len(items),"chrF_not_accuracy":round(CHRF().corpus_score(preds,[gold]).score,4),
            "BLEU_not_accuracy":round(BLEU().corpus_score(preds,[gold]).score,4),
            "first_pred":preds[0],
            "human_adequacy_verified":False,
            "publisher_development_only":True
        }
    return scores,timings


def run():
    if EXP.exists():
        raise FileExistsError("Do not overwrite real inference research evidence")
    cfg=config()
    if cfg["budget"]["max_paid_spend_usd"]!=0:
        raise RuntimeError("No paid inference permitted")
    if os.environ.get("BRIDGE_NO_PAID_APIS","true").lower()!="true":
        raise RuntimeError("Do not call paid inference APIs")
    from huggingface_hub import HfApi,snapshot_download
    from sentencepiece import SentencePieceProcessor
    import ctranslate2
    from sacrebleu.metrics import CHRF,BLEU
    began=time.monotonic()
    meta=HfApi().model_info(MODEL_ID,files_metadata=True)
    source=safe_metadata(meta)
    disk=shutil.disk_usage(os.environ.get("RUNNER_TEMP","/tmp")).free
    if disk < source["candidate_model_size_bytes"]+2*1000000000:
        raise RuntimeError("Insufficient disk to download/verify 3B INT8 conversion")
    model_folder=Path(snapshot_download(
        repo_id=MODEL_ID,revision=source["converted_commit"],
        allow_patterns=list(source["file_sizes"]),
        local_files_only=False))
    for filename,byte_count in source["file_sizes"].items():
        item=model_folder/filename
        if not item.is_file() or item.stat().st_size!=byte_count:
            raise RuntimeError("Pinned converted model asset size mismatch")
    source["files_sha256"]={
        filename:file_sha256(model_folder/filename) for filename in source["file_sizes"]}
    sp=SentencePieceProcessor(model_file=str(model_folder/"sentencepiece.model"))
    for code in ("en","es","fr","de","pt","ar","zh"):
        token="<2"+code+">"
        if sp.piece_to_id(token)==sp.unk_id():
            raise RuntimeError("Quantized model tokenizer lacks native target marker "+token)
    load_begin=time.monotonic()
    translator=ctranslate2.Translator(
        str(model_folder),device="cpu",compute_type="int8",
        intra_threads=3,inter_threads=1,max_queued_batches=2)
    load_seconds=time.monotonic()-load_begin
    dev,groups=sampled_dev(n=12)
    measurements={}
    for beam in (1,2):
        data,durations=evaluate_single_setting(translator,sp,dev,beam_width=beam,
                                               per_route=4)
        measurements["beam_"+str(beam)]={
            "per_route_metrics":data,
            "tested_utterances":len(durations),
            "mean_seconds":round(statistics.mean(durations),4),
            "median_seconds":round(statistics.median(durations),4),
            "p95_seconds":round(p95(durations),4),
            "worst_seconds":round(max(durations),4),
            "total_decode_seconds":round(sum(durations),4),
            "output_sample_not_independently_rated":True
        }
        print("REAL_MADLAD_INT8_CPU_BEAM_ABLATION",json.dumps({
            "beam":beam,"sample_count":len(durations),
            "p95_s":measurements["beam_"+str(beam)]["p95_seconds"],
            "mean_s":measurements["beam_"+str(beam)]["mean_seconds"],
            "human_accuracy":False
        }),flush=True)
    sentinel_results={}
    for route,source_text in SENTINELS.items():
        src,tgt=route.split("-")
        tokens=sp.encode("<2"+tgt+"> "+source_text,out_type=str)
        if len(tokens)>72:
            raise ValueError("Native sentinel source too long")
        start=time.monotonic()
        pred=sp.decode(translator.translate_batch(
            [tokens],beam_size=1,max_decoding_length=48)[0].hypotheses[0]).strip()
        if not pred:
            raise RuntimeError("Tokenizer-native route returned no words")
        sentinel_results[route]={
            "source_text":source_text,
            "translated":pred,
            "seconds":round(time.monotonic()-start,4),
            "no_verified_bilingual_reference":True
        }
    receipt={
        "status":"actual_free_MADLAD400_int8_CPU_inference_ablation_completed",
        "pinned_conversion":source,
        "upstream_google_revision":cfg["model"]["revision"],
        "actual_CTranslate2_version":ctranslate2.__version__,
        "actual_sentencepiece_version":"native_spm",
        "model_loads":1,
        "model_load_seconds":round(load_seconds,4),
        "beam_decoding_comparisons":measurements,
        "additional_native_language_sentinels":sentinel_results,
        "frozen_publisher_development_group_count":len(groups),
        "publisher_test_accessed":False,
        "old_reviewed_500_reused_for_training":False,
        "human_95_accuracy_verified":False,
        "production_activated":False,
        "new_training_updates":0,
        "file_format":"CTranslate2 3B INT8 pretrained conversion (no LoRA training)",
        "peak_rss_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds":round(time.monotonic()-began,2),
        "paid_spend_usd":0,
        "warning":"A third-party converted checkpoint is not proved exactly equivalent "
                  "to Google full-precision pretrained base. BLEU/chrF on 4 "
                  "samples per route and native target tags cannot establish "
                  "real-world conversational meaning or quality retention."
    }
    EXP.mkdir(parents=True)
    (EXP/"report.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    (EXP/"report.sha256").write_text(file_sha256(EXP/"report.json")+"  report.json\n")
    print("REAL_CT2_3B_INT8_CPU_SPEED_AND_DEVELOPMENT_EVIDENCE",json.dumps({
        "mean_beam1_s":measurements["beam_1"]["mean_seconds"],
        "mean_beam2_s":measurements["beam_2"]["mean_seconds"],
        "p95_beam1_s":measurements["beam_1"]["p95_seconds"],
        "rss_mib":round(receipt["peak_rss_kib_linux"]/1024,1),
        "permissive_license":source["license"],
        "meaning_accuracy_certified":False
    }),flush=True)


if __name__=="__main__":
    run()
