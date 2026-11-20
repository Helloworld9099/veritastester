#!/usr/bin/env python3
"""Controlled English->Spanish UNADAPTED Marian development + latency baseline.

English->Spanish has no proven-compatible previously learned LoRA adapter.
This is a deliberately labeled original-weight baseline, not a new training
round. Uses official MASSIVE VALIDATION only; never reads test/holdout cases.
"""
from __future__ import annotations

import hashlib
import json
import resource
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE="Helsinki-NLP/opus-mt-en-es"
AUTHOR="AmazonScience/massive"
REV="ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
PHRASES=[
    "Can you explain what happened yesterday?",
    "I am joining the meeting in five minutes.",
    "The movie was much better than I expected.",
    "I need another copy of the receipt.",
    "She hasn't called me back yet.",
    "Can you help me find the right train?",
    "I was joking, please do not take it literally.",
    "I would like to talk with the manager.",
    "Don't change the number on my reservation.",
    "Thank you for explaining the instructions.",
    "This song is really good.",
    "I'll be there as soon as I can.",
    "I am not sure what he meant.",
    "Let's try again tomorrow.",
    "Please give me two tickets.",
    "Are you free to talk after class?",
]


def digest(path):
    sha=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            sha.update(block)
    return sha.hexdigest()


def run(out: Path):
    if out.exists():
        raise FileExistsError("Never overwrite baseline report")
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi,snapshot_download
    from transformers import MarianMTModel,MarianTokenizer
    from sacrebleu.metrics import CHRF,BLEU

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t0=time.perf_counter()
    resolved=HfApi().model_info(SOURCE).sha
    if not resolved or len(resolved)!=40:
        raise RuntimeError("Could not pin exact publisher model revision")
    cache=Path(snapshot_download(
        repo_id=SOURCE,revision=resolved,
        allow_patterns=["config.json","generation_config.json","pytorch_model.bin",
                        "source.spm","target.spm","vocab.json",
                        "tokenizer_config.json","special_tokens_map.json"],
    ))
    files=[p for p in cache.iterdir() if p.is_file()]
    tokenizer=MarianTokenizer.from_pretrained(str(cache),local_files_only=True)
    model=MarianMTModel.from_pretrained(str(cache),local_files_only=True,
                                       use_safetensors=False,weights_only=True)
    model.eval()
    load_seconds=time.perf_counter()-t0

    times=[]
    def translate(texts):
        outputs=[]
        with torch.no_grad():
            for text in texts:
                t=time.perf_counter()
                x=tokenizer(text,return_tensors="pt",truncation=True,max_length=128)
                y=model.generate(**x,num_beams=4,max_new_tokens=96)
                outputs.append(tokenizer.decode(y[0],skip_special_tokens=True))
                times.append(time.perf_counter()-t)
        return outputs

    _=translate(PHRASES)
    times.clear()
    _=translate(PHRASES)
    latencies=sorted(times)
    latency={
        "n":len(latencies),"mean_seconds":statistics.mean(latencies),
        "p50_seconds":latencies[round((len(latencies)-1)*.5)],
        "p95_seconds":latencies[round((len(latencies)-1)*.95)],
    }

    english=[dict(r) for r in load_dataset(AUTHOR,"en-US",revision=REV,
                                            split="validation",trust_remote_code=True)]
    spanish=[dict(r) for r in load_dataset(AUTHOR,"es-ES",revision=REV,
                                            split="validation",trust_remote_code=True)]
    en={str(r["id"]):r for r in english}
    es={str(r["id"]):r for r in spanish}
    if set(en)!=set(es):
        raise RuntimeError("Publisher validation ID mismatch")
    ids=sorted(en,key=lambda k:hashlib.sha256(("bridge-v2-en-es-dev"+k).encode()).hexdigest())
    pairs=[]
    seen_src,seen_tgt=set(),set()
    for key in ids:
        a=str(en[key]["utt"]).strip();b=str(es[key]["utt"]).strip()
        if not a or not b or en[key]["intent"]!=es[key]["intent"]:
            continue
        sa,sb=a.casefold(),b.casefold()
        if sa in seen_src or sb in seen_tgt:
            continue
        seen_src.add(sa);seen_tgt.add(sb)
        pairs.append((key,a,b))
        if len(pairs)==128:break
    if len(pairs)!=128:
        raise RuntimeError("Not enough official validation pairs")
    predictions=translate([a for _,a,_ in pairs])
    references=[b for _,_,b in pairs]
    report={
        "status":"measured_real_en_to_es_original_marian_baseline",
        "model":SOURCE,"model_sha":resolved,
        "source_dataset":AUTHOR,"dataset_revision":REV,
        "source_license":"CC-BY-4.0",
        "model_license":"Apache-2.0",
        "direction":"en-es",
        "development_split":"publisher_validation_not_test",
        "development_case_count":len(pairs),
        "development_id_sha256":hashlib.sha256(
            "\n".join(k for k,_,_ in pairs).encode()).hexdigest(),
        "dev_chrf_not_accuracy":CHRF().corpus_score(predictions,[references]).score,
        "dev_bleu_not_accuracy":BLEU().corpus_score(predictions,[references]).score,
        "synthetic_translation_only_latency":latency,
        "model_load_seconds":load_seconds,
        "peak_resident_set_size_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model_parameters":sum(x.numel() for x in model.parameters()),
        "torch_version":torch.__version__,
        "model_file_sha256":{p.name:digest(p) for p in sorted(files)},
        "paid_spend_usd":0,
        "human_95_percent_adequacy_verified":False,
        "test_or_final_holdout_accessed":False,
        "live_full_audio_pipeline_benchmarked":False,
        "note":"Publisher validation references can be culturally localized, not literal. "
               "The baseline is a controlled original-weight comparator, never a new training round.",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"en_es_original_baseline.json").write_text(json.dumps(report,indent=2))
    print("EN_ES_REAL_BASELINE",json.dumps(report),flush=True)


if __name__=="__main__":
    run(ROOT/"research/es_en_v2/benchmarks/en_es")
