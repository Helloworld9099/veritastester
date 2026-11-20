#!/usr/bin/env python3
"""Budget-limited REAL public synthetic-dialogue corpus inspection.

Read at most 8 MiB from the canonical 2.74GB MIT dataset using an exact
Hugging Face revision (discovered and recorded at execution); never read
private/final test data, publish source text or train on generated translations.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.es_en_v2.baby_dialogue import summarize_records

REPO_ID="lindazeng979/bilingual-babyLM"
DATAFILE="bilingual-babyLM.jsonl"
MAX_BYTES=8*1024*1024
MAX_GROUPS=1500


def run(out: Path):
    if out.exists():
        raise FileExistsError("Do not overwrite source manifests")
    from huggingface_hub import HfApi
    info=HfApi().dataset_info(repo_id=REPO_ID)
    revision=info.sha
    if not revision or len(revision)!=40:
        raise RuntimeError("Could not resolve immutable dataset git revision")
    name=DATAFILE
    url=f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}/{name}"
    started=time.monotonic()
    # urllib streams, and the request is closed immediately when the cap is hit.
    # Never obtain the 2.74GB raw source file or any external private corpus.
    with urlopen(Request(url,headers={
        "User-Agent":"BridgeCaption-v2-nontraining-open-dialogue-inspector/1.0"
    }),timeout=70) as response:
        data=response.read(MAX_BYTES)
    actual_bytes=len(data)
    if not data or actual_bytes>MAX_BYTES:
        raise RuntimeError("Inspected source exceeded configured 8 MiB cap")
    if b"\n" not in data:
        raise RuntimeError("Unrecognized newline JSONL source format")
    raw=data.rsplit(b"\n",1)[0]
    rows=[]
    for line in raw.splitlines():
        if not line.strip():
            continue
        doc=json.loads(line)
        if not isinstance(doc,dict):
            raise RuntimeError("Synthetic dialogue record not a JSON object")
        rows.append(doc)
        if len(rows)>=MAX_GROUPS:
            break
    if len(rows)<10:
        raise RuntimeError(f"Too few raw bilingual dialogues in bounded sample: {len(rows)}")
    summary=summarize_records(rows,max_groups=MAX_GROUPS)
    summary.update({
        "source_exact_revision":revision,
        "source_repo":REPO_ID,
        "source_file":DATAFILE,
        "inspected_bytes_cap":MAX_BYTES,
        "received_bytes":actual_bytes,
        "received_prefix_sha256":hashlib.sha256(data).hexdigest(),
        "elapsed_seconds":time.monotonic()-started,
        "new_human_verified_parallel_pairs":0,
        "model_training_started_from_synthetic_data":False,
        "no_actual_source_text_saved":True,
        "next_boundary":"Any synthetic candidate requires license, utterance alignment "
                        "and independent bilingual meaning review before supervised training.",
    })
    out.mkdir(parents=True,exist_ok=False)
    (out/"source_sample_manifest.json").write_text(json.dumps(summary,indent=2))
    print("REAL_SYNTHETIC_SOURCE_SAMPLE_INSPECTION",json.dumps(summary),flush=True)


if __name__=="__main__":
    run(ROOT/"research/es_en_v2/dataset_receipts/babylm")
