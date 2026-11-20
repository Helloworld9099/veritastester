#!/usr/bin/env python3
"""Bounded read-only inspection of original UTEP CC0 DRAL author archive.

Never download all 11GB. Scan at most 16 MiB of compressed tar to identify
whether bilingual text metadata precedes audio and whether metadata-only
dataset extraction is feasible within zero paid compute.

Only file names/aggregate counts; no speaker recordings/transcript text saved.
"""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from pathlib import Path
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"research/es_en_v2/dataset_receipts/dral"
URL="https://cs.utep.edu/nigel/dral/DRAL-16kHz.tgz"
CAP=16*1024*1024
MAX_MEMBERS=1000


def inspect():
    if OUT.exists():
        raise FileExistsError("Never overwrite DRAL provenance inspection")
    started=time.monotonic()
    req=Request(URL,headers={
        "User-Agent":"BridgeCaption-v2-CC0-metadata-readonly-inspection",
        "Range":f"bytes=0-{CAP-1}"})
    with urlopen(req,timeout=75) as response:
        status=response.status
        length=response.headers.get("Content-Length")
        data=response.read(CAP)
    if len(data)>CAP or not data:
        raise RuntimeError("Source exceeded bounded read or was empty")
    stats={"source":"UTEP original DRAL-16kHz", "publisher":"Nigel G. Ward et al.",
           "reported_license":"CC0 (original UTEP author distribution)",
           "LDC_repackaging_has_separate_terms":True,
           "original_url":URL,"compressed_read_cap_bytes":CAP,
           "actual_bytes_read":len(data),"http_status":status,
           "http_content_length_header":length,
           "sha256_compressed_prefix":hashlib.sha256(data).hexdigest(),
           "members_inspected":0,"metadata_text_members":[],"audio_members":0,
           "audio_bytes_downloaded_but_not_saved":len(data),
           "text_transcripts_saved":0,
           "full_archive_downloaded":False,
           "human_semantic_labels_collected":0,
           "safe_to_train_without_manual_source_rights_check":False}
    try:
        with tarfile.open(fileobj=io.BytesIO(data),mode="r|gz") as archive:
            for member in archive:
                if stats["members_inspected"]>=MAX_MEMBERS:
                    break
                stats["members_inspected"]+=1
                name=member.name
                if not member.isfile() or member.size<0:
                    continue
                if name.lower().endswith((".wav",".mp3",".flac",".ogg",".aiff")):
                    stats["audio_members"]+=1
                if name.lower().endswith((".csv",".tsv",".txt",".eaf",".json")) and member.size<=2*1024*1024:
                    try:
                        src=archive.extractfile(member)
                        if src is None:
                            continue
                        raw=src.read(2*1024*1024+1)
                        if len(raw)>2*1024*1024:
                            continue
                        heading=raw.splitlines()[0].decode("utf-8",errors="replace")[:160] if raw else ""
                        stats["metadata_text_members"].append({
                            "path":name,"size_bytes":member.size,
                            "line_count":len(raw.splitlines()),
                            "first_line_sha256_not_plaintext":hashlib.sha256(heading.encode()).hexdigest(),
                        })
                    except (OSError,UnicodeError):
                        pass
    except (tarfile.ReadError,EOFError,OSError) as error:
        stats["bounded_partial_archive_note"]=type(error).__name__
    stats["elapsed_seconds"]=time.monotonic()-started
    stats["has_small_text_metadata_early"]=bool(stats["metadata_text_members"])
    stats["next_step"]="Try metadata-only author release if it exists; do not ingest 11GB or restricted LDC version automatically."
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"metadata_probe.json").write_text(json.dumps(stats,indent=2))
    print("REAL_CC0_DRAL_16MIB_BOUNDED_METADATA_PROBE",json.dumps(stats),flush=True)


if __name__=="__main__":
    inspect()
