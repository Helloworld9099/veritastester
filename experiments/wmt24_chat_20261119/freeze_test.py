#!/usr/bin/env python3
"""Freeze real WMT24 French chat TEST metadata before any BridgeCaption inference.

No training, no model inference, no raw sentences or references in artifacts.
The official CC BY-NC 4.0 test is RESEARCH-ONLY; not for commercial use.
"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import io
import json
from urllib.request import Request,urlopen

SRC_REV="12a34149267dd9ae98e60e0ec2e415707fad5618"
GOLD_REV="394632f24291d1edcad035636b922c3e3d4dfb01"
SOURCE="https://raw.githubusercontent.com/WMT-Chat-task/chat-task-2024-data/"+SRC_REV+"/test/en-fr.csv"
REFERENCES="https://raw.githubusercontent.com/WMT-Chat-task/chat-task-2024-results/"+GOLD_REV+"/test/en-fr.csv"
SOURCE_FIELDS=("source_language","target_language","source","doc_id","client_id","sender")
REFERENCE_FIELDS=("source_language","target_language","source","reference","doc_id","client_id","sender","tags")
FRENCH_MIN=600
TOTAL_EXPECTED=2091


def checksum(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def load_csv(content:bytes,columns:tuple[str,...]):
    reader=csv.DictReader(io.StringIO(content.decode("utf-8-sig")),
                          delimiter=",",quotechar='"')
    if tuple(reader.fieldnames or ())!=columns:
        raise ValueError("Official WMT24 chat CSV schema changed")
    rows=list(reader)
    if any(None in row or not row.get("source") or row.get("source")=="" for row in rows):
        raise ValueError("Malformed WMT24 CSV source")
    return rows


def freeze_rows(original:list[dict], gold:list[dict]):
    if len(original)!=TOTAL_EXPECTED or len(gold)!=TOTAL_EXPECTED:
        raise ValueError("Official WMT24 en-fr test row count changed")
    groups, subjects, directions,source_digests,case_hashes=set(),set(),Counter(),set(),set()
    per_direction_groups={"fr-en":set(),"en-fr":set()}
    fr_en_count=0
    for i,(source,ref) in enumerate(zip(original,gold)):
        for col in SOURCE_FIELDS:
            if source[col]!=ref[col]:
                raise ValueError("Gold reference row mismatch at index "+str(i))
        if source["source_language"]=="fr" and source["target_language"]=="en":
            direction="fr-en"
        elif source["source_language"]=="en" and source["target_language"]=="fr":
            direction="en-fr"
        else:
            raise ValueError("Unsupported WMT24 chat language direction")
        if not isinstance(ref.get("reference"),str) or not ref["reference"].strip():
            raise ValueError("Gold test reference unexpectedly empty")
        if source["sender"] not in ("agent","customer"):
            raise ValueError("Unexpected sender role")
        doc=source["doc_id"]
        if not doc or not source["client_id"]:
            raise ValueError("Unidentifiable conversation grouping")
        groups.add(doc)
        subjects.add(source["sender"])
        per_direction_groups[direction].add(doc)
        directions[direction]+=1
        # Hash every source, not a copy of the original text/reference.
        source_digests.add(checksum(source["source"].encode("utf-8")))
        case_hashes.add(checksum(
            json.dumps([i,doc,direction,source["source"],ref["reference"]],
                       ensure_ascii=False,separators=(",",":")).encode("utf-8")))
        if direction=="fr-en":
            fr_en_count+=1
    if not (len(groups)>=60 and fr_en_count>=FRENCH_MIN and directions["en-fr"]>0):
        raise ValueError("Unexpected language group diversity or content shortage")
    if len(case_hashes)!=TOTAL_EXPECTED:
        raise ValueError("Duplicate chat test cases")
    return {
        "source":"WMT-Chat-task/chat-task-2024-data",
        "source_commit":SRC_REV,
        "human_reference_source":"WMT-Chat-task/chat-task-2024-results",
        "human_reference_commit":GOLD_REV,
        "original_publisher_split":"test",
        "total_test_segments":len(original),
        "translation_directions":dict(directions),
        "french_to_english_segments":fr_en_count,
        "conversation_groups":len(groups),
        "french_to_english_groups":len(per_direction_groups["fr-en"]),
        "group_ids_sha256":checksum("\n".join(sorted(groups)).encode()),
        "all_case_hashes_sha256":checksum("\n".join(sorted(case_hashes)).encode()),
        "all_source_hashes_sha256":checksum("\n".join(sorted(source_digests)).encode()),
        "license":"CC-BY-NC-4.0 — noncommercial research only; no model training",
        "test_policy":"Freeze test before any Stage8 scoring; NEVER train, adapt, select, tune, or prompt-optimize on this WMT24 chat test.",
        "previous_legacy_evaluations":"No previously scored WMT24 2024 CHAT TEST found in inspected project audit, but overlap with original model pretraining and other unindexed runs cannot be fully excluded.",
        "human_rater_note":"The released gold translations are human-written references, NOT bilingual human adequacy ratings of BridgeCaption outputs.",
        "model_evaluation_performed":False,
        "90_percent_human_adequacy_verified":False,
    }


def download_bytes(url:str,limit=2_000_000):
    with urlopen(Request(url,headers={"User-Agent":"BridgeCaption-research/1.0"}),
                 timeout=45) as handle:
        raw=handle.read(limit+1)
    if not raw or len(raw)>limit:
        raise ValueError("External WMT24 chat test source changed size")
    return raw


def main():
    source=download_bytes(SOURCE)
    gold=download_bytes(REFERENCES)
    doc=freeze_rows(load_csv(source,SOURCE_FIELDS),load_csv(gold,REFERENCE_FIELDS))
    doc["source_csv_sha256"]=checksum(source)
    doc["gold_csv_sha256"]=checksum(gold)
    # Hash-only manifest: NO raw customer conversations or gold reference texts.
    print("WMT24_CHAT_NEW_FROZEN_TEST_METADATA")
    print(json.dumps(doc,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
