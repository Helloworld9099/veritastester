"""Prepare licensed Mandarin parallel corpora for the existing M2M100 model.

Downloads OPUS Tatoeba Moses pairs, never publisher test data. Separates
TRAIN and frozen DEV by pair identity and normalized text across all routes.
Does not claim multi-turn context, verified license of every individual line,
or human translation proficiency.
"""
from __future__ import annotations
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path

from research.conversation_growth.data import (
    download, normalize, get_reviewed_sources, MAX_UNCOMPRESSED,
    MAX_COMPRESSED, LINKS, DIGITS,
)
from research.multilingual_student.model import ROOT
from research.multilingual_student.data import text_quality

PAIRS=("en-zh","es-zh","fr-zh")
VERSION="v2023-04-12"
SOURCE="https://tatoeba.org/en/downloads"
BASE="https://object.pouta.csc.fi/OPUS-Tatoeba/"+VERSION+"/moses/"
OUTPUT=ROOT/"research/mandarin_growth/experiment_001"
CJK=re.compile(r"[\u3400-\u9fff]")
MAX_PAIRS=450
DEV_PAIRS=60


def parse_moses_zip(blob:bytes,pair:str):
    if pair not in PAIRS:
        raise ValueError("Only audited Mandarin routes allowed")
    a,b=pair.split("-")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        files={}
        for lang in (a,b):
            candidates=[x for x in z.namelist()
                        if x.rsplit("/",1)[-1].startswith(f"Tatoeba.{pair}.") and x.endswith("."+lang)]
            if len(candidates)!=1:
                raise RuntimeError("Unexpected OPUS archive layout: "+pair+" "+lang)
            info=z.getinfo(candidates[0])
            if info.file_size>MAX_UNCOMPRESSED:
                raise RuntimeError("OPUS zip column exceeds safe size")
            with z.open(info) as f:
                raw=f.read(MAX_UNCOMPRESSED+1)
            if len(raw)>MAX_UNCOMPRESSED:
                raise RuntimeError("OPUS zip bomb limit")
            files[lang]=raw.decode("utf-8-sig").splitlines()
    if len(files[a])!=len(files[b]):
        raise RuntimeError("Parallel alignment row count mismatch")
    return list(zip(files[a],files[b]))


def robust_chinese_parallel(a: str, b: str) -> bool:
    """Chinese has no whitespace word delimiters; do not reuse Latin word count.

    Check source language word quality, Han content, bounded dimensions,
    URLs/PII and digit preservation. This is a *noise filter*, not independent
    confirmation that every translation is semantically correct.
    """
    if not text_quality(a) or not isinstance(b,str):
        return False
    if not 4 <= len(b) <= 200 or len(CJK.findall(b)) < 4:
        return False
    if LINKS.search(a) or LINKS.search(b) or "\\ufffd" in b:
        return False
    # Chinese characters are often much more compact than European text.
    ratio=len(b)/max(len(a),1)
    if not 0.18 <= ratio <= 2.5:
        return False
    if normalize(a)==normalize(b):
        return False
    left,right=DIGITS.findall(a),DIGITS.findall(b)
    if left and right and left!=right:
        return False
    return True


def choose(raw,pair,*,ban,seed,train_count=MAX_PAIRS,dev_count=DEV_PAIRS):
    if pair not in PAIRS or train_count<1 or dev_count<1:
        raise ValueError("Invalid Mandarin pair/budget")
    candidates=[]
    seen=set(ban)
    for index,(a,b) in enumerate(raw):
        a=a.strip()
        b=b.strip()
        if not robust_chinese_parallel(a,b):
            continue
        x,y=normalize(a),normalize(b)
        if x in seen or y in seen:
            continue
        seen.update((x,y))
        candidates.append({"id":pair+":"+str(index),"source":a,"target":b,
                           "pair":pair,"license":"CC-BY-2.0-FR",
                           "provenance":"Tatoeba via OPUS "+VERSION})
    candidates.sort(key=lambda row:hashlib.sha256(
        (str(seed)+"|"+row["id"]).encode()).hexdigest())
    if len(candidates)<train_count+dev_count:
        raise RuntimeError("Insufficient distinct Mandarin parallel pairs: "+pair)
    # Freeze DEV first; ensure no overlap with training or other language pairs.
    return candidates[dev_count:dev_count+train_count],candidates[:dev_count]


def prepare(*,downloader=download,seed=2026112203):
    if OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite frozen Mandarin corpus")
    reviewed=get_reviewed_sources()
    ban=set(reviewed)
    train=[]
    dev=[]
    sources={}
    for pair in PAIRS:
        url=BASE+pair+".txt.zip"
        blob=downloader(url)
        if len(blob)>MAX_COMPRESSED:
            raise RuntimeError("Oversized OPUS archive")
        raw=parse_moses_zip(blob,pair)
        tr,dv=choose(raw,pair,ban=ban,seed=seed)
        for row in tr+dv:
            ban.update((normalize(row["source"]),normalize(row["target"])))
        train+=tr
        dev+=dv
        sources[pair]={"url":url,"sha256":hashlib.sha256(blob).hexdigest(),
                       "raw_pairs":len(raw),"train_pairs":len(tr),"dev_pairs":len(dv),
                       "license":"CC-BY-2.0-FR","attribution":SOURCE}
    train_text={normalize(v) for row in train for v in (row["source"],row["target"])}
    dev_text={normalize(v) for row in dev for v in (row["source"],row["target"])}
    if train_text & dev_text or reviewed & (train_text|dev_text):
        raise RuntimeError("Train/dev or reviewed-source leakage")
    OUTPUT.mkdir(parents=True)
    for name,rows in (("train",train),("dev",dev)):
        with (OUTPUT/(name+".jsonl")).open("w",encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
    manifest={"status":"mandarin_parallel_corpus_prepared_not_trained",
              "base_model":"facebook/m2m100_418M","seed":seed,
              "sources":sources,"train_count":len(train),"dev_count":len(dev),
              "train_ids_sha256":hashlib.sha256(
                  "|".join(sorted(r["id"] for r in train)).encode()).hexdigest(),
              "dev_ids_sha256":hashlib.sha256(
                  "|".join(sorted(r["id"] for r in dev)).encode()).hexdigest(),
              "all_old_champions_untouched":True,
              "no_final_test_used":True,"no_reviewer_500_training":True,
              "human_semantic_accuracy_verified":False,
              "warning":"Sentence-level Tatoeba data are not dialogue/slang gold. Attribution required."}
    (OUTPUT/"manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return manifest


if __name__=="__main__":
    print(json.dumps(prepare(),ensure_ascii=False))
