"""Reproducible CC BY 4.0 six-direction tri-lingual data preparation.

Uses only publisher TRAIN for optimization, publisher VALIDATION for development,
and never calls publisher TEST, previously reviewed 500 sources as training,
or any NC/research-only subtitle corpus. Alignment by original shared intent ID
is not independently verified semantic parallel gold.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path

from .model import ROOT, ROUTES, read_config, sha256

PAIR_DIRECTIONS=(("es","en"),("fr","en"),("en","es"),
                 ("en","fr"),("es","fr"),("fr","es"))
LOCALES={"en":"en-US", "es":"es-ES", "fr":"fr-FR"}
SPACE=re.compile(r"\s+")
NUMBERS=re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
EMAIL=re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
URL=re.compile(r"https?://\S+")
PHONE=re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")


def normalize(s:str) -> str:
    return SPACE.sub(" ",s.strip().casefold())


def text_quality(text:str) -> bool:
    return (isinstance(text,str) and 3 <= len(text.split()) <= 35 and
            4 <= len(text) <= 230 and
            not EMAIL.search(text) and not URL.search(text) and
            not PHONE.search(text) and "\ufffd" not in text)


def group_valid(texts:dict) -> bool:
    if set(texts) != {"en","es","fr"} or not all(text_quality(t) for t in texts.values()):
        return False
    if len({normalize(v) for v in texts.values()}) != 3:
        return False
    lengths=[len(v) for v in texts.values()]
    if max(lengths)>min(lengths)*3.2:
        return False
    # When numeric digits appear in two languages, do not train on a
    # different spelled quantity, time, or number.
    nums=[NUMBERS.findall(s) for s in texts.values()]
    nonempty=[items for items in nums if items]
    if len(nonempty)>=2 and any(items!=nonempty[0] for items in nonempty[1:]):
        return False
    return True


def hash_groups(groups:list[dict]) -> str:
    s="\n".join(str(row["id"])+"|"+normalize(row["en"])+"|"+
                normalize(row["es"])+"|"+normalize(row["fr"])
                for row in sorted(groups,key=lambda x:str(x["id"])))
    return hashlib.sha256(s.encode()).hexdigest()


def get_reviewed_sources() -> set[str]:
    cfg=read_config()["data"]
    path=ROOT / cfg["review_source"]
    manifest=json.loads((path.parent/"manifest.json").read_text())
    if manifest["actual_inference_executed"] is not True:
        raise RuntimeError("Review source must be an actual historical model export")
    expected=manifest["files_sha256"][path.name]
    if sha256(path)!=expected:
        raise RuntimeError("Already reviewed original 500-source CSV changed")
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=500:
        raise RuntimeError("Expected all 500 reviewed model utterances")
    return {normalize(row["Original spoken text"]) for row in rows}


def load_groups(split:str, *, dataset_loader=None) -> list[dict]:
    if split not in {"train","validation"}:
        raise ValueError("Publisher TEST/holdout is not permitted in student training")
    cfg=read_config()["data"]
    if dataset_loader is None:
        from datasets import load_dataset
        dataset_loader=load_dataset

    localized={}
    for lang,locale in LOCALES.items():
        loaded=dataset_loader(cfg["repo"],locale,revision=cfg["revision"],
                              split=split,trust_remote_code=True)
        byid={str(row["id"]):dict(row) for row in loaded}
        if len(byid)!=len(loaded):
            raise RuntimeError("Repeated publisher group ID")
        localized[lang]=byid
    ids=set.intersection(*(set(v) for v in localized.values()))
    groups=[]
    for ident in sorted(ids):
        rec={lang: localized[lang][ident] for lang in LOCALES}
        if len({str(x.get("intent")) for x in rec.values()}) != 1:
            continue
        text={lang:str(row.get("utt","")).strip() for lang,row in rec.items()}
        if not group_valid(text):
            continue
        groups.append({"id":ident,**text,
                       "intent":str(rec["en"].get("intent","")),
                       "scenario":str(rec["en"].get("scenario",""))})
    if len(groups)<150:
        raise RuntimeError("Too few publisher-aligned tri-lingual groups")
    return groups


def prepare(*, dataset_loader=None, exclude_reviewed:bool=True):
    cfg=read_config()
    reviewed=get_reviewed_sources() if exclude_reviewed else set()
    raw_train=load_groups("train",dataset_loader=dataset_loader)
    raw_dev=load_groups("validation",dataset_loader=dataset_loader)
    # Hold out ALL publisher-validation utterance strings, not just 12
    # selected examples. Source/target crossover is tested across languages.
    dev_texts={normalize(r[lang]) for r in raw_dev for lang in LOCALES}
    raw_train=[
        r for r in raw_train
        if all(normalize(r[lang]) not in reviewed | dev_texts for lang in LOCALES)
    ]
    seed=cfg["training"]["seed"]
    def sort_by_group(rows,tag):
        return sorted(rows,key=lambda r:hashlib.sha256(
            (str(seed)+"|"+tag+"|"+str(r["id"])).encode()).hexdigest())
    raw_dev=sort_by_group(raw_dev,"development")
    raw_train=sort_by_group(raw_train,"training")
    def unique_groups(rows,n,unavailable):
        picked=[]
        seen=set(unavailable)
        for r in rows:
            current={normalize(r[lang]) for lang in LOCALES}
            if current&seen:
                continue
            seen|=current
            picked.append(r)
            if len(picked)==n:
                break
        if len(picked)!=n:
            raise RuntimeError("Not enough unique publisher tri-lingual groups")
        return picked
    dev=unique_groups(raw_dev,cfg["data"]["number_of_dev_groups"],reviewed)
    training=unique_groups(raw_train,cfg["data"]["number_of_training_groups"],
                           reviewed | dev_texts)
    if {r["id"] for r in dev}&{r["id"] for r in training}:
        raise RuntimeError("Publisher document/conversation ID collision")
    selected_dev={normalize(r[lang]) for r in dev for lang in LOCALES}
    train_text={normalize(r[lang]) for r in training for lang in LOCALES}
    if selected_dev & train_text or reviewed & train_text or reviewed & selected_dev:
        raise RuntimeError("Reviewer or development data leaked to multilingual student")
    def rows_for_route(groups):
        return {
            src+"-"+tgt:[{"group_id":r["id"],"src":r[src],"tgt":r[tgt]}
                        for r in groups]
            for src,tgt in PAIR_DIRECTIONS
        }
    train=rows_for_route(training)
    development=rows_for_route(dev)
    if set(train)!=ROUTES or set(development)!=ROUTES:
        raise RuntimeError("Unexpected route count")
    manifest={
        "dataset":cfg["data"]["repo"],"revision":cfg["data"]["revision"],
        "license":cfg["data"]["license"],
        "training_split":"train","development_split":"validation",
        "final_test_accessed":False,
        "train_groups":len(training),"development_groups":len(dev),
        "directions":list(cfg["directions"]),
        "train_examples_per_direction":len(training),
        "dev_examples_per_direction":len(dev),
        "train_sha256":hash_groups(training),
        "dev_sha256":hash_groups(dev),
        "reviewed_500_never_train_or_dev":True,
        "group_aligned_source_target_intent":True,
        "group_disjoint_train_and_development":True,
        "exact_text_cross_split_overlap":False,
        "synthetic_training_pair_labels_used":False,
        "semantic_correctness_independently_reviewed":False,
        "no_nc_licensed_data_used":True,
        "quality_filter":"intent-ID, length/ratio, exact dedup, digit sanity, email/URL/phone screening",
        "limitations":"MASSIVE utterances are usually isolated voice commands; no real spoken multi-turn "
                       "slang context. Publisher locale alignment is not independent bilingual validation."
    }
    return train,development,manifest


def make_step_schedule(train:dict, *,seed:int,steps:int,batch_size:int=2):
    if set(train)!=ROUTES or steps<len(PAIR_DIRECTIONS):
        raise ValueError("A complete multilingual schedule must cover every direction")
    order=list(train.keys())
    schedule=[]
    positions={}
    for route in order:
        candidates=list(train[route])
        random.Random(seed+sum(map(ord,route))).shuffle(candidates)
        positions[route]=candidates
    for step in range(steps):
        route=order[step%len(order)]
        source=positions[route]
        start=(step//len(order))*batch_size
        if start+batch_size>len(source):
            raise ValueError("Too few training examples to supply route")
        batch=source[start:start+batch_size]
        schedule.append({"step":step+1,"route":route,"examples":batch})
    if set(r["route"] for r in schedule)!=ROUTES:
        raise RuntimeError("Missing training language direction")
    return schedule
