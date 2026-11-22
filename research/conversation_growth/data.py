"""Commercial-license-screened *parallel* new data for single multilingual model.

This module never opens MASSIVE test, final holdout, or already-reviewed
500 responses as supervised examples. Tatoeba sentences are sourced under
CC BY 2.0 FR (attribution mandatory) via OPUS v2023-04-12. They are naturally
short but do NOT establish multi-turn slang or conversational competence.
OpenAssistant is monolingual conversational data and NEVER treated as
parallel translation ground truth.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path

from research.multilingual_student.model import ROOT,sha256
from research.multilingual_student.data import (
    load_groups,prepare,get_reviewed_sources,normalize,text_quality
)

CONFIG=ROOT/"research/conversation_growth/config.json"
MAX_COMPRESSED=70*1024*1024
MAX_UNCOMPRESSED=160*1024*1024
DIGITS=re.compile(r"(?<!\w)[0-9]+(?:[.,][0-9]+)?")
LINKS=re.compile(r"(?:https?://|www\.|[\w.+-]+@[\w.-]+\.[a-z]{2,})",re.I)


def settings()->dict:
    cfg=json.loads(CONFIG.read_text())
    if (cfg["schema_version"]!=1 or cfg["release_approved"] is not False
        or cfg["paid_spend_cap_usd"]!=0
        or cfg["final_holdout_access"] is not False
        or cfg["reviewed_500_as_training"] is not False):
        raise ValueError("Unapproved conversational data/train policy")
    if len(cfg["parallel_sources"])!=3 or any(
        x["license"]!="CC-BY-2.0-FR" or
        x["training_permission"]!="allowed_with_attribution"
        for x in cfg["parallel_sources"]
    ):
        raise ValueError("Only verified commercially usable source license is allowed")
    if any("OpenAssistant" in item["id"] and "review" not in item["use"]
           for item in cfg["nonparallel_sources"]):
        raise ValueError("Monolingual conversational messages are not gold translation")
    return cfg


def download(url:str)->bytes:
    req=urllib.request.Request(url,headers={
        "User-Agent":"BridgeCaption/0.1 (legal parallel-corpus quality research)",
        "Accept":"application/zip,application/octet-stream,*/*"
    })
    with urllib.request.urlopen(req,timeout=70) as response:
        raw=response.read(MAX_COMPRESSED+1)
    if len(raw)>MAX_COMPRESSED:
        raise ValueError("OPUS Tatoeba zip exceeded bounded download limit")
    return raw


def pairs_from_zip(data:bytes,pair:str)->list[tuple[str,str]]:
    """Read ONLY the two expected Moses text columns; no unrelated data."""
    if pair not in {"en-es","en-fr","es-fr"}:
        raise ValueError("Unapproved source pair")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=archive.namelist()
        first,second=pair.split("-")
        exact={lang:[n for n in names if n.endswith("."+lang) and
                     n.rsplit("/",1)[-1].startswith("Tatoeba."+pair+".")]
               for lang in (first,second)}
        if any(len(v)!=1 for v in exact.values()):
            raise RuntimeError("OPUS archive does not contain exactly two aligned Tatoeba text columns")
        files={}
        for lang,names in exact.items():
            item=archive.getinfo(names[0])
            if item.file_size>MAX_UNCOMPRESSED:
                raise ValueError("OPUS archive column exceeded safe size")
            with archive.open(item) as stream:
                raw=stream.read(MAX_UNCOMPRESSED+1)
            if len(raw)>MAX_UNCOMPRESSED:
                raise ValueError("Zip bomb rejected")
            files[lang]=raw.decode("utf-8-sig").splitlines()
        if len(files[first])!=len(files[second]):
            raise RuntimeError("OPUS aligned columns have unequal row counts")
        return list(zip(files[first],files[second]))


def robust_filter(a:str,b:str)->bool:
    if not text_quality(a) or not text_quality(b):
        return False
    if LINKS.search(a) or LINKS.search(b):
        return False
    if max(len(a),len(b)) > min(len(a),len(b))*2.8:
        return False
    if normalize(a)==normalize(b):
        return False
    nums_a=DIGITS.findall(a)
    nums_b=DIGITS.findall(b)
    if nums_a and nums_b and nums_a!=nums_b:
        return False
    return True


def sample_pairs(raw:list[tuple[str,str]],pair:str,limit:int,ban:set[str],seed:int)->list[dict]:
    """Deterministic source+target leakage screen and near-copy dedup."""
    if limit<=0 or limit>1500:
        raise ValueError("Unbounded corpus sample request")
    if pair not in {"en-es","en-fr","es-fr"}:
        raise ValueError("Unapproved training route")
    order=sorted(enumerate(raw),key=lambda it:hashlib.sha256(
        (str(seed)+"|"+pair+"|"+str(it[0])).encode()).hexdigest())
    result=[]
    seen=set()
    for position,(a,b) in order:
        a=a.strip()
        b=b.strip()
        if not robust_filter(a,b):
            continue
        x,y=normalize(a),normalize(b)
        if x in ban or y in ban or x in seen or y in seen:
            continue
        # Strip punctuation and compare again; avoids training trivial
        # case/punctuation duplicates, which inflate held-out metrics.
        bare=(re.sub(r"[\W_]+","",x),re.sub(r"[\W_]+","",y))
        if not all(bare) or any(key in seen for key in bare):
            continue
        seen.update((x,y,*bare))
        result.append({"source":a,"target":b,"pair":pair,
                       "row_id":str(position),"license":"CC-BY-2.0-FR",
                       "provenance":"Tatoeba via OPUS v2023-04-12"})
        if len(result)==limit:
            break
    if len(result)<max(32,limit//2):
        raise RuntimeError("Not enough eligible new licensed parallel pairs")
    return result


def build_new_parallel_training(*,downloader=download):
    cfg=settings()
    reviewed=get_reviewed_sources()
    _,_,old_manifest=prepare()
    source_dev=load_groups("validation")
    dev_strings={normalize(item[lang]) for item in source_dev
                 for lang in ("en","es","fr")}
    forbidden=reviewed | dev_strings
    archive_meta={}
    pairs={}
    for src in cfg["parallel_sources"]:
        data=None
        errors=[]
        for url in src["download_urls"]:
            try:
                candidate=downloader(url)
                raw=pairs_from_zip(candidate,src["pair"])
                data=candidate
                selected=sample_pairs(raw,src["pair"],
                                      src["max_selected_pairs"],forbidden,
                                      cfg["training"]["seed"])
                archive_meta[src["pair"]]={
                    "url":url,"archive_sha256":hashlib.sha256(data).hexdigest(),
                    "compressed_bytes":len(data),"raw_pairs":len(raw),
                    "accepted_pairs":len(selected),
                    "license":src["license"],
                    "original_source":src["original_corpus"],
                    "attribution_required":True,
                    "publisher_train_or_test_split":
                        "OPUS Tatoeba released parallel sentence list; NOT an unseen final test"
                }
                pairs[src["pair"]]=selected
                break
            except (RuntimeError,ValueError,OSError,TimeoutError,zipfile.BadZipFile) as exc:
                errors.append(type(exc).__name__+": "+str(exc)[:130])
        if src["pair"] not in pairs:
            raise RuntimeError("All OPUS Tatoeba mirrors failed for "+src["pair"]+
                               ". Reasons: "+"; ".join(errors))
    all_text=set()
    for source in pairs.values():
        for row in source:
            if normalize(row["source"]) in forbidden or normalize(row["target"]) in forbidden:
                raise RuntimeError("Final holdout or previous reviewed source contamination")
            all_text.update((normalize(row["source"]),normalize(row["target"])))
    provenance={
        "status":"new_licensed_parallel_corpus_compiled_no_final_test",
        "sources":archive_meta,
        "source_license":"Tatoeba CC BY 2.0 FR, original contributors",
        "source_attribution_url":"https://tatoeba.org/en/downloads",
        "allowed_to_retrain_only_in_research":True,
        "developer_validation_hash_previous":old_manifest["dev_sha256"],
        "all_original_publisher_validation_excluded":True,
        "all_500_reviewed_sources_excluded":True,
        "independent_human_semantic_validation":False,
        "real_multi_turn_dialogue_pairs":0,
        "record_count":sum(len(x) for x in pairs.values()),
        "pairs_per_route":{
            a+"-"+b:len(pairs[key])
            for key in pairs for a,b in (key.split("-"),)}
        | {
            b+"-"+a:len(pairs[key])
            for key in pairs for a,b in (key.split("-"),)
        },
        "final_holdout_downloaded":False,
        "noncommercial_data_used":False,
        "warning":"Tatoeba is useful genuine parallel language data but it has "
                  "no reliable previous-turn context or guaranteed slang. "
                  "Do not claim conversational or 95% semantic success from this data alone."
    }
    return pairs,provenance


def sampled_dev(n:int=36)->tuple[dict,list]:
    """New publisher VALIDATION-only data, disjoint from earlier 12 groups."""
    if not 8<=n<=100:
        raise ValueError("Invalid DEV budget")
    cfg=settings()
    _,old_dev,_=prepare()
    used={item["group_id"] for rows in old_dev.values() for item in rows}
    official=load_groups("validation")
    reviewed=get_reviewed_sources()
    seen=set()
    selected=[]
    for row in sorted(official,key=lambda r:hashlib.sha256(
        ("conv-dev-"+str(cfg["training"]["seed"])+"-"+str(r["id"])).encode()).hexdigest()):
        if row["id"] in used:
            continue
        tokens={normalize(row[lang]) for lang in ("en","es","fr")}
        if tokens & (reviewed | seen):
            continue
        selected.append(row)
        seen|=tokens
        if len(selected)==n:
            break
    if len(selected)!=n:
        raise RuntimeError("Insufficient unused development groups")
    result={}
    for src,tgt in (("es","en"),("fr","en"),("en","es"),
                    ("en","fr"),("es","fr"),("fr","es")):
        result[src+"-"+tgt]=[
            {"group_id":r["id"],"src":r[src],"tgt":r[tgt]}
            for r in selected]
    return result,selected
