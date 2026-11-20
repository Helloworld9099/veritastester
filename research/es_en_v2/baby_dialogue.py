"""Context-rich bilingual dialogue candidate parser, with conservative quarantine.

Source: https://huggingface.co/datasets/lindazeng979/bilingual-babyLM
Research paper: https://arxiv.org/abs/2603.29552 (Zeng, Feng, Frank 2026)
HF card/license: MIT; generated synthetic dialogue using proprietary GPT-4
upstream. The data is NOT independently human-verified direct translation.
No inference calls or data downloads occur in this module.

Never train on a returned candidate until explicit source/target alignment
checks and a licensed bilingual quality review certify the pairs. Keep
speaker/dialgoue identity and only preceding source-language turns.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import Counter

MARKER = re.compile(r"\*\*([A-Za-z][A-Za-z -]{1,30})\*\*\s*:\s*")
LANGUAGE_ALLOWED = {"Dad", "Mom", "Child", "Father", "Mother", "Baby"}
SPLIT_SEED = "bridgecaption-v2-babylm-split-v1"
MIN_RATIO, MAX_RATIO = .4, 2.5


def parse_turns(text: str) -> list[tuple[str, str]]:
    if not isinstance(text, str) or len(text) > 9000:
        return []
    cleaned = text.replace("<|endoftext|>", " ").strip()
    points = list(MARKER.finditer(cleaned))
    if not points:
        return []
    if cleaned[:points[0].start()].strip():
        return []
    turns=[]
    for i, m in enumerate(points):
        end = points[i+1].start() if i+1<len(points) else len(cleaned)
        utterance = " ".join(cleaned[m.end():end].split())
        speaker = m.group(1).strip()
        if speaker not in LANGUAGE_ALLOWED or not 2<=len(utterance.split())<=45:
            return []
        if len(utterance)>300:
            return []
        turns.append((speaker,utterance))
    return turns


def split_for_group(group: str) -> str:
    if not isinstance(group,str) or not group:
        raise ValueError("Missing dialogue group")
    bucket=int(hashlib.sha256((SPLIT_SEED+group).encode()).hexdigest()[:10],16)%100
    return "dev" if bucket>=90 else "train"


def parse_record(row: dict, *, direction: str = "es-en"):
    if direction not in {"es-en","en-es"}:
        raise ValueError("Unsupported direction")
    expected = {"custom_id","content_en","content_es","parent","context"}
    if not expected.issubset(row):
        raise ValueError("Missing original bilingual synthetic dialogue fields")
    group=str(row["custom_id"])
    en,es=parse_turns(row["content_en"]),parse_turns(row["content_es"])
    if len(en)!=len(es) or len(es)<2:
        return {"status":"quarantined", "reason":"turn_count_or_marker_mismatch", "group":group}
    src_lang = direction[:2]
    source = es if src_lang=="es" else en
    target = en if src_lang=="es" else es
    split = split_for_group(group)
    outputs=[]
    for i,((sp,s),(tp,t)) in enumerate(zip(source,target)):
        if sp!=tp:
            return {"status":"quarantined", "reason":"speaker_alignment", "group":group}
        if not MIN_RATIO<=len(s)/len(t)<=MAX_RATIO:
            return {"status":"quarantined", "reason":"suspect_length_ratio", "group":group}
        if s.casefold()==t.casefold():
            return {"status":"quarantined", "reason":"untranslated_surface", "group":group}
        prev=[{"speaker":p,"source":text} for p,text in source[max(0,i-3):i]]
        outputs.append({
            "case_id":hashlib.sha256((group+"|"+direction+"|"+str(i)).encode()).hexdigest()[:20],
            "original_group_id":group, "direction":direction,
            "split":split,"speaker":sp,
            "source":s,"reference_candidate":t,
            "previous_source_turns":prev,
            "conversation_scene":str(row["context"])[:100],
            "source_speaker_type":str(row["parent"])[:50],
            "authentic_human_dialogue":False,
            "synthetic_upstream":True,
            "source_license":"MIT",
            "upstream_generation":"GPT-4 synthetic by original authors",
            "semantics_independently_verified":False,
            "permitted_as_human_gold":False,
            "eligible_for_training":False,
            "requires_bilingual_review":True,
        })
    return {"status":"candidate_for_review","group":group,"split":split,
            "rows":outputs,"reviewed":0}


def summarize_records(raw_records, max_groups=10000):
    summary=Counter()
    sample=[]
    seen=set()
    for item in raw_records:
        if len(seen)>=max_groups:
            break
        parsed=parse_record(item)
        gid=parsed["group"]
        if gid in seen:
            raise ValueError("Duplicate source group")
        seen.add(gid)
        summary[parsed["status"]]+=1
        if parsed["status"]=="candidate_for_review":
            summary["candidate_turns"]+=len(parsed["rows"])
            summary[parsed["split"]+"_groups"]+=1
            if len(sample)<12:
                sample.append({"group_hash":hashlib.sha256(gid.encode()).hexdigest(),
                               "turns":len(parsed["rows"]),"split":parsed["split"]})
        else:
            summary[parsed["reason"]]+=1
    return {
        "source": "lindazeng979/bilingual-babyLM",
        "dataset_license_claim":"MIT",
        "upstream_content_origin":"synthetic GPT-4",
        "inspected_group_count":len(seen),
        "counts":dict(summary),
        "representative_group_hashes_not_text":sample,
        "eligible_for_training_without_bilingual_quality_review":0,
        "human_gold_examples":0,
        "allowed_final_holdout_examples":0,
        "no_final_holdout_accessed":True,
    }
