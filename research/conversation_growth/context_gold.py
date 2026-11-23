"""Human-approved conversation translation examples, grouped by entire dialog.

Only independent bilingual verified source/target translations can enter the
context-conditioned training pool. Corpus examples without target or bilingual
approval (including OASST1) are candidates, NEVER gold. No future utterance
may be fed as context in a live speech translation application.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED=frozenset(("en","es","fr"))
ALLOWED_LICENSE=frozenset(("CC0-1.0","CC-BY-4.0","CC-BY-2.0-FR","Apache-2.0","MIT"))
PII_RE=re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b|https?://\S+",re.I)


@dataclass(frozen=True)
class VerifiedContextExample:
    conversation_id:str
    turn_id:str
    source_language:str
    target_language:str
    current_text:str
    reference_translation:str
    previous_utterances:tuple[str,...]
    review_status:str
    dataset_license:str
    dataset_source:str

    def __post_init__(self):
        if (self.source_language not in SUPPORTED or self.target_language not in SUPPORTED
            or self.source_language==self.target_language):
            raise ValueError("Invalid English/French/Spanish direction")
        if self.review_status!="independent_bilingual_verified":
            raise ValueError("Unverified auto-generated references cannot be used for context training")
        if self.dataset_license not in ALLOWED_LICENSE:
            raise ValueError("Noncommercial or unclear training ancestry blocked")
        if not self.dataset_source or not self.conversation_id or not self.turn_id:
            raise ValueError("Every sentence needs a documented source and conversation identity")
        if not self.current_text.strip() or not self.reference_translation.strip():
            raise ValueError("A bilingual reference and source current speech must exist")
        if len(self.previous_utterances)>4:
            raise ValueError("Real-time context is limited to four past turns")
        if any(len(x)>400 for x in (self.current_text,self.reference_translation,*self.previous_utterances)):
            raise ValueError("Long utterance rejected; not silently truncated")
        if any(PII_RE.search(x) for x in
               (self.current_text,self.reference_translation,*self.previous_utterances)):
            raise ValueError("Sensitive-looking text needs a human privacy review first")


def history_only_source(example:VerifiedContextExample)->str:
    """Past-only bounded context; never include reference, target, future turn."""
    # Plain source utterance ALWAYS comes last. This format MUST be learned
    # from verified pairs before a context-aware live model can be activated.
    def escape(text):
        return text.replace("<","‹").replace(">","›").replace("\n"," ")
    prev=" ".join("<past> "+escape(u) for u in example.previous_utterances)
    return (prev+" <current> "+escape(example.current_text)).strip()


def load_verified_jsonl(path:Path)->list[VerifiedContextExample]:
    result=[]
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row=json.loads(line)
            row["previous_utterances"]=tuple(row.get("previous_utterances",[]))
            result.append(VerifiedContextExample(**row))
    if len({(x.conversation_id,x.turn_id) for x in result})!=len(result):
        raise ValueError("Same dialogue-turn used twice")
    return result


def group_split(examples:list[VerifiedContextExample],seed:int,
                *,development_fraction:float=.2)->tuple[list, list,dict]:
    if not .05<=development_fraction<=.35:
        raise ValueError("Bad conversation split")
    groups={}
    for item in examples:
        groups.setdefault(item.conversation_id,[]).append(item)
    if len(groups)<5:
        raise ValueError("Too few independent conversations for a reliable split")
    ids=sorted(groups,key=lambda x:hashlib.sha256(
        (str(seed)+"|"+x).encode()).hexdigest())
    dev_n=max(1,round(len(ids)*development_fraction))
    dev_ids=set(ids[:dev_n])
    train=[item for item in examples if item.conversation_id not in dev_ids]
    dev=[item for item in examples if item.conversation_id in dev_ids]
    if {x.conversation_id for x in train}&{x.conversation_id for x in dev}:
        raise RuntimeError("Cross-turn leakage after splitting by conversation")
    return train,dev,{
        "dataset_license_status":"independently bilingual verified only",
        "train_conversation_count":len(set(x.conversation_id for x in train)),
        "development_conversation_count":len(dev_ids),
        "future_context_leakage":False,
        "automatic_teacher_outputs_used_as_gold":False,
        "final_holdout_used":False,
        "all_examples_from_bilingual_review":True,
    }
