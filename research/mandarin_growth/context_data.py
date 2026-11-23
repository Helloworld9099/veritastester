"""Context/slang dataset admission gate for shared multilingual training.

Never invent bilingual references, auto-translate monolingual slang into gold,
or count Tatoeba sentences as multi-turn contextual supervision.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path

SUPPORTED=frozenset({"en","es","fr","zh"})
REQUIRED=frozenset({"source_language","target_language","source","target",
                    "preceding_turns","domain","phenomenon","license",
                    "provenance","reviewer_verified"})
PHENOMENA=frozenset({"slang","idiom","coreference","ellipsis","politeness",
                     "code_switching","negation","number","name","context"})


def validate_record(row):
    if not isinstance(row,dict) or not REQUIRED.issubset(row):
        raise ValueError("Incomplete context training provenance")
    src,tgt=row["source_language"],row["target_language"]
    if src not in SUPPORTED or tgt not in SUPPORTED or src==tgt:
        raise ValueError("Unapproved language pair")
    if row["phenomenon"] not in PHENOMENA:
        raise ValueError("Unknown context phenomenon")
    if row["reviewer_verified"] is not True:
        raise ValueError("Only human-verified parallel context gold allowed")
    if not isinstance(row["preceding_turns"],list) or not row["preceding_turns"]:
        raise ValueError("Context requires at least one actual preceding turn")
    if not all(isinstance(t,str) and t.strip() for t in row["preceding_turns"]):
        raise ValueError("Invalid preceding turns")
    if any(not isinstance(row[k],str) or not row[k].strip()
           for k in ("source","target","domain","license","provenance")):
        raise ValueError("Missing aligned text or rights")
    if row["license"].lower() in {"unknown","unverified","cc-by-nc","cc-by-nc-nd"}:
        raise ValueError("License not cleared for commercial use")
    return True


def validate_dataset(path:Path):
    records=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    seen=set()
    for r in records:
        validate_record(r)
        key=hashlib.sha256(json.dumps([r["source_language"],r["target_language"],
            r["source"],r["target"],r["preceding_turns"]],ensure_ascii=False,
            sort_keys=True).encode()).hexdigest()
        if key in seen:
            raise ValueError("Repeated dialogue group")
        seen.add(key)
    return {"verified_records":len(records),"language_pairs":sorted({
        r["source_language"]+"-"+r["target_language"] for r in records
    }),"phenomena":sorted({r["phenomenon"] for r in records}),
    "warning":"Valid provenance alone does not establish train/dev separation or proficiency."}
