"""Development-only machine-translation hard-example mining.

Produces *suspected*, not confirmed, meaning errors. Never opens or trains on
a final holdout. No teacher output, private user audio or new truth labels.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

DIGIT = re.compile(r"(?<!\w)\d+(?:[,.]\d+)?(?!\w)")
NEGATIVE_EN = re.compile(
    r"\b(?:not|no|never|nothing|nobody|neither|without|cannot|can't|won't|don't|doesn't|didn't|isn't|aren't|wasn't|weren't)\b",
    flags=re.IGNORECASE)
NEGATIVE_ES = re.compile(
    r"\b(?:no|nunca|jamás|tampoco|nadie|ninguno|ninguna|sin|nada)\b",
    flags=re.IGNORECASE)
CATEGORY_TERMS = {
    "slang": ("chido","chévere","guay","mola","pana","parce","bacán","brutal","no manches","órale"),
    "idiom": ("tomar el pelo","metí la pata","estar en las nubes","break a leg","pulling my leg"),
    "code_switching": ("anyway","whatever","okay","hello","gracias","please","yeah"),
}
CRITICAL = {"number_disagreement", "negation_disagreement", "entity_disagreement"}
TARGET_MAX_LENGTH = 1500


def normalize(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC",s).casefold().split())


def target_negation(s: str, language: str) -> bool:
    if language == "en":
        return bool(NEGATIVE_EN.search(s))
    if language == "es":
        return bool(NEGATIVE_ES.search(s))
    raise ValueError("Unknown target language")


def issue_tags(source, reference, hypothesis, *, source_language, target_language,
               prior_source=()):
    if not all(isinstance(x,str) for x in (source,reference,hypothesis)):
        raise ValueError("Nontext dev example")
    if len(source)>TARGET_MAX_LENGTH or len(reference)>TARGET_MAX_LENGTH:
        raise ValueError("Unexpectedly long dev example")
    flags=[]
    ref_numbers=DIGIT.findall(reference)
    hyp_numbers=DIGIT.findall(hypothesis)
    if ref_numbers!=hyp_numbers:
        flags.append("number_disagreement")
    if target_negation(reference,target_language)!=target_negation(hypothesis,target_language):
        flags.append("negation_disagreement")
    if not hypothesis.strip():
        flags.append("empty_output")
    elif len(normalize(hypothesis).split())<max(1,len(normalize(reference).split())//3):
        flags.append("possibly_omitted_content")
    context_words=normalize(" ".join(prior_source)+ " "+source)
    for category, terms in CATEGORY_TERMS.items():
        if any(term in context_words for term in terms):
            flags.append("contains_"+category+"_needs_review")
    if prior_source:
        flags.append("context_sensitivity_needs_review")
    # The following are NOT independently established human error labels.
    return sorted(set(flags))


def mine_dev(rows, max_cases=200, *, min_chrf=50.0):
    """Each row: ID, source, reference, hypothesis, split='dev', languages.

    The threshold only prioritizes reviewer attention; a high chrF can still
    be a fatal semantic error. Never import any heldout row to this function.
    """
    if not 1<=max_cases<=10000 or not 0<=min_chrf<=100:
        raise ValueError("Invalid mining budget")
    from sacrebleu.metrics import CHRF
    metric=CHRF()
    candidates=[]
    seen_ids=set()
    tags_counter=Counter()
    for row in rows:
        if row.get("split")!="dev":
            raise ValueError("Only development examples allowed in mining")
        case=row.get("case_id")
        if not case or case in seen_ids:
            raise ValueError("Duplicate or missing dev ID")
        seen_ids.add(case)
        reference=row["reference"]
        translation=row["hypothesis"]
        score=metric.sentence_score(translation,[reference]).score
        tags=issue_tags(row["source"],reference,translation,
                        source_language=row["source_language"],
                        target_language=row["target_language"],
                        prior_source=tuple(row.get("previous_source",[])))
        for tag in tags:
            tags_counter[tag]+=1
        if score<min_chrf or tags:
            severity=sum(4 if t in CRITICAL else 1 for t in tags)
            candidates.append({
                "case_id":case, "chrF_not_meaning_accuracy":round(score,3),
                "suspected_error_tags":tags,
                "risk_priority":round(severity*2+(100-score)/25,3),
                "original_license":row.get("license_id"),
                "provenance":row.get("provenance_url"),
                "needs_bilingual_review":True,
            })
    return {
        "input_dev_n":len(seen_ids),
        "flagged_n":len(candidates),
        "returned_n":min(len(candidates),max_cases),
        "tag_counts":dict(tags_counter),
        "ranked_cases":sorted(candidates,
                              key=lambda x:(-x["risk_priority"],x["case_id"]))[:max_cases],
        "human_confirmed_translation_errors":0,
        "no_final_holdout_accessed":True,
        "warning":"Heuristic flags are not verified semantic errors. New training "
                  "labels require permitted genuine bilingual references and human validation.",
    }
