"""Auditable, conservative bilingual corpus hygiene. Standard library only.

No model-generated bilingual labels are accepted as trustworthy ground truth.
External source licenses must be verified at source, not guessed by this tool.
Splits must be assigned by conversation/speaker/document BEFORE invoking this.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

ALLOWED_LICENSES = frozenset({"CC0-1.0", "CC-BY-2.0", "CC-BY-3.0",
                              "CC-BY-4.0", "CC-BY-SA-4.0", "Apache-2.0",
                              "MIT", "Public-Domain"})
LANGUAGES = frozenset({"es", "en"})
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
URL = re.compile(r"https?://\S+")
DIGITS = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)")
TOKENS = re.compile(r"[^\W_]+", re.UNICODE)
# Do not claim that numeric or character overlap determines semantic equivalence.
LOADED_DATA_WARNING = "Heuristic screening only; every new bilingual reference requires rights and alignment verification."


@dataclass(frozen=True)
class Example:
    case_id: str
    source: str
    reference: str
    source_language: str
    target_language: str
    group_id: str
    speaker_id: str
    document_id: str
    split: str
    dataset: str
    license_id: str
    original_id: str
    rights_verified: bool
    provenance_url: str
    category: str = "general"
    region: str = "unspecified"
    is_synthetic: bool = False
    verified_parallel: bool = False
    previous_source: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, item):
        keys = set(cls.__dataclass_fields__)
        return cls(**{key: tuple(value) if key == "previous_source" else value
                      for key, value in item.items() if key in keys})

    def validate(self):
        for field in ("case_id", "source", "reference", "group_id", "speaker_id",
                      "document_id", "dataset", "original_id", "provenance_url"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError("Missing or invalid " + field)
        if self.source_language not in LANGUAGES or self.target_language not in LANGUAGES:
            raise ValueError("Unknown language")
        if self.source_language == self.target_language:
            raise ValueError("Source and target must differ")
        if self.split not in {"train", "dev", "final_holdout"}:
            raise ValueError("Invalid split")
        if self.license_id not in ALLOWED_LICENSES or self.rights_verified is not True:
            raise ValueError("Dataset license or upstream rights not approved")
        if self.verified_parallel is not True:
            raise ValueError("Bilingual pair not independently aligned/verified")
        if not isinstance(self.previous_source, tuple) or any(
            not isinstance(x, str) for x in self.previous_source
        ):
            raise ValueError("History must be prior source utterance strings")
        if self.split == "final_holdout" and self.is_synthetic:
            raise ValueError("Final holdout cannot contain synthetic examples")
        if any(len(text) > 2000 for text in (self.source, self.reference, *self.previous_source)):
            raise ValueError("Unexpectedly long text")
        if self.source.casefold().strip() == self.reference.casefold().strip():
            raise ValueError("Potential untranslated pair")


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fingerprint(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def ngrams(text: str, n: int = 3) -> set[str]:
    clean = "".join(TOKENS.findall(norm(text)))
    return {clean[i:i+n] for i in range(max(0, len(clean)-n+1))}


def near_duplicate(a: str, b: str, threshold: float = .89) -> bool:
    if norm(a) == norm(b):
        return True
    left, right = ngrams(a), ngrams(b)
    if not left or not right:
        return False
    return len(left & right) / len(left | right) >= threshold


def assess(example: Example) -> list[str]:
    example.validate()
    findings = []
    for name, text in (("source", example.source), ("reference", example.reference)):
        if EMAIL.search(text) or PHONE.search(text):
            findings.append("possible_personal_information_" + name)
        if URL.search(text):
            findings.append("url_requires_review_" + name)
        if "\ufffd" in text or "\x00" in text:
            findings.append("corrupt_" + name)
    a, b = example.source, example.reference
    if not (1 <= len(a.split()) <= 50 and 1 <= len(b.split()) <= 60):
        findings.append("out_of_length_bound")
    if len(a) and len(b) and not .25 <= len(a) / len(b) <= 4.0:
        findings.append("suspicious_alignment_ratio")
    src_nums = DIGITS.findall(a)
    ref_nums = DIGITS.findall(b)
    if src_nums and ref_nums and src_nums != ref_nums:
        findings.append("digit_mismatch_needs_bilingual_review")
    if example.previous_source and example.split == "final_holdout":
        # In production this must be checked at source time against event
        # timestamps. Tuple order alone does not establish causality.
        findings.append("verify_context_is_earlier_by_timestamp")
    return findings


def audit(rows: list[Example]) -> dict:
    """Fail closed on leakage; do not read final-holdout text in training code."""
    if not rows:
        raise ValueError("Empty corpus")
    id_seen, group_splits = set(), {}
    speaker_splits, document_splits = {}, {}
    texts = {"train": (set(), set()), "dev": (set(), set()),
             "final_holdout": (set(), set())}
    counts, flags, datasets = Counter(), Counter(), {}
    for row in rows:
        row.validate()
        if row.case_id in id_seen:
            raise ValueError("Duplicate case_id")
        id_seen.add(row.case_id)
        for mapping, key in ((group_splits, row.group_id),
                             (speaker_splits, row.dataset + "\0" + row.speaker_id),
                             (document_splits, row.dataset + "\0" + row.document_id)):
            old = mapping.setdefault(key, row.split)
            if old != row.split:
                raise ValueError("Cross-split conversation/speaker/document leakage")
        ss, tt = fingerprint(row.source), fingerprint(row.reference)
        src_set, tgt_set = texts[row.split]
        if ss in src_set or tt in tgt_set:
            raise ValueError("Duplicate source or reference inside a split")
        src_set.add(ss)
        tgt_set.add(tt)
        counts[row.split] += 1
        for flag in assess(row):
            flags[flag] += 1
        key = (row.dataset, row.license_id, row.provenance_url)
        datasets[key] = datasets.get(key, 0) + 1
    for one, other in (("train", "dev"), ("train", "final_holdout"), ("dev", "final_holdout")):
        if texts[one][0] & texts[other][0] or texts[one][1] & texts[other][1]:
            raise ValueError("Cross-split exact text collision")
    return {
        "rows": dict(counts),
        "groups": len(group_splits),
        "speakers": len(speaker_splits),
        "documents": len(document_splits),
        "warnings_not_certification": dict(flags),
        "source_license_manifest": [
            {"dataset": key[0], "license_id": key[1], "provenance_url": key[2],
             "rows": n} for key, n in sorted(datasets.items())],
        "limitation": "Exact text/group guards only; near-duplicates and semantic "
                      "paraphrases require a separate approved human review.",
    }


def review_near_duplicates(rows: list[Example], max_pairs: int = 10000):
    """Yield possible collisions for human inspection; does not certify no leakage.

    Quadratic exhaustive comparison may be expensive; refuse large data rather
    than silently checking just a subset and proclaiming leak-free.
    """
    if len(rows) * (len(rows)-1) // 2 > max_pairs:
        raise ValueError("Near-duplicate exhaustive comparison exceeds budget; use indexed retrieval")
    problems = []
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            if rows[i].split == rows[j].split:
                continue
            if (near_duplicate(rows[i].source, rows[j].source) or
                    near_duplicate(rows[i].reference, rows[j].reference)):
                problems.append((rows[i].case_id, rows[j].case_id))
    return problems


def manifest_digest(report: dict) -> str:
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
