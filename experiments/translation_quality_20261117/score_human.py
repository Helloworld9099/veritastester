#!/usr/bin/env python3
"""Independent bilingual human adequacy scoring for BridgeCaption's 90% target.

This evaluates human annotations only, never BLEU/chrF or training-set accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

CATEGORIES = ("everyday", "idiom", "slang", "context", "technical")
MIN_SAMPLES = 200
MIN_PER_CATEGORY = 20
# An overall 90% can conceal near-total failure on slang or context cases.
# The category-level requirement is an observed rate, NOT a 95%-confidence claim.
MIN_CATEGORY_OBSERVED_ADEQUACY = 0.90
MIN_GROUPS = 20
Z = 1.959963984540054


def wilson_lower(successes: int, n: int) -> float:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Invalid adequate-translation count")
    p = successes / n
    denom = 1.0 + Z * Z / n
    center = p + Z * Z / (2.0 * n)
    error = Z * math.sqrt(p * (1.0 - p) / n + Z * Z / (4.0 * n * n))
    return max(0.0, (center - error) / denom)


def score(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No independently annotated cases")
    seen, labels, category_adequate, groups = set(), Counter(), Counter(), set()
    successful, disagreements, manifest, checkpoint, language = 0, 0, None, None, None
    for number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Case {number}: expected JSON object")
        lang = row.get("language")
        check = row.get("checkpoint_sha256")
        man = row.get("evaluation_manifest_sha256")
        case = row.get("sample_id")
        group = row.get("source_group_sha256")
        kind = row.get("category")
        if lang not in ("french", "spanish") or not isinstance(case, str) or not case:
            raise ValueError(f"Case {number}: invalid language or identifier")
        if row.get("split") != "independent_test":
            raise ValueError(f"Case {number}: ratings must come from independent_test")
        if kind not in CATEGORIES:
            raise ValueError(f"Case {number}: missing category coverage")
        if not all(isinstance(v, str) and len(v) == 64
                   and all(c in "0123456789abcdef" for c in v)
                   for v in (check, man, group)):
            raise ValueError(f"Case {number}: missing SHA-256 provenance")
        if language is None:
            language, manifest, checkpoint = lang, man, check
        if (lang, man, check) != (language, manifest, checkpoint):
            raise ValueError("Do not pool languages, frozen samples, or checkpoint versions")
        if case in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(case)
        groups.add(group)
        labels[kind] += 1
        a, b = row.get("rater_a_adequate"), row.get("rater_b_adequate")
        if type(a) is not bool or type(b) is not bool:
            raise ValueError(f"Case {number}: requires two independent boolean ratings")
        if a != b:
            disagreements += 1
            accepted = row.get("adjudicated_adequate")
            if type(accepted) is not bool:
                raise ValueError(f"Case {number}: disagreement requires independent adjudication")
        else:
            accepted = a
            if row.get("adjudicated_adequate") not in (None, accepted):
                raise ValueError(f"Case {number}: contradictory adjudication")
        successful += int(accepted)
        category_adequate[kind] += int(accepted)

    n = len(rows)
    lower = wilson_lower(successful, n)
    cover = all(labels[kind] >= MIN_PER_CATEGORY for kind in CATEGORIES)
    per_category = {
        kind: {
            "n": labels[kind],
            "adequate": category_adequate[kind],
            "observed_fraction": (
                category_adequate[kind] / labels[kind] if labels[kind] else None),
        }
        for kind in CATEGORIES
    }
    # Balanced pragmatic coverage: do not hide poor idiom/slang/context results
    # behind a larger number of straightforward literal translations.
    category_ok = all(
        labels[kind] >= MIN_PER_CATEGORY and
        category_adequate[kind] / labels[kind] >= MIN_CATEGORY_OBSERVED_ADEQUACY
        for kind in CATEGORIES)
    size_ok = n >= MIN_SAMPLES
    groups_ok = len(groups) >= MIN_GROUPS
    report = {
        "language": language, "checkpoint_sha256": checkpoint,
        "evaluation_manifest_sha256": manifest,
        "n": n, "adequate": successful, "observed_fraction": successful / n,
        "wilson_lower_95": lower, "disagreements_adjudicated": disagreements,
        "by_category_count": dict(labels), "by_category_quality": per_category,
        "group_count": len(groups),
        "meets_90_percent_human_adequacy_gate": (
            size_ok and cover and groups_ok and category_ok and lower >= 0.90),
        "eligibility": {
            "at_least_200_cases": size_ok,
            "at_least_20_of_each_category": cover,
            "at_least_90_percent_observed_in_each_category": category_ok,
            "at_least_20_distinct_source_groups": groups_ok,
        },
        "warnings": (
            "Assumes genuinely independent human bilingual ratings and a locked "
            "heldout sample. A script cannot independently verify reviewer identity, "
            "pretraining contamination, or undisclosed sample selection. "
            "Overall 95% Wilson lower bound and >=90% observed in every category "
            "(especially idiom, slang and context) are separate gates. Category rates "
            "have their own uncertainty. This cannot prove full production, ASR, "
            "spoken-dialogue or context-aware accuracy."
        ),
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("annotations", type=Path, help="Private JSONL; never commit source or outputs")
    p.add_argument("--output", type=Path, help="Save aggregate JSON only")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.annotations.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    result = score(rows)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        if args.output.exists():
            p.error("Refusing to overwrite earlier evaluation report")
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
