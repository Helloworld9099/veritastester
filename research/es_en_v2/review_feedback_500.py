#!/usr/bin/env python3
"""Join qualitative bilingual reviewer flags to *actual* saved model predictions.

This is DEVELOPMENT feedback, not a new independent holdout and not formal
pass/fail annotation. Rows that weren't mentioned are UNKNOWN, never passes.
DO NOT write derived target translations or feed reviewed dev rows to training.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "research" / "es_en_v2"
SPEC = BASE / "reviewer_feedback_20261120" / "issue_groups.json"
ORIGINAL = BASE / "reviewer_examples_500" / "BridgeCaption_500_Real_Model_Translations_Review.csv"
MANIFEST = BASE / "reviewer_examples_500" / "manifest.json"
OUTPUT = BASE / "reviewer_feedback_20261120" / "processed"
COLS = [
    "Excel row", "Case ID", "Direction", "Source type", "Category",
    "Prior context (reviewer only)", "Original spoken text",
    "Actual model translation", "Reported problem categories",
    "Review status", "Formal pass/fail", "Adjudicated?",
]


def load_review(csv_path: Path = ORIGINAL) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"Case ID", "Direction", "Category", "Prior context (reviewer only)",
                    "Original spoken text", "Actual model translation", "Source type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Original generated-model output columns changed")
        rows = []
        seen = set()
        for number, item in enumerate(reader, start=2):
            if number > 501:
                raise ValueError("Too many rows in original reviewer workbook")
            if not item["Case ID"] or item["Case ID"] in seen:
                raise ValueError("Missing or repeated actual model response")
            if not item["Actual model translation"].strip():
                raise ValueError("No bot translation for a row")
            seen.add(item["Case ID"])
            rows.append({"Excel row": number, **item})
    if len(rows) != 500:
        raise ValueError("Original reviewer workbook must contain exactly 500 model outputs")
    return rows


def build(rows: list[dict], spec: dict) -> tuple[list[dict], dict]:
    if len(rows) != 500:
        raise ValueError("Cannot map qualitative annotations without 500 rows")
    if spec.get("status") != "development_review_feedback_not_formal_labels":
        raise ValueError("Unexpected review status")
    if spec.get("final_holdout_eligible") is not False or spec.get("train_on_these_dev_examples") is not False:
        raise ValueError("Feedback may not silently become independent test or training")
    group_map = defaultdict(list)
    for category, numbers in spec["issue_groups"].items():
        if not category or len(numbers) != len(set(numbers)):
            raise ValueError("Missing issue category or duplicate row")
        for number in numbers:
            if type(number) is not int or not 2 <= number <= 501:
                raise ValueError("Invalid Excel row number")
            group_map[number].append(category)

    result = []
    category_direction = defaultdict(Counter)
    by_direction = Counter()
    for row in rows:
        number = row["Excel row"]
        tags = sorted(group_map.get(number, []))
        direction = row["Direction"]
        if direction not in {"es-en", "en-es"}:
            raise ValueError("Unknown model direction")
        by_direction[direction] += 1 if tags else 0
        for tag in tags:
            category_direction[tag][direction] += 1
        entry = {
            "Excel row": number,
            "Case ID": row["Case ID"],
            "Direction": direction,
            "Source type": row["Source type"],
            "Category": row["Category"],
            "Prior context (reviewer only)": row["Prior context (reviewer only)"],
            "Original spoken text": row["Original spoken text"],
            "Actual model translation": row["Actual model translation"],
            "Reported problem categories": ";".join(tags),
            "Review status": "flagged_for_adjudication" if tags else "not_individually_rated",
            "Formal pass/fail": "",
            "Adjudicated?": "No",
        }
        result.append(entry)
    summary = {
        "source_status": "reviewer_reported_dev_issues_not_formal_complete_labels",
        "input_model_responses": len(rows),
        "distinct_rows_flagged_for_adjudication": len(group_map),
        "unflagged_not_assumed_correct": len(rows) - len(group_map),
        "by_direction_distinct_flagged": dict(by_direction),
        "category_flag_counts": {key: len(value) for key, value in spec["issue_groups"].items()},
        "category_direction_counts": {key: dict(value) for key, value in category_direction.items()},
        "requires_independent_adjudication": True,
        "per_row_pass_fail_labels_collected": 0,
        "independently_evaluated_95_percent": False,
        "final_holdout_consumed": False,
        "training_examples_added": 0,
        "note": "Numbers are issue flags, not independent accuracy rates. Categories overlap, "
                "may include subjective translations and source errors. No examples are corrected."
    }
    if len(result) != 500 or len(group_map) != len(set(group_map)):
        raise RuntimeError("Feedback join has missing/duplicate cases")
    return result, summary


def run() -> None:
    manifest = json.loads(MANIFEST.read_text())
    expected = manifest["files_sha256"]["BridgeCaption_500_Real_Model_Translations_Review.csv"]
    observed = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    if observed != expected or manifest["actual_inference_executed"] is not True:
        raise RuntimeError("Original 500 actual model predictions changed")
    spec = json.loads(SPEC.read_text())
    result, summary = build(load_review(), spec)
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite a human-feedback snapshot")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    csv_path = OUTPUT / "reviewer_flagged_500_outputs.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(result)
    summary["source_csv_sha256"] = observed
    summary["reviewer_flagged_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BRIDGECAPTION_REVIEW_FEEDBACK_LINKED", json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
