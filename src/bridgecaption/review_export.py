"""Export private unlabelled candidates without manufacturing grounded fields."""
import argparse
import csv
import json
import os
from pathlib import Path

HEADERS = {
    "memory": "case_id,group_id,language,utterance,previous,situation,candidate_memory,reference_translation,provenance".split(","),
    "sense": "case_id,group_id,language,expression,utterance,previous,situation,provenance".split(","),
}
METADATA = ["case_id", "split", "needs_human_review", "missing_fields", "rights_status",
            "source_group_id", "source_document_id",
            "source_sha256", "source_filename", "candidate_group_id", "source_label",
            "source_expression_matches_verbatim", "warning"]


def write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_candidates(rows, task, directory):
    if task not in HEADERS:
        raise ValueError("Unknown review task")
    if "private" not in directory.parts and "results" not in directory.parts:
        raise ValueError("Source text must remain under private/ or results/")
    headers = HEADERS[task]
    ready, incomplete, metadata, seen = [], [], [], set()
    for r in rows:
        if r.get("split") not in {"train", "dev"}:
            raise ValueError("Holdout/unknown split cannot enter review queues")
        if not r.get("case_id") or r["case_id"] in seen:
            raise ValueError("Missing or duplicate case ID")
        seen.add(r["case_id"])
        if not r.get("rights_status") or r["rights_status"] in {"unknown", "review-required"}:
            raise ValueError("Review candidates require assessed source rights")
        row = {k: r.get(k, "") or "" for k in headers}
        missing = [k for k, v in row.items() if not isinstance(v, str) or not v.strip()]
        if task == "sense" and len(row["expression"]) > 60:
            missing.append("expression_surface_form_under_61_characters")
        (incomplete if missing else ready).append(row)
        metadata.append({**r, "needs_human_review": True, "missing_fields": ";".join(missing)})
    paths = [directory / f"{task}_queue.csv", directory / f"{task}_needs_review.csv",
             directory / f"{task}_review_metadata.csv"]
    if any(p.exists() for p in paths):
        raise ValueError("Refusing to overwrite existing review work")
    for path, fields, items in zip(paths, [headers, headers, METADATA], [ready, incomplete, metadata]):
        write_csv(path, fields, items)
    return {"task": task, "ready_rows": len(ready), "needs_review_rows": len(incomplete),
            "independently_verified_labels": 0, "paths": [str(p) for p in paths]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--task", choices=HEADERS, required=True)
    p.add_argument("--directory", type=Path, default=Path("data/private"))
    args = p.parse_args()
    try:
        rows = [json.loads(x) for x in args.candidates.read_text().splitlines() if x.strip()]
        print(json.dumps(export_candidates(rows, args.task, args.directory), indent=2))
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
