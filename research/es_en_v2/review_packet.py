#!/usr/bin/env python3
"""Prepare private, blinded, tamper-evident bilingual-review materials.

No model calls, final-holdout translations, score judgments or network access.
Human reviewers handle actual sealed final data outside this PUBLIC repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
from pathlib import Path

CATEGORIES = ("general", "context", "idiom", "regional_slang",
              "domain_terms", "noisy_transcript")
DIRECTIONS = ("es-en", "en-es")
FIELDS = ("case_id", "direction", "category", "group_id", "speaker_id",
          "document_id", "region", "speaker_history", "source", "model_code",
          "model_translation", "adequate", "critical_error", "error_types",
          "notes", "reviewer_code")
CRITICAL_TYPES = ("negation", "wrong_number", "wrong_name",
                  "fabricated_claim", "wrong_speaker")
ERROR_TYPES = CRITICAL_TYPES + ("omission", "slang_sense", "idiom_sense",
                                "pragmatics", "register", "modality",
                                "regional", "agreement", "incoherent", "other")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_file(path: Path, callback) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        callback(f)


def write_private_csv(path: Path, rows):
    def callback(f):
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    private_file(path, callback)


def seal(path: Path, out: Path) -> dict:
    """No file contents enter the public log; this is a cryptographic receipt."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("Holdout file must be a real local file")
    if "data/private" not in str(path) and "results/private" not in str(path):
        raise ValueError("The sealed holdout must stay under data/private or results/private")
    digest = sha256(path)
    record = {
        "version": 1,
        "sealed_file_name": path.name,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "created_by": "bridgecaption-v2-seal",
        "holdout_text_not_uploaded_to_public_repo": True,
        "trained_model_access_count": 0,
        "human_evaluation_status": "not_started",
        "warning": "A file hash does not prove independence, balanced sampling, "
                   "informed consent or reviewer fluency.",
    }
    private_file(out, lambda f: f.write(json.dumps(record, indent=2)))
    return record


def create_blank_review_packet(cases, systems, out_dir: Path, *, seed: int = 20261120):
    """Blinding requires system outputs already frozen by an independent operator.

    This function only scrambles their labels; intentionally refuses to
    translate, inspect final errors or calculate adequacy.
    """
    import random
    if len(systems) < 2 or len(set(systems)) != len(systems):
        raise ValueError("Compare at least two distinct frozen systems")
    if not cases:
        raise ValueError("Empty reviewer package")
    counts = {direction: 0 for direction in DIRECTIONS}
    ids = set()
    grouped = {}
    for case in cases:
        if case.get("direction") not in DIRECTIONS:
            raise ValueError("Unknown direction")
        if case.get("category") not in CATEGORIES:
            raise ValueError("Unknown category")
        cid = case.get("case_id")
        if not cid or cid in ids:
            raise ValueError("Duplicate or missing case ID")
        ids.add(cid)
        counts[case["direction"]] += 1
        if not case.get("source") or not case.get("group_id"):
            raise ValueError("Missing source or conversation group")
        outputs = case.get("frozen_system_outputs", {})
        if set(outputs) != set(systems) or not all(
            isinstance(x, str) and x.strip() for x in outputs.values()
        ):
            raise ValueError("All frozen system outputs required; no model calls here")
        grouped[cid] = outputs
    if min(counts.values()) < 1:
        raise ValueError("Both translation directions required")
    # Mapping and test examples are private; seed is reproducibility metadata,
    # not a pseudonymization key. A fresh per-case random permutation is used.
    rows, secret_map = [], []
    rng = random.SystemRandom()
    for case in cases:
        cid = case["case_id"]
        shuffled = list(systems)
        rng.shuffle(shuffled)
        for n, system in enumerate(shuffled):
            code = f"system_{n+1:02d}"
            packet = {
                "case_id": cid, "direction": case["direction"],
                "category": case["category"], "group_id": case["group_id"],
                "speaker_id": case.get("speaker_id", ""),
                "document_id": case.get("document_id", ""),
                "region": case.get("region", ""),
                "speaker_history": json.dumps(case.get("previous_source", []),
                                               ensure_ascii=False),
                "source": case["source"],
                "model_code": code,
                "model_translation": grouped[cid][system],
                "adequate": "", "critical_error": "", "error_types": "",
                "notes": "", "reviewer_code": "",
            }
            rows.append(packet)
            secret_map.append({"case_id": cid, "code": code, "system": system})
    out_dir.mkdir(parents=True, exist_ok=False)
    write_private_csv(out_dir / "blind_rater_1.csv", rows)
    write_private_csv(out_dir / "blind_rater_2.csv", rows)
    private_file(out_dir / "system_key_private.csv", lambda f: (
        csv.DictWriter(f, fieldnames=["case_id", "code", "system"]).writeheader(),
        csv.DictWriter(f, fieldnames=["case_id", "code", "system"]).writerows(secret_map)
    ))
    report = {"cases": len(cases), "direction_case_counts": counts,
              "compared_systems": len(systems), "rows_per_reviewer": len(rows),
              "reviewer_files_hashes": {
                  name: sha256(out_dir / name) for name in
                  ("blind_rater_1.csv", "blind_rater_2.csv", "system_key_private.csv")},
              "human_labels_written": 0,
              "human_review_required": True,
              "note": "Reviewer packets are identical but reviewers must annotate independently. "
                      "Third bilingual adjudicator must reconcile disagreements outside this tool. "
                      "Never publish final source text, model key or final labels until evaluation protocol permits."}
    private_file(out_dir / "packet_manifest_private.json",
                 lambda f: f.write(json.dumps(report, indent=2)))
    return report


def validate_review(row):
    if row["adequate"] not in {"pass", "fail", "unrateable"}:
        raise ValueError("Invalid human adequacy label")
    if row["critical_error"] not in {"yes", "no"}:
        raise ValueError("Explicit human critical error label required")
    for item in [x.strip() for x in row["error_types"].split(";") if x.strip()]:
        if item not in ERROR_TYPES:
            raise ValueError("Unknown error type")
    if row["critical_error"] == "yes" and row["adequate"] == "pass":
        raise ValueError("Critical error cannot pass meaning preservation")


def wilson_interval(passes: int, total: int, z: float = 1.95996398454):
    """For completed *human* labels only; never interpret chrF as accuracy."""
    if total <= 0 or not 0 <= passes <= total:
        raise ValueError("Invalid human label counts")
    n = float(total)
    p = passes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*((p*(1-p)/n + z*z/(4*n*n))**.5)/denominator
    return max(0.,center-half), min(1.,center+half)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sealing = sub.add_parser("seal", help="SHA256 a locally stored private holdout")
    sealing.add_argument("--private-holdout", type=Path, required=True)
    sealing.add_argument("--private-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "seal":
        print(json.dumps(seal(args.private_holdout, args.private_manifest), indent=2))


if __name__ == "__main__":
    main()
