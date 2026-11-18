"""Generate blinded translation-review assignments and aggregate real human scores.

Private source and model outputs are never committed by this tool.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from score_human import CATEGORIES, score


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode()).hexdigest()


def jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def prepare(rows, checkpoint, folder):
    if folder.exists():
        raise FileExistsError("Refusing to overwrite previous reviewer packet")
    if len(checkpoint) != 64 or any(c not in "0123456789abcdef" for c in checkpoint):
        raise ValueError("Checkpoint must be SHA-256")
    seen, cases, review = set(), [], []
    language = None
    for row in rows:
        sid = row.get("sample_id")
        lang = row.get("language")
        group = row.get("source_group_sha256")
        category = row.get("category")
        source = row.get("source_text")
        translation = row.get("candidate_translation")
        if (not isinstance(sid, str) or not sid or sid in seen
                or lang not in ("french", "spanish") or category not in CATEGORIES
                or not isinstance(group, str) or len(group) != 64
                or any(c not in "0123456789abcdef" for c in group)
                or not isinstance(source, str) or not source.strip()
                or not isinstance(translation, str) or not translation.strip()):
            raise ValueError("Invalid sample, language, translation or source grouping")
        if language is not None and language != lang:
            raise ValueError("Different languages must have separate rating packets")
        language = lang
        seen.add(sid)
        token = digest([checkpoint, sid, lang, group, category, source, translation])
        cases.append({"sample_id": sid, "category": category,
                      "source_group_sha256": group, "case_hash": token})
        review.append({"sample_id": sid, "case_hash": token,
                       "source_text": source, "candidate_translation": translation,
                       "category": category, "adequate": None})
    if not cases:
        raise ValueError("No cases")
    manifest = {"language": language, "checkpoint_sha256": checkpoint,
                "cases": sorted(cases, key=lambda x: x["sample_id"])}
    manifest["evaluation_manifest_sha256"] = digest(manifest)
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    import random
    for label in ("a", "b"):
        ordered = list(review)
        random.Random(digest(manifest["evaluation_manifest_sha256"] + label)).shuffle(ordered)
        (folder / ("rater_" + label + ".jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))
    return manifest["evaluation_manifest_sha256"]


def assemble(manifest, a, b, arbitration=None):
    frozen = json.loads(manifest.read_text())
    manifest_sha = frozen["evaluation_manifest_sha256"]
    if digest({k: v for k, v in frozen.items() if k != "evaluation_manifest_sha256"}) != manifest_sha:
        raise ValueError("Frozen manifest changed")
    if a.resolve() == b.resolve():
        raise ValueError("Separate reviewer submissions required")
    cases = {row["sample_id"]: row for row in frozen["cases"]}
    def ratings(rows):
        seen = {}
        for row in rows:
            key = row.get("sample_id")
            if key not in cases or key in seen or row.get("case_hash") != cases[key]["case_hash"]:
                raise ValueError("Unknown, duplicated or altered evaluation case")
            if type(row.get("adequate")) is not bool:
                raise ValueError("Every human judgment must be a boolean")
            seen[key] = row["adequate"]
        if set(seen) != set(cases):
            raise ValueError("Missing bilingual reviewer judgments")
        return seen
    left, right = ratings(jsonl(a)), ratings(jsonl(b))
    disagreed = {key for key in left if left[key] != right[key]}
    judged = {}
    if arbitration is not None:
        for row in jsonl(arbitration):
            key = row.get("sample_id")
            if (key not in disagreed or key in judged or
                    row.get("case_hash") != cases[key]["case_hash"] or
                    type(row.get("adequate")) is not bool):
                raise ValueError("Untrusted or duplicate third-rater judgment")
            judged[key] = row["adequate"]
    if set(judged) != disagreed:
        raise ValueError("An independent third reviewer must adjudicate every disagreement")
    evaluated = []
    for key, case in cases.items():
        entry = {"sample_id": key, "language": frozen["language"],
                 "split": "independent_test",
                 "checkpoint_sha256": frozen["checkpoint_sha256"],
                 "evaluation_manifest_sha256": manifest_sha,
                 "source_group_sha256": case["source_group_sha256"],
                 "category": case["category"],
                 "rater_a_adequate": left[key], "rater_b_adequate": right[key]}
        if key in disagreed:
            entry["adjudicated_adequate"] = judged[key]
        evaluated.append(entry)
    return score(evaluated)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("predictions", type=Path)
    prep.add_argument("--checkpoint-sha256", required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    check = sub.add_parser("score")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--rater-a", type=Path, required=True)
    check.add_argument("--rater-b", type=Path, required=True)
    check.add_argument("--adjudication", type=Path)
    check.add_argument("--output", type=Path, required=True)
    opts = p.parse_args()
    if opts.action == "prepare":
        print("Frozen evaluation manifest:", prepare(jsonl(opts.predictions),
              opts.checkpoint_sha256, opts.output_dir))
    else:
        report = assemble(opts.manifest, opts.rater_a, opts.rater_b, opts.adjudication)
        if opts.output.exists():
            p.error("Refusing to overwrite historical evaluation report")
        opts.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
