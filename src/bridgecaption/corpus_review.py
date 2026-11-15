"""Turn local unlabelled corpus candidates into reviewed BridgeCaption training rows.

Never infers meaning, relevance or situational facts automatically. Prompts a
human for independent references and a second grounded view. Test/holdout
candidates are rejected. Output is private, owner-readable only on POSIX.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .research_training import MemoryExample
from .sense_model import SenseExample


def safe_display(text: str) -> str:
    """Strip control characters so downloaded corpus text cannot send terminal escapes."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", str(text))


def source_rows(path: Path, task: str):
    rows = []
    ids = set()
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("split") not in {"train", "dev"}:
                raise ValueError(f"Test/holdout or invalid split found on line {line_no}")
            if task == "memory":
                required = {"case_id","group_id","language","utterance","previous",
                            "candidate_memory","provenance"}
            else:
                required = {"case_id","group_id","language","utterance",
                            "expression","provenance"}
            if not required.issubset(item) or any(not isinstance(item[key], str) or
                    not item[key].strip() for key in required):
                raise ValueError(f"Missing required grounded fields at line {line_no}")
            if item["case_id"] in ids:
                raise ValueError("Duplicate case identifier in review candidates")
            ids.add(item["case_id"])
            rows.append(item)
    if not rows:
        raise ValueError("No review candidates")
    return rows


def seen_ids(path: Path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as stream:
        return {json.loads(line)["case_id"] for line in stream if line.strip()}


def review(candidates: Path, out: Path, task: str, *, append=False,
           input_fn=input, output_fn=print):
    if task not in {"memory", "sense"}:
        raise ValueError("Only memory/sense human review is supported")
    rows = source_rows(candidates, task)
    existing = seen_ids(out) if append else set()
    if "private" not in out.parts and "results" not in out.parts:
        raise ValueError("Reviewed transcripts must remain in private/ or results/")
    out.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_EXCL)
    fd = os.open(out, flags, 0o600)
    completed, skipped = 0, 0
    with os.fdopen(fd, "a" if append else "w", encoding="utf-8") as stream:
        for candidate in rows:
            if candidate["case_id"] in existing:
                skipped += 1
                continue
            output_fn(f"\nID {candidate['case_id']} | group {candidate['group_id']}")
            output_fn(f"Language: {candidate['language']}")
            output_fn(f"Original: {safe_display(candidate['utterance'])}")
            previous_text = candidate.get("previous", "").strip()
            if not previous_text and task == "sense":
                previous_text = "[NO_PREVIOUS_TURN_IN_SOURCE]"
            output_fn(f"Previous: {safe_display(previous_text)}")
            if task == "memory":
                output_fn(f"Candidate memory: {safe_display(candidate['candidate_memory'])}")
                output_fn("NOTE: Same-dialogue is not necessarily helpful; other-dialogue is not necessarily harmful.")
            else:
                output_fn(f"Expression: {safe_display(candidate['expression'])}")
                output_fn("NOTE: Existing idiomaticity labels or dictionary definitions are NOT automatically the sense.")
            output_fn("Independent review required. 'q' at any prompt stops without saving that row.")

            def ask(prompt, *, options=None, limit=1200):
                while True:
                    reply = input_fn(prompt).strip()
                    if reply.casefold() == "q":
                        return None
                    if not reply or len(reply) > limit:
                        output_fn(f"Enter a meaningful answer with 1–{limit} characters")
                        continue
                    if options is not None and reply not in options:
                        output_fn("Choose one of: " + ", ".join(sorted(options)))
                        continue
                    return reply

            situation = ask("Grounded scene/situation (not inferred from target translation)> ")
            if situation is None:
                break
            if task == "memory":
                reference = ask("Independent English reference translation> ")
                if reference is None:
                    break
                label = ask("Is this memory helpful, neutral, or harmful? ",
                            options={"helpful","neutral","harmful"}, limit=16)
                if label is None:
                    break
                record = MemoryExample.from_dict({
                    **{k:candidate[k] for k in
                       ("case_id","group_id","language","utterance","previous",
                        "candidate_memory","provenance")},
                    "situation": situation,
                    "reference_translation": reference,
                    "human_judgement": label})
            else:
                meaning = ask("Independently verified meaning, not merely idiomatic/literal> ",
                              limit=120)
                if meaning is None:
                    break
                record = SenseExample.from_dict({
                    **{k:candidate[k] for k in
                       ("case_id","group_id","language","utterance",
                        "expression","provenance")},
                    "previous": previous_text,
                    "situation": situation, "meaning": meaning})
            stream.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
            stream.flush()
            existing.add(record.case_id)
            completed += 1
    return {"labelled": completed, "already_reviewed": skipped,
            "remaining": len(rows) - completed - skipped,
            "training_started": False}


def main():
    parser = argparse.ArgumentParser(description="Human review of unlabelled corpus candidates")
    parser.add_argument("--task", choices=("memory","sense"), required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--append", action="store_true", help="Continue without duplicating cases")
    args = parser.parse_args()
    try:
        report = review(args.candidates, args.out, args.task, append=args.append)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
