"""Local-only human annotation of memory and slang sense training examples.

No LLM generates ground-truth labels. Never saves audio or uploads meeting text.
Only use public/licensed sources or conversations with informed permission.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from .research_training import MemoryExample
from .sense_model import SenseExample

TASKS = {
    "memory": (MemoryExample, "human_judgement"),
    "sense": (SenseExample, "meaning"),
}


def read_queue(path: Path, task: str) -> list[dict[str, str]]:
    if task not in TASKS:
        raise ValueError("Unknown annotation task")
    model, label = TASKS[task]
    fields = set(model.__dataclass_fields__) - {label}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not fields.issubset(reader.fieldnames):
            raise ValueError("Queue must contain: " + ", ".join(sorted(fields)))
        rows = []
        ids = set()
        for index, row in enumerate(reader, start=2):
            if not all(isinstance(row.get(key), str) and row[key].strip() for key in fields):
                raise ValueError(f"Missing data in row {index}")
            if row["case_id"] in ids:
                raise ValueError("Duplicate case_id in queue")
            ids.add(row["case_id"])
            rows.append({key: row[key].strip() for key in fields})
    if not rows:
        raise ValueError("Queue has no examples")
    return rows


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                obj = json.loads(line)
                result.add(obj["case_id"])
    return result


def annotate(queue: Path, out: Path, task: str, *, append: bool = False,
             input_fn=input, output_fn=print) -> dict[str, int]:
    """Prompt a human for labels. 'q' stops without inventing labels."""
    if task not in TASKS:
        raise ValueError("Unknown annotation task")
    model, label_name = TASKS[task]
    rows = read_queue(queue, task)
    done = existing_ids(out) if append else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_EXCL)
    fd = os.open(out, flags, 0o600)
    created = 0
    skipped = 0
    try:
        with os.fdopen(fd, "a" if append else "w", encoding="utf-8") as stream:
            for row in rows:
                if row["case_id"] in done:
                    skipped += 1
                    continue
                output_fn(f"\nCase: {row['case_id']} | {row['language']} | {row['group_id']}")
                output_fn(f"Utterance: {row['utterance']}")
                output_fn(f"Previous: {row['previous']}")
                output_fn(f"Situation view: {row['situation']}")
                if task == "memory":
                    output_fn(f"Candidate: {row['candidate_memory']}")
                    output_fn(f"Reference: {row['reference_translation']}")
                    output_fn("Label as helpful, neutral or harmful; 'q' to stop.")
                else:
                    output_fn(f"Expression: {row['expression']}")
                    output_fn("Enter a short meaning (independent human judgement); 'q' to stop.")
                while True:
                    answer = input_fn(f"{label_name}> ").strip()
                    if answer.casefold() == "q":
                        return {"labelled": created, "skipped": skipped,
                                "remaining": len(rows) - created - skipped}
                    if task == "memory" and answer not in {"helpful", "neutral", "harmful"}:
                        output_fn("Choose helpful, neutral or harmful.")
                        continue
                    if task == "sense" and (not answer or len(answer) > 120):
                        output_fn("Use a nonempty meaning of at most 120 characters.")
                        continue
                    break
                obj = {**row, label_name: answer}
                record = model.from_dict(obj)
                stream.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
                stream.flush()
                done.add(row["case_id"])
                created += 1
    finally:
        # os.fdopen owns fd after it opens, but close it if wrapping failed.
        try:
            os.close(fd)
        except OSError:
            pass
    return {"labelled": created, "skipped": skipped, "remaining": len(rows) - created - skipped}


def main():
    parser = argparse.ArgumentParser(description="Human-label local contextual training examples")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--append", action="store_true", help="Resume without duplicating case IDs")
    args = parser.parse_args()
    try:
        result = annotate(args.queue, args.out, args.task, append=args.append)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
