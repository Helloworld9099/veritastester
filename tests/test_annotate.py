import csv
import json
import os
from pathlib import Path

import pytest

from bridgecaption.annotate import annotate, read_queue


def write_queue(folder: Path, task: str, rows: list[dict]):
    fields = sorted(rows[0])
    name = folder / (task + "_queue.csv")
    with name.open("w", encoding="utf-8", newline="") as file:
        w = csv.DictWriter(file, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return name


def memory_row(case_id="one"):
    return {"case_id": case_id, "group_id": "meeting1", "language": "en",
            "utterance": "That was sick", "previous": "She played well",
            "situation": "A sports match", "candidate_memory": "A great shot",
            "reference_translation": "That was excellent", "provenance": "licensed dialogue"}


def sense_row(case_id="one"):
    return {"case_id": case_id, "group_id": "meeting1", "language": "en",
            "expression": "sick", "utterance": "That was sick",
            "previous": "She played well", "situation": "A sports match",
            "provenance": "licensed dialogue"}


def test_memory_annotation_requires_human_judgement(tmp_path):
    csv_path = write_queue(tmp_path, "memory", [memory_row()])
    out = tmp_path / "labels.jsonl"
    lines = iter(["wrong", "helpful"])
    messages = []
    result = annotate(csv_path, out, "memory", input_fn=lambda _: next(lines),
                      output_fn=messages.append)
    assert result["labelled"] == 1
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["human_judgement"] == "helpful"
    assert row["case_id"] == "one"
    assert any("Choose helpful" in m for m in messages)
    if os.name == "posix":
        assert out.stat().st_mode & 0o777 == 0o600


def test_sense_annotation_and_resume(tmp_path):
    csv_path = write_queue(tmp_path, "sense", [sense_row(), sense_row("two")])
    out = tmp_path / "sense.jsonl"
    first = iter(["amazing", "q"])
    a = annotate(csv_path, out, "sense", input_fn=lambda _: next(first),
                 output_fn=lambda _: None)
    assert a["labelled"] == 1 and a["remaining"] == 1
    second = annotate(csv_path, out, "sense", append=True,
                      input_fn=lambda _: "excellent", output_fn=lambda _: None)
    assert second["skipped"] == 1 and second["labelled"] == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [x["case_id"] for x in rows] == ["one", "two"]


def test_existing_output_is_not_overwritten(tmp_path):
    csv_path = write_queue(tmp_path, "memory", [memory_row()])
    out = tmp_path / "labels.jsonl"
    out.write_text("prior data")
    with pytest.raises(FileExistsError):
        annotate(csv_path, out, "memory", input_fn=lambda _: "helpful",
                 output_fn=lambda _: None)
    assert out.read_text() == "prior data"


def test_queue_rejects_duplicate_ids(tmp_path):
    csv_path = write_queue(tmp_path, "memory", [memory_row(), memory_row()])
    with pytest.raises(ValueError, match="Duplicate"):
        read_queue(csv_path, "memory")
