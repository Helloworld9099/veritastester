import json
import os

import pytest

from bridgecaption.corpus_review import review, source_rows, safe_display


def write_rows(tmp_path, rows):
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows),
                    encoding="utf-8")
    return path


def memory_case(case_id="one", split="train"):
    return {
        "case_id": case_id, "group_id": "dialogue-one", "language": "en",
        "utterance": "That was sick", "previous": "We saw a great game",
        "candidate_memory": "A player scored",
        "provenance": "multi3woz:licensed-conversation",
        "split": split, "human_judgement": None,
    }


def sense_case(case_id="two", split="train"):
    return {
        "case_id": case_id, "group_id": "slang:sick", "language": "en",
        "utterance": "That was sick", "previous": "",
        "expression": "sick", "provenance": "slanginterp:research-only",
        "split": split, "meaning": None,
    }


def test_memory_reviewer_collects_grounded_two_views_and_utility(tmp_path):
    path = write_rows(tmp_path, [memory_case()])
    out = tmp_path / "private" / "memory.jsonl"
    prompts = iter(["The speakers discussed a sports game",
                    "That was amazing", "helpful"])
    result = review(path, out, "memory",
                    input_fn=lambda _: next(prompts), output_fn=lambda _: None)
    assert result["labelled"] == 1
    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["situation"] == "The speakers discussed a sports game"
    assert obj["human_judgement"] == "helpful"
    assert obj["reference_translation"] == "That was amazing"
    if os.name == "posix":
        assert out.stat().st_mode & 0o777 == 0o600


def test_slang_examples_with_no_previous_are_explicitly_marked(tmp_path):
    path = write_rows(tmp_path, [sense_case()])
    out = tmp_path / "private" / "sense.jsonl"
    responses = iter(["The speaker praised a performance", "amazing"])
    review(path, out, "sense", input_fn=lambda _: next(responses),
           output_fn=lambda _: None)
    obj = json.loads(out.read_text(encoding="utf-8"))
    assert obj["previous"] == "[NO_PREVIOUS_TURN_IN_SOURCE]"
    assert obj["meaning"] == "amazing"


def test_review_rejects_holdout_and_does_not_overwrite(tmp_path):
    path = write_rows(tmp_path, [memory_case(split="test")])
    with pytest.raises(ValueError, match="Test/holdout"):
        review(path, tmp_path / "private" / "m.jsonl", "memory")
    path = write_rows(tmp_path, [memory_case()])
    out = tmp_path / "private" / "m.jsonl"
    out.parent.mkdir()
    out.write_text("old-file")
    with pytest.raises(FileExistsError):
        review(path, out, "memory")
    assert out.read_text() == "old-file"


def test_review_resume_skips_prior_completed_rows(tmp_path):
    path = write_rows(tmp_path, [memory_case(), memory_case("three")])
    out = tmp_path / "private" / "m.jsonl"
    responses = iter(["A", "B", "neutral", "q"])
    first = review(path, out, "memory", input_fn=lambda _: next(responses),
                   output_fn=lambda _: None)
    assert first["labelled"] == 1 and first["remaining"] == 1
    responses2 = iter(["C", "D", "harmful"])
    resumed = review(path, out, "memory", append=True,
                     input_fn=lambda _: next(responses2),
                     output_fn=lambda _: None)
    assert resumed["labelled"] == 1
    assert resumed["already_reviewed"] == 1


def test_raw_terminal_control_sequences_are_removed():
    assert "\x1b" not in safe_display("\x1b[2Jsome corpus text")
    assert "some corpus text" in safe_display("\x1b[2Jsome corpus text")


def test_missing_situation_cannot_be_silently_used_for_training(tmp_path):
    path = write_rows(tmp_path, [memory_case()])
    output = tmp_path / "private" / "m.jsonl"
    responses = iter(["", "q"])
    result = review(path, output, "memory", input_fn=lambda _: next(responses),
                    output_fn=lambda _: None)
    assert result["labelled"] == 0
    assert output.read_text() == ""
