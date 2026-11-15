"""Offline contract tests for eight corpus adapters. No data download or training."""
import csv
import json
import os

import pytest

from bridgecaption.corpus_prep import (prepare, make_record, private_write,
                                       audit, group_split)


def write_jsonl(tmp_path, name, records):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                    encoding="utf-8")
    return path


def write_csv(tmp_path, name, records):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return path


def test_magpie_keeps_actual_idiom_label_and_source_split(tmp_path):
    path = write_jsonl(tmp_path, "magpie.jsonl", [
        {"id": 52, "idiom": "bear fruit", "context":
         ["They started the project", "Their work began to bear fruit",
          "The project succeeded"], "label": "i", "split": "test",
         "document_id": "fictional_doc"}])
    item, = prepare("magpie", path)
    assert item["role"] == "idiomaticity"
    assert item["source_label"] == "idiomatic"
    assert item["label_kind"] == "idiomaticity_not_exact_sense"
    assert item["split"] == "test"
    assert item["previous"] == "They started the project"
    assert item["reference_translation"] == ""


def test_magpie_refuses_ambiguous_target_location(tmp_path):
    path = write_jsonl(tmp_path, "magpie.jsonl", [
        {"id": 4, "idiom": "break a leg", "context":
         ["break a leg today", "break a leg tomorrow"], "label": "i"}])
    with pytest.raises(ValueError, match="not uniquely"):
        prepare("magpie", path)


def test_official_magpie_training_split_alias(tmp_path):
    path = write_jsonl(tmp_path, "official.jsonl", [{
        "id": 9, "idiom": "bear fruit", "context": ["It will bear fruit"],
        "label": "i", "split": "training", "document_id": "doc-9"}])
    item, = prepare("magpie", path)
    assert item["split"] == "train"
    assert item["original_split"] == "training"


def test_dialogue_preserves_source_speaker_turn_and_parallel_group(tmp_path):
    path = tmp_path / "dialogue.json"
    path.write_text(json.dumps({"aligned-1": {"log": [
        {"text": "Where next?", "speaker": "USER", "turn_id": "source-7"}]}}))
    en, = prepare("multi3woz", path, language="en", source_split="train")
    fr, = prepare("multi3woz", path, language="fr", source_split="train")
    assert en["extra"]["speaker"] == "USER"
    assert en["extra"]["source_turn_id"] == "source-7"
    assert en["extra"]["source_dialogue_id"] == "aligned-1"
    assert en["group_id"] == fr["group_id"]


def test_semeval_original_header_and_label_are_honored(tmp_path):
    path = write_csv(tmp_path, "semeval.csv", [
        {"DataID": "foo", "Language": "EN", "MWE": "spill the beans",
         "Previous": "We were discussing a secret", "Target": "He spilled the beans",
         "Next": "Everyone was surprised", "Label": "0"}])
    item, = prepare("semeval", path, source_split="train")
    assert item["source_label"] == "idiomatic"
    assert item["previous"] == "We were discussing a secret"
    assert item["split"] == "train"
    assert item["expression"] == "spill the beans"


def test_slanginterp_context_mask_preserves_candidate_not_train_label(tmp_path):
    path = write_csv(tmp_path, "slang.csv", [
        {"Word": "chido", "Definition": "cool or impressive",
         "Example": "That is [*SLANGAAAP*]."}])
    item, = prepare("slanginterp", path, language="es",
        columns={"expression": "Word", "definition": "Definition", "utterance": "Example"})
    assert item["utterance"] == "That is chido."
    assert item["candidate_definition"] == "cool or impressive"
    assert item["source_label"] == ""
    assert item["label_kind"] == "unverified_dictionary_definition"


def test_multi3woz_keeps_all_turns_in_one_dialogue_and_language(tmp_path):
    path = tmp_path / "multi3.json"
    path.write_text(json.dumps({"dialog-01": {"log": [
        {"text": "Where are you going?"}, {"text": "To the station."},
        {"text": "Which station?"}]}}), encoding="utf-8")
    rows = prepare("multi3woz", path, language="en")
    assert len(rows) == 3
    assert len({r["group_id"] for r in rows}) == 1
    assert rows[1]["previous"] == "Where are you going?"
    assert rows[2]["previous"] == "Where are you going? To the station."
    assert all(r["source_label"] == "" for r in rows)


def test_bangor_requires_normalized_transcript_and_preserves_group(tmp_path):
    path = write_csv(tmp_path, "miami.tsv", [
        {"conversation_id": "miami01", "speaker": "A", "language": "es",
         "utterance": "Hola, ¿qué tal?"},
        {"conversation_id": "miami01", "speaker": "B", "language": "es",
         "utterance": "Todo bien."}])
    # A CSV with TSV extension needs tab-delimited content.
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["conversation_id","speaker","language","utterance"],
                                delimiter="\t")
        writer.writeheader()
        writer.writerow({"conversation_id":"miami01","speaker":"A","language":"es",
                         "utterance":"Hola, ¿qué tal?"})
        writer.writerow({"conversation_id":"miami01","speaker":"B","language":"es",
                         "utterance":"Todo bien."})
    rows = prepare("bangor", path)
    assert len(rows) == 2
    assert rows[1]["previous"] == "Hola, ¿qué tal?"
    assert rows[0]["group_id"] == rows[1]["group_id"]


def test_diabla_is_never_training_data_and_requires_true_dialogue_id(tmp_path):
    row = {"id":"1", "orig":"Ça va ?", "ref":"How are you?",
           "utterance_meta":{"lang":"fr"}}
    path = write_jsonl(tmp_path, "diabla.jsonl", [row])
    with pytest.raises(ValueError, match="verified conversation ID"):
        prepare("diabla", path)
    path = write_jsonl(tmp_path, "diabla_grouped.jsonl",
                       [{**row, "dialogue_id":"actual-conversation-01"}])
    item, = prepare("diabla", path)
    assert item["split"] == "test"
    assert item["role"] == "evaluation_only"
    assert item["reference_translation"] == "How are you?"


def test_naijasenti_sentiment_not_misused_as_meaning(tmp_path):
    path = write_csv(tmp_path, "naija.csv", [
        {"id":"7", "text":"This one na real wahala", "sentiment":"positive"}])
    item, = prepare("naijasenti", path, language="pcm",
        columns={"utterance":"text","sentiment":"sentiment"})
    assert item["label_kind"] == "sentiment_is_not_sense"
    assert item["source_label"] == ""
    assert item["role"] == "informal_review"


def test_okwu_parallel_pair_has_reference_not_dialogue_context(tmp_path):
    path = write_csv(tmp_path, "okwu.csv", [
        {"source_text":"Habari", "target_text":"Hello"}])
    item, = prepare("okwu", path, language="sw",
        columns={"utterance":"source_text", "reference_translation":"target_text"})
    assert item["reference_translation"] == "Hello"
    assert item["previous"] == ""
    assert item["role"] == "parallel_text"


def test_group_split_is_stable_not_row_order():
    assert group_split("a sample family") == group_split("a sample family")


def test_cross_split_group_leak_is_detected():
    a = make_record("multi3woz", case_id="1", group_id="family-1",
                    language="en", utterance="One", original_split="train")
    b = make_record("multi3woz", case_id="2", group_id="family-1",
                    language="en", utterance="Two", original_split="test")
    assert audit([a,b])["cross_split_group_conflicts"] == 1


def test_private_write_refuses_tracked_paths_and_overwrite(tmp_path):
    r = make_record("naijasenti", case_id="1", group_id="family-1",
                    language="yo", utterance="Good day")
    with pytest.raises(ValueError, match="private"):
        private_write(tmp_path / "public.jsonl", [r])
    target = tmp_path / "private" / "rows.jsonl"
    private_write(target, [r])
    assert json.loads(target.read_text(encoding="utf-8"))["dataset"] == "naijasenti"
    if os.name == "posix":
        assert target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        private_write(target, [r])


def test_empty_and_missing_schema_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        prepare("magpie", write_jsonl(tmp_path, "empty.jsonl", []))
    with pytest.raises(ValueError):
        prepare("multi3woz", write_jsonl(tmp_path, "incorrect.jsonl", [{"hello":"world"}]),
                language="en")


def test_slanginterp_official_headers_and_partition(tmp_path):
    path = write_csv(tmp_path, "UD_test.csv", [
        {"Slang": "balsa", "Definition": "a colloquial meaning",
         "Usage Context": "This was [*SLANGAAAP*] all along."}])
    row, = prepare("slanginterp", path, language="en")
    assert row["split"] == "test"
    assert row["expression"] == "balsa"
    assert row["candidate_definition"] == "a colloquial meaning"
    assert row["source_sha256"]


def test_okwu_official_fixture_columns_and_source_lang(tmp_path):
    path = write_jsonl(tmp_path, "okwu.jsonl", [
        {"record_id": "YOR-01", "source_lang": "yor", "source_text": "Báwo ní",
         "english_translation": "Hi", "verified": True, "source": "lessons"}])
    row, = prepare("okwu", path)
    assert row["language"] == "yo"
    assert row["reference_translation"] == "Hi"
    assert row["extra"]["source_verified_claim"] is True
    assert row["source_sha256"]


def test_skip_invalid_quarantines_but_does_not_fabricate(tmp_path):
    path = write_jsonl(tmp_path, "magpie.jsonl", [
        {"id": 1, "idiom": "bear fruit",
         "context": ["Work can bear fruit"], "label": "i"},
        {"id": 2, "idiom": "other expression",
         "context": ["The text does not have the phrase"], "label": "i"}])
    with pytest.raises(ValueError, match="row 2"):
        prepare("magpie", path)
    stats = {}
    rows = prepare("magpie", path, skip_invalid=True, stats=stats)
    assert len(rows) == 1
    assert stats["accepted"] == 1
    assert sum(stats["rejected"].values()) == 1
