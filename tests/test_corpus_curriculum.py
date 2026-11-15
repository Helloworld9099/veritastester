"""Human review candidate construction; never invent helpful/harmful labels."""
import json

import pytest

from bridgecaption.corpus_prep import make_record, private_write
from bridgecaption.corpus_curriculum import (
    human_memory_candidates, human_sense_candidates,
    leakage_report, load_normalized, read_terms, temporal_alignment_pairs
)


def turn(group, number, split="train", language="en"):
    return make_record("multi3woz", case_id=f"{group}:{number}",
        group_id=group, language=language,
        utterance=f"{group} utterance {number} about music", original_split=split)


def test_memory_candidates_use_real_turns_and_not_fake_labels():
    records = [turn("meetingA", i) for i in range(4)]
    records += [turn("meetingB", i) for i in range(4)]
    proposals = human_memory_candidates(records, seed=2026)
    assert proposals
    origins = {p["candidate_origin"] for p in proposals}
    assert origins == {"prior_turn_unjudged", "other_dialogue_unjudged"}
    assert all(p["human_judgement"] is None for p in proposals)
    assert all(p["reference_translation"] == "" for p in proposals)
    assert all(p["situation"] == "" for p in proposals)
    assert all(p["split"] == "train" for p in proposals)
    assert any(p["candidate_memory"].startswith("meetingB") and
               p["utterance"].startswith("meetingA") for p in proposals)


def test_memory_candidates_exclude_test_and_no_cross_language_distractors():
    rows = [turn("A", i, split="test") for i in range(4)]
    rows += [turn("B", i, split="train", language="fr") for i in range(4)]
    proposals = human_memory_candidates(rows)
    assert all(p["split"] != "test" for p in proposals)
    assert all(p["language"] == "fr" for p in proposals)
    assert all(p["candidate_origin"] == "prior_turn_unjudged" for p in proposals)


def test_sense_candidates_preserve_source_idiom_label_but_not_as_meaning():
    row = make_record("magpie", case_id="mag-1", group_id="idiom:spill",
        language="en", utterance="He spilled the beans about it",
        previous="He knew something secret", expression="spill the beans",
        label="idiomatic", label_kind="idiomaticity_not_exact_sense",
        original_split="train")
    result, = human_sense_candidates([row])
    assert result["meaning"] is None
    assert result["source_label"] == "idiomatic"
    assert result["situation"] == ""


def test_slang_source_dictionary_definition_remains_unverified():
    row = make_record("slanginterp", case_id="slang-1", group_id="slang:sick",
        language="en", utterance="That was sick", expression="sick",
        candidate_definition="amazing", original_split="train")
    proposal, = human_sense_candidates([row])
    assert proposal["source_suggested_definition"] == "amazing"
    assert proposal["meaning"] is None


def test_naijasenti_requires_human_terms_and_test_split_is_heldout(tmp_path):
    row = make_record("naijasenti", case_id="tweet-1", group_id="author-1",
        language="pcm", utterance="This na real wahala", original_split="train")
    terms_path = tmp_path / "slang_terms.csv"
    terms_path.write_text("language,expression\npcm,wahala\n")
    terms = read_terms(terms_path)
    proposal, = human_sense_candidates([row], terms=terms)
    assert proposal["expression"] == "wahala"
    assert proposal["meaning"] is None
    test_item = dict(row, split="test")
    assert human_sense_candidates([test_item], terms=terms) == []


def test_diaBla_evaluation_never_becomes_training_candidate():
    row = make_record("diabla", case_id="test-1", group_id="dialog-1",
        language="fr", utterance="Bonjour", expression="bonjour",
        reference_translation="Hello")
    assert row["split"] == "test"
    assert human_sense_candidates([row]) == []
    assert human_memory_candidates([row]) == []


def test_load_normalized_rejects_duplicate_ids(tmp_path):
    r = turn("group1", 1)
    private = tmp_path / "private"
    private_write(private / "a.jsonl", [r])
    private_write(private / "b.jsonl", [r])
    with pytest.raises(ValueError, match="Duplicate case ID"):
        load_normalized([private / "a.jsonl", private / "b.jsonl"])


def test_leakage_audit_rejects_groups_in_train_and_test():
    rows = [turn("one", 1, "train"), turn("one", 2, "test")]
    report = leakage_report(rows)
    assert report["cross_split_group_conflicts"] == 1
    assert not report["training_ready"]


def test_all_candidate_generation_is_deterministic():
    records = [turn("A", i) for i in range(5)] + [turn("B", i) for i in range(5)]
    assert human_memory_candidates(records, seed=11) == human_memory_candidates(records, seed=11)


def test_temporal_pairs_are_only_proxy_labels():
    data = [turn("A", i) for i in range(5)] + [turn("B", i) for i in range(5)]
    pairs = temporal_alignment_pairs(data, seed=2026)
    assert pairs
    assert {p["proxy_label"] for p in pairs} == {
        "same_dialogue_adjacent", "different_dialogue"}
    assert all(p["proxy_task"] == "temporal_coherence_not_memory_usefulness"
               for p in pairs)
    assert all(p["human_judgement"] is None for p in pairs)
    assert all(p["situation_view"] == "" for p in pairs)
