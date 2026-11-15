"""Idiom ablation is a distinct supervised task, not slang translation."""
import pytest

from bridgecaption.corpus_prep import make_record
from bridgecaption.idiom_ablation import eligible, render, score_binary, validate_partitions


def row(i, split, label, expression=None, previous="We talked yesterday"):
    expression = expression or f"idiom {i}"
    return make_record("magpie", case_id=f"{split}:{i}:{label}",
        group_id=f"idiom:en:{expression}", language="en",
        utterance=f"Someone said {expression} to everyone",
        previous=previous, expression=expression, label=label,
        label_kind="idiomaticity_not_exact_sense", original_split=split)


def test_three_way_idiom_type_grouping_not_sentence_shuffle():
    rows = [row(i, split, label)
            for split, ids in (("train", range(6)), ("dev", range(6,9)),
                               ("test", range(9,12)))
            for i in ids for label in ("literal", "idiomatic")]
    parts = validate_partitions(rows)
    assert len(parts["train"]) == 12
    assert len(parts["dev"]) == 6
    assert len(parts["test"]) == 6


def test_same_idiom_in_training_and_test_is_refused():
    rows = [row(i, "train", l) for i in range(6) for l in ("idiomatic","literal")]
    rows += [row(i, "dev", l) for i in range(6,9) for l in ("idiomatic","literal")]
    rows += [row(i, "test", l) for i in range(9,12) for l in ("idiomatic","literal")]
    rows.append(row(0, "test", "idiomatic"))
    with pytest.raises(ValueError, match="Same idiom"):
        validate_partitions(rows)


def test_two_view_adds_grounded_context_without_leaking_gold_label():
    r = row(1, "train", "idiomatic")
    single = render(r, two_view=False)
    double = render(r, two_view=True)
    assert "We talked yesterday" not in single
    assert "We talked yesterday" in double
    assert "idiomatic" not in single
    assert "idiomatic" not in double


def test_metrics_do_not_hide_imbalanced_errors():
    result = score_binary([0, 0, 1, 1], [0, 0, 0, 0])
    assert result["macro_f1"] < .5
    assert result["by_class"]["idiomatic"]["recall"] == 0.0


def test_only_magpie_semeval_binary_labels_are_eligible():
    a = row(1, "train", "literal")
    b = make_record("slanginterp", case_id="s", group_id="term",
        language="en", utterance="That is sick",
        expression="sick", candidate_definition="amazing")
    assert eligible([a,b]) == [a]
