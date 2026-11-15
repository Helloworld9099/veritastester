"""Synthetic unit tests: no provider calls, no actual model trained for deployment."""
from dataclasses import replace

import pytest

from bridgecaption.research_training import (
    MemoryExample, features, load_examples, probability, run_experiment, split_groups,
    train_predictor, choose_threshold, score,
)


def row(i, group=None, label="helpful", language="es"):
    return MemoryExample(case_id=f"id-{i}", group_id=group or f"meeting-{i // 5}",
        language=language, utterance="That deployment is fire", previous="The launch is approved",
        situation="Technical software release review", candidate_memory="The feature looked fire",
        reference_translation="That deployment is excellent", human_judgement=label,
        provenance="synthetic-consented-test")


def test_schema_rejects_missing_and_forged_labels():
    ex = row(1).__dict__
    assert MemoryExample.from_dict(ex) == row(1)
    with pytest.raises(ValueError):
        MemoryExample.from_dict({**ex, "human_judgement": "perfect"})
    with pytest.raises(ValueError):
        MemoryExample.from_dict({k: v for k, v in ex.items() if k != "provenance"})


def test_group_splits_never_mix_related_meetings():
    records = [row(i) for i in range(150)]
    train, dev, test = split_groups(records)
    groups = [{r.group_id for r in subset} for subset in (train, dev, test)]
    assert all(groups)
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])


def test_group_minimum_blocks_small_and_leaky_training():
    with pytest.raises(ValueError):
        split_groups([row(i) for i in range(5)])
    with pytest.raises(ValueError):
        run_experiment([row(i) for i in range(150)], min_examples_per_language=200)


def test_reference_does_not_leak_into_training_features():
    x = row(0)
    y = replace(x, human_judgement="harmful", reference_translation="secret solution")
    assert features(x, False) == features(y, False)
    assert features(x, True) == features(y, True)


def test_complementary_view_changes_only_multiview_features():
    x = row(0)
    y = replace(x, situation="Entirely different topic")
    assert features(x, False) == features(y, False)
    assert features(x, True) != features(y, True)


def test_training_predictor_yields_probabilities_and_abstention():
    rows = [row(i, label="helpful" if i % 3 == 0 else "harmful") for i in range(100)]
    train, dev, test = split_groups(rows)
    weights = train_predictor(train, True, epochs=3)
    p = probability(weights, test[0], True)
    assert 0 < p < 1
    threshold = choose_threshold(dev, weights, True)
    assert .5 <= threshold <= 1
    assert score(test, weights, True, threshold)["count"] == len(test)


def test_validation_rejects_duplicate_example_ids(tmp_path):
    import json
    p = tmp_path / "dataset.jsonl"
    instance = row(1).__dict__
    p.write_text(json.dumps(instance) + "\n" + json.dumps(instance) + "\n")
    with pytest.raises(ValueError, match="Duplicate case_id"):
        load_examples(p)
