import json
import pytest

from bridgecaption.sense_model import (SenseExample, contains_expression, fit,
                                       predict, load_predictor, train)


def example(i, label, phrase="sick", language="en"):
    previous = "That was a fantastic gaming play" if label == "excellent" else "The patient had a fever"
    return SenseExample(f"s{i}", f"meeting-{i//2}", language, phrase,
                        f"The result is {phrase}", previous,
                        "gaming" if label == "excellent" else "medical",
                        label, "synthetic-test")


def test_word_boundaries_and_unicode_scripts():
    assert contains_expression("This is sick", "sick")
    assert not contains_expression("This is a sickness", "sick")
    assert contains_expression("他上班摸鱼", "摸鱼")


def test_fit_requires_two_meanings_and_can_abstain():
    model = fit([example(i, "excellent" if i % 2 else "ill") for i in range(20)])
    assert "en\tsick" in model["expressions"]
    assert predict(model, "sick", "", "", "en", min_probability=.99) == {}


def test_unknown_term_or_language_is_not_guessed():
    model = fit([example(i, "excellent" if i % 2 else "ill") for i in range(20)])
    assert predict(model, "That is lit", "gaming", "", "en") == {}
    assert predict(model, "That is sick", "gaming", "", "es") == {}


def test_reject_untrusted_model_and_insufficient_training(tmp_path):
    artifact = tmp_path / "invalid.json"
    artifact.write_text(json.dumps({"model": {"format": "unexpected"}}))
    with pytest.raises(ValueError):
        load_predictor(artifact)
    with pytest.raises(ValueError):
        fit([example(1, "excellent")])


def test_heldout_group_splits():
    rows = [example(i, "excellent" if (i//2)%2 else "ill") for i in range(100)]
    outcome = train(rows)
    assert outcome["train_n"] + outcome["dev_n"] + outcome["test_n"] == 100
    assert outcome["model"]["format"] == "bridgecaption_senses_v1"
    assert 0.5 <= outcome["threshold"] < 1


def test_training_rejects_tiny_datasets():
    with pytest.raises(ValueError, match="60 examples"):
        train([example(i, "excellent" if i % 2 else "ill") for i in range(20)])
