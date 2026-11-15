import json
import pytest
from bridgecaption.context_engine import ContextSelector


def test_context_disabled_removes_all_memories():
    assert ContextSelector().select("That was sick", ["He was sick"], "en",
                                    enabled=False).memories == ()


def test_ambiguous_slang_uses_only_recent_two():
    result = ContextSelector().select("That was sick!",
        ["very old", "a meeting", "a great game"], "en")
    assert result.memories == ("a meeting", "a great game")
    assert result.mode == "heuristic_untrained"


def test_unrelated_context_abstains():
    result = ContextSelector().select("The invoice totals two hundred",
                                      ["children playing outdoors"], "en")
    assert result.memories == ()
    assert result.abstained


def test_opt_in_artifact_can_abstain(tmp_path):
    f = tmp_path / "trained.json"
    f.write_text(json.dumps({
        "method": "supervised_utility_predictor_not_RL_or_JEPA",
        "models": {"two_view": {"weights": [0.0] * 8, "threshold": 1.0}}
    }))
    selection = ContextSelector(f).select("That was sick",
                                          ["We went to the doctor"], "en")
    assert selection.memories == ()
    assert selection.mode == "trained_predictor"


def test_reject_invalid_weights(tmp_path):
    f = tmp_path / "invalid.json"
    f.write_text(json.dumps({
        "method": "supervised_utility_predictor_not_RL_or_JEPA",
        "models": {"two_view": {"weights": [float("inf")] * 8, "threshold": .5}}
    }))
    with pytest.raises(ValueError):
        ContextSelector(f)
