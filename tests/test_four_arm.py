import json
from bridgecaption.evaluate_context import ARMS, build_requests, evaluate
from bridgecaption.research_training import MemoryExample


def example():
    return MemoryExample("x", "speaker-one", "en", "That was sick",
        "People discussing a game", "The match had an excellent finish",
        "That player made a great move", "That was amazing", "helpful", "synthetic")


def artifact(path):
    model = {"method": "supervised_utility_predictor_not_RL_or_JEPA",
             "models": {"single_view": {"weights": [0] * 8, "threshold": .5},
                        "two_view": {"weights": [0] * 8, "threshold": .5}}}
    path.write_text(json.dumps(model))
    return path


def test_four_arms_use_same_source_and_independent_context(tmp_path):
    requests = build_requests(example(), artifact(tmp_path / "weights.json"))
    assert set(requests) == set(ARMS)
    assert len({req.text for req in requests.values()}) == 1
    assert requests["A_no_context"].previous == []
    assert requests["A_no_context"].context_enabled is False
    assert requests["B_all_context_two_view"].previous
    assert requests["C_selected_single_view"].previous == [example().candidate_memory]
    assert requests["D_selected_two_view"].topic == example().situation


def test_four_conditions_call_only_one_configured_provider(tmp_path):
    class Fake:
        def __init__(self):
            self.calls = []
        def translate(self, req):
            self.calls.append(req)
            return "translated"
    engine = Fake()
    out = evaluate([example()], engine, artifact(tmp_path / "weights.json"))
    assert len(out) == 1
    assert len(engine.calls) == 4
