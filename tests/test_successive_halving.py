from copy import deepcopy
import hashlib

import pytest

from research.unified_training.successive_halving import (
    RaceError, development_receipt_sha256, plan_next_rung,
)


def _spec(**overrides):
    return {
        "schema_version": 1, "kind": "bounded_development_race", "scope": "development",
        "opt_in": True, "fixture_sha256": "1" * 64, "method_sha256": "2" * 64,
        "baseline_sha256": "3" * 64, "warm_start_sha256": "4" * 64,
        "directions": {"es->en": 18, "fr->en": 20},
        "candidates": {f"trial-{i}": hashlib.sha256(f"recipe-{i}".encode()).hexdigest()
                       for i in range(8)}, "rung_updates": [4, 8, 16],
        "eta": 2, "max_total_updates": 128, "patience": 0, "min_gain": 0,
        **overrides,
    }


def _rung(spec, ids, budget):
    bound = {key: spec[key] for key in (
        "fixture_sha256", "method_sha256", "baseline_sha256", "warm_start_sha256",
    )}
    return {name: {**bound, "candidate_id": name, "updates": budget, "scope": "development",
                   "m2_report_sha256": "a" * 64, "recipe_sha256": spec["candidates"][name],
                   "evaluated_checkpoint_sha256": hashlib.sha256(f"{name}/{budget}".encode()).hexdigest(),
                   "directions": {route: {"paired_rows": rows, "chrf_delta": int(name[-1]) / 10,
                                           "new_critical_errors": 0, "subgroups": "pass",
                                           "performance": "pass"}
                                  for route, rows in spec["directions"].items()}}
            for name in ids}


def _trusted(*rungs):
    return [development_receipt_sha256(receipt) for r in rungs for receipt in r.values()]


def test_halving_has_measured_synthetic_budget_savings_without_assuming_exact_resume():
    spec = _spec()
    planned = plan_next_rung(spec)
    assert planned["status"] == "suggested_research_rung"
    assert planned["projected_further_updates"] == 32
    one = _rung(spec, planned["candidate_ids"], 4)
    next_ = plan_next_rung(spec, [one], trusted_receipt_sha256s=_trusted(one))
    assert next_["candidate_ids"] == ["trial-7", "trial-6", "trial-5", "trial-4"]
    two = _rung(spec, next_["candidate_ids"], 8)
    next_ = plan_next_rung(spec, [one, two], trusted_receipt_sha256s=_trusted(one, two))
    three = _rung(spec, next_["candidate_ids"], 16)
    final = plan_next_rung(spec, [one, two, three], trusted_receipt_sha256s=_trusted(one, two, three))
    assert final["status"] == "finished_experimental"
    assert final["experimental_leads"] == ["trial-7", "trial-6"]
    assert final["allocated_updates_so_far"] == 96 < 8 * 16
    assert not final["promotion_eligible"] and not final["training_dispatched"]
    assert final["retained_incumbent"]


def test_large_gain_never_masks_a_regressed_direction():
    spec = _spec()
    one = _rung(spec, spec["candidates"], 4)
    one["trial-7"]["directions"]["es->en"]["chrf_delta"] = 50
    one["trial-7"]["directions"]["fr->en"]["chrf_delta"] = -0.5001
    one["trial-6"]["directions"]["es->en"]["new_critical_errors"] = 1
    result = plan_next_rung(spec, [one], trusted_receipt_sha256s=_trusted(one))
    assert "trial-7" not in result["candidate_ids"]
    assert "trial-6" not in result["candidate_ids"]
    assert result["stopped_by_reason"]["direction_gate"] == 2


def test_missing_comparable_performance_stops_next_rung():
    spec = _spec()
    one = _rung(spec, spec["candidates"], 4)
    one["trial-7"]["directions"]["es->en"]["performance"] = "not_verified"
    result = plan_next_rung(spec, [one], trusted_receipt_sha256s=_trusted(one))
    assert result["status"] == "not_verified" and result["next_rung"] is None


def test_changed_or_reused_receipt_is_not_authoritative():
    spec = _spec()
    one = _rung(spec, spec["candidates"], 4)
    pins = _trusted(one)
    changed = deepcopy(one)
    changed["trial-7"]["directions"]["es->en"]["chrf_delta"] = 10
    with pytest.raises(RaceError, match="pinned"):
        plan_next_rung(spec, [changed], trusted_receipt_sha256s=pins)
    changed = {**one, "trial-7": one["trial-6"]}
    with pytest.raises(RaceError, match="pinned"):
        plan_next_rung(spec, [changed], trusted_receipt_sha256s=pins)


@pytest.mark.parametrize("overrides", [{"scope": "holdout"}, {"opt_in": False}, {"eta": True},
                                       {"rung_updates": [8, 4]}])
def test_no_sealed_data_defaults_or_invalid_budgets(overrides):
    with pytest.raises(RaceError):
        plan_next_rung(_spec(**overrides))


def test_fixed_budget_and_optional_early_stop():
    tight = plan_next_rung(_spec(max_total_updates=31))
    assert tight["status"] == "stopped_at_budget"
    spec = _spec(patience=1, min_gain=0.05)
    one = _rung(spec, spec["candidates"], 4)
    two = _rung(spec, ["trial-7", "trial-6", "trial-5", "trial-4"], 8)
    result = plan_next_rung(spec, [one, two], trusted_receipt_sha256s=_trusted(one, two))
    assert result["status"] == "stopped_no_eligible_candidates"
    assert result["stopped_by_reason"]["insufficient_progress"] == 4


def test_even_a_trusted_receipt_cannot_switch_candidate_recipe():
    spec = _spec()
    one = _rung(spec, spec["candidates"], 4)
    one["trial-7"]["recipe_sha256"] = spec["candidates"]["trial-6"]
    with pytest.raises(RaceError, match="same-protocol"):
        plan_next_rung(spec, [one], trusted_receipt_sha256s=_trusted(one))
