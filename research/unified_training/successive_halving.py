"""Opt-in development-only allocation and early stopping, with no executor.

The caller must obtain and verify M2 receipts out of band. This planner reasons
about authenticated *inputs as supplied*; it cannot establish honest upstream
measurement, checkpoint compatibility, rights, or safety for deployment.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import re

_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_ROUTE = re.compile(r"([a-z]{2,3})->([a-z]{2,3})\Z")
_MAX_CHRF_DROP = Decimal("0.5")
_LIMIT = 100_000_000


class RaceError(ValueError):
    """The bounded development allocation is invalid or incomparable."""


def _require(condition, message):
    if not condition:
        raise RaceError(message)


def _keys(obj, keys):
    return type(obj) is dict and set(obj) == keys


def _sha(v):
    return type(v) is str and _SHA.fullmatch(v) is not None


def _uint(v, lo=1, hi=_LIMIT):
    return type(v) is int and lo <= v <= hi


def _route(v):
    m = _ROUTE.fullmatch(v) if type(v) is str else None
    return bool(m and m.group(1) != m.group(2))


def _score(value, lo=-100, hi=100):
    _require(type(value) in (int, float) and lo <= value <= hi, "Non-finite or out-of-range development measure")
    return Decimal(str(value))


def _protocol(spec):
    _require(_keys(spec, {"kind", "schema_version", "opt_in", "scope", "fixture_sha256",
                         "method_sha256", "baseline_sha256", "warm_start_sha256", "directions",
                         "candidates", "rung_updates", "eta", "max_total_updates",
                         "patience", "min_gain"}), "Invalid race protocol")
    _require(type(spec["schema_version"]) is int and spec["schema_version"] == 1
             and spec["kind"] == "bounded_development_race" and spec["scope"] == "development"
             and spec["opt_in"] is True
             and all(_sha(spec[k]) for k in ("fixture_sha256", "method_sha256", "baseline_sha256",
                                            "warm_start_sha256")),
             "Explicit opt-in and pinned development, method, baseline and warm start required")
    candidates, budgets, routes = spec["candidates"], spec["rung_updates"], spec["directions"]
    _require(type(candidates) is dict and 2 <= len(candidates) <= 256
             and all(type(c) is str and _ID.fullmatch(c) and _sha(value)
                     for c, value in candidates.items())
             and len(set(candidates.values())) == len(candidates),
             "Each candidate must have a distinct ID and immutable recipe SHA-256")
    _require(type(budgets) is list and 1 <= len(budgets) <= 16 and all(_uint(x) for x in budgets)
             and budgets == sorted(set(budgets)), "Independent rung update bounds must increase")
    _require(type(routes) is dict and 1 <= len(routes) <= 128
             and all(_route(k) and _uint(v) for k, v in routes.items()),
             "Every comparable direction must declare paired row coverage")
    _require(_uint(spec["eta"], 2, 16) and _uint(spec["max_total_updates"])
             and _uint(spec["patience"], 0, 15), "Invalid resource or stopping policy")
    gain = _score(spec["min_gain"], 0, 100)
    _require(spec["patience"] == 0 or gain > 0, "Patience requires a positive progress floor")
    return sorted(candidates), budgets, routes, gain


def development_receipt_sha256(receipt):
    """Fingerprint the exact parsed metadata, not the underlying M2 report."""
    try:
        data = json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise RaceError("Development receipt must be finite JSON metadata") from None
    return hashlib.sha256(data).hexdigest()


def _receipt(spec, data, routes, budget, candidate, pinned_receipts: set[str]):
    _require(_keys(data, {"scope", "candidate_id", "updates", "fixture_sha256", "method_sha256", "baseline_sha256",
                          "warm_start_sha256", "m2_report_sha256", "recipe_sha256",
                          "evaluated_checkpoint_sha256", "directions"})
             and data["scope"] == "development" and data["candidate_id"] == candidate
             and data["recipe_sha256"] == spec["candidates"][candidate]
             and _sha(data["evaluated_checkpoint_sha256"])
             and type(data["updates"]) is int
             and data["updates"] == budget
             and all(data[k] == spec[k] for k in ("fixture_sha256", "method_sha256", "baseline_sha256",
                                                 "warm_start_sha256"))
             and _sha(data["m2_report_sha256"])
             and development_receipt_sha256(data) in pinned_receipts,
             "Only independently pinned, same-protocol development receipts are comparable")
    values = data["directions"]
    _require(type(values) is dict and set(values) == set(routes),
             "Receipts must cover every declared direction")
    deltas, fails, unknown = [], False, False
    for route, row in values.items():
        _require(_keys(row, {"paired_rows", "chrf_delta", "new_critical_errors",
                             "subgroups", "performance"})
                 and type(row["paired_rows"]) is int and row["paired_rows"] == routes[route]
                 and row["subgroups"] in ("pass", "fail", "not_verified")
                 and row["performance"] in ("pass", "fail", "not_verified"),
                 "Direction receipt is incomplete or not paired")
        delta = None if row["chrf_delta"] is None else _score(row["chrf_delta"])
        critical = row["new_critical_errors"]
        _require(critical is None or _uint(critical, 0, _LIMIT), "Invalid critical error count")
        fails |= (delta is not None and delta < -_MAX_CHRF_DROP) or (
            critical is not None and critical > 0) or "fail" in (row["subgroups"], row["performance"])
        unknown |= delta is None or critical is None or "not_verified" in (row["subgroups"], row["performance"])
        if delta is not None:
            deltas.append(delta)
    return ("fail", None) if fails else ("not_verified", None) if unknown else ("eligible", min(deltas))


def plan_next_rung(spec, completed_rungs=(), *, trusted_receipt_sha256s=()):
    """Replay exact ordered rungs and recommend a further bounded DEV rung.

    Budget is charged as an independent run from the same pinned warm start in
    every rung: no imaginary savings from unsupported exact resume. Any
    unverified still-eligible receipt blocks a rung; a failing direction cannot
    be offset by another. The custodian verifies the M2 reports and pins each
    candidate-specific derived receipt; a claimed report hash alone is not accepted.
    No schedules are launched, and the winner is only an experimental lead.
    """
    active, budgets, routes, min_gain = _protocol(spec)
    _require(type(completed_rungs) in (tuple, list) and len(completed_rungs) <= len(budgets),
             "Invalid completed rung sequence")
    _require(type(trusted_receipt_sha256s) in (tuple, list, set, frozenset)
             and len(trusted_receipt_sha256s) <= 4096 and all(_sha(h) for h in trusted_receipt_sha256s),
             "Trusted receipt pins must be a bounded explicit collection")
    pins = set(trusted_receipt_sha256s)
    spent = 0
    previous = {}
    stagnant = {}
    stopped = {}

    def result(status, *, idx=None, projected=None, leads=()):
        return {
            "status": status, "scope": "development", "training_dispatched": False,
            "promotion_eligible": False, "retained_incumbent": True,
            "allocated_updates_so_far": spent, "next_rung": idx,
            "next_updates_per_candidate": budgets[idx] if idx is not None else None,
            "projected_further_updates": projected, "candidate_ids": list(active) if idx is not None else [],
            "experimental_leads": list(leads), "stopped_by_reason": dict(sorted(stopped.items())),
        }

    for idx, receipts in enumerate(completed_rungs):
        _require(bool(active) and type(receipts) is dict and set(receipts) == set(active),
                 "Each completed rung must include every expected candidate and no substitute")
        spent += len(active) * budgets[idx]
        _require(spent <= spec["max_total_updates"], "Prior allocation exceeded the frozen budget")
        ranked = []
        unverified = False
        for candidate in active:
            status, worst = _receipt(spec, receipts[candidate], routes, budgets[idx], candidate, pins)
            if status == "not_verified":
                unverified = True
            elif status == "fail":
                stopped["direction_gate"] = stopped.get("direction_gate", 0) + 1
            else:
                if candidate in previous and worst - previous[candidate] < min_gain:
                    stagnant[candidate] = stagnant.get(candidate, 0) + 1
                else:
                    stagnant[candidate] = 0
                previous[candidate] = worst
                if spec["patience"] and stagnant[candidate] >= spec["patience"]:
                    stopped["insufficient_progress"] = stopped.get("insufficient_progress", 0) + 1
                else:
                    ranked.append((worst, candidate))
        if unverified:
            _require(idx == len(completed_rungs) - 1, "An unverified rung cannot be followed by another")
            return result("not_verified")
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked:
            _require(idx == len(completed_rungs) - 1, "No eligible candidate for a subsequent rung")
            return result("stopped_no_eligible_candidates")
        if idx == len(budgets) - 1:
            return result("finished_experimental", leads=[candidate for _, candidate in ranked])
        keep = min(len(ranked), max(1, (len(active) + spec["eta"] - 1) // spec["eta"]))
        stopped["halved"] = stopped.get("halved", 0) + len(ranked) - keep
        active = [candidate for _, candidate in ranked[:keep]]

    idx = len(completed_rungs)
    units = len(active) * budgets[idx]
    if spent + units > spec["max_total_updates"]:
        return result("stopped_at_budget")
    return result("suggested_research_rung", idx=idx, projected=units)
