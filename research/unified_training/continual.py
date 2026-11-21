"""Opt-in deterministic train-only curriculum and language replay planning.

Plans neither load models nor run training. Names in the receipt are language
directions; rows are represented by opaque identifiers. A quota is a sampling
constraint, not proof that any language retained quality or avoided forgetting.
Run independent per-language regression checks before considering promotion.
Never load sealed, test, or dev data to construct bundles for this planner.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from ..unified_models.datasets import DatasetBundle, DatasetError, training_rows

SCHEMA_VERSION = 1
_LANG = r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*"
_DIRECTION = re.compile(rf"({_LANG})->({_LANG})", re.ASCII)


class ContinualPlanError(ValueError):
    """Invalid input or unsatisfiable sampling quotas."""


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _direction(value: Any) -> str:
    match = _DIRECTION.fullmatch(value) if isinstance(value, str) else None
    if match is None or match.group(1) == match.group(2):
        raise ContinualPlanError("expected a direction like es->en using distinct lowercase language tags")
    return value


def _quotas(values: Mapping[str, int], *, allow_zero: bool) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise ContinualPlanError("quotas must be a mapping of directions to integers")
    quotas = {}
    for direction, count in values.items():
        direction = _direction(direction)
        if type(count) is not int or count < (0 if allow_zero else 1):
            raise ContinualPlanError("invalid directional sample quota")
        quotas[direction] = count
    return quotas


def row_key(row: Mapping[str, Any]) -> str:
    """Opaque key for a verified canonical row, resolvable only within the source bundles."""
    return _hash(("bridgecaption-continual-row-v1", row["dataset_id"].casefold(),
                  row["dataset_version"], row["original_id"]))


def _flag(row: Mapping[str, Any]) -> tuple[bool, bool]:
    # Only explicit annotation counts. Merely furnishing context, or observing
    # a nonempty dialect/tags field, is not evidence a row needs context/slang.
    return row["context_required"] is True, row["slang"] is True


def _draw(
    pool: list[dict[str, Any]], size: int, context_min: int, slang_min: int, seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(pool, key=lambda row: (_hash(("bridgecaption-continual-order-v1", seed, row_key(row))), row_key(row)))
    groups = defaultdict(list)
    for row in ordered:
        groups[_flag(row)].append(row)
    both, just_context, just_slang = groups[(True, True)], groups[(True, False)], groups[(False, True)]
    if len(ordered) < size:
        raise ContinualPlanError("insufficient distinct verified examples for a directional quota")
    # Use rows that deliberately cover both axes where possible. The maximum of
    # the scarcity constraints is the minimum overlap needed for feasibility.
    required_overlap = max(0, context_min - len(just_context), slang_min - len(just_slang),
                           context_min + slang_min - size)
    if required_overlap > len(both):
        raise ContinualPlanError("insufficient jointly feasible annotated context/slang examples")
    overlap = max(required_overlap, min(len(both), context_min, slang_min))
    reserve = (
        both[:overlap] + just_context[:max(0, context_min - overlap)] + just_slang[:max(0, slang_min - overlap)]
    )
    pinned = {row_key(row) for row in reserve}
    pinned.update([row_key(row) for row in ordered if row_key(row) not in pinned][:size - len(pinned)])
    return [row for row in ordered if row_key(row) in pinned]


def plan_continual_replay(
    bundles: Iterable[DatasetBundle],
    *,
    expected_manifest_sha256s: Iterable[str],
    current_direction: str,
    counts_by_direction: Mapping[str, int],
    context_min_by_direction: Mapping[str, int] | None = None,
    slang_min_by_direction: Mapping[str, int] | None = None,
    seed: int,
    purpose: str = "private_research",
) -> dict[str, Any]:
    """Plan a without-replacement current+optional replay schedule from pinned train-only bundles.

    Quotas are explicit row counts, not token balances. Noncurrent directions
    occur only if named in `counts_by_direction`. Minima count rows with verified
    explicit annotations, and a row can satisfy both minima. No oversampling,
    semantic label inference, auto-relaxation, or access to an evaluation split.
    Keep dataset pins independently trusted; hashing is not authorization to load
    private or sealed data.
    """
    current = _direction(current_direction)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ContinualPlanError("seed must be an unsigned 64-bit integer")
    counts = _quotas(counts_by_direction, allow_zero=False)
    ctx = _quotas({} if context_min_by_direction is None else context_min_by_direction, allow_zero=True)
    slang = _quotas({} if slang_min_by_direction is None else slang_min_by_direction, allow_zero=True)
    if current not in counts or (set(ctx) | set(slang)) - set(counts):
        raise ContinualPlanError("current direction and all annotation minima need an explicit sample quota")
    if any(ctx.get(d, 0) > size or slang.get(d, 0) > size for d, size in counts.items()):
        raise ContinualPlanError("annotation minima cannot exceed the directional sample quota")
    bundles, pins = tuple(bundles), tuple(expected_manifest_sha256s)

    # Look at role metadata only first. Never convert/reclassify a mixed
    # bundle. training_rows then verifies every row snapshot against the trust
    # pins, validates schema, and applies the underlying data rights policy.
    if not bundles or any(not isinstance(bundle, DatasetBundle) or any(
        not isinstance(row, Mapping) or row.get("split") != "train" or row.get("holdout") is not False
        for row in bundle.records
    ) for bundle in bundles):
        raise ContinualPlanError("planner accepts only unsealed, train-only bundles")
    try:
        rows = training_rows(bundles, purpose=purpose, expected_manifest_sha256s=pins)
    except DatasetError as exc:
        raise ContinualPlanError(str(exc)) from None
    # Use verified copies for the final eligibility and deduplication decision.
    # Compare independently verified record counts to the initial input view:
    # no silently filtered nontrain records can enter the plan.
    if len(rows) != sum(len(bundle.records) for bundle in bundles):
        raise ContinualPlanError("all verified rows must be eligible train records")
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    originals = set()
    for row in rows:
        if row["split"] != "train" or row["holdout"] is not False:
            raise ContinualPlanError("verified row is not eligible for isolated training")
        identity = (row["dataset_id"].casefold(), row["original_id"])
        if identity in originals:
            raise ContinualPlanError("same original row should not be mixed across dataset versions")
        originals.add(identity)
        direction = _direction(f'{row["source_language"].lower()}->{row["target_language"].lower()}')
        if direction not in counts:
            raise ContinualPlanError("supplied direction needs an explicit current or replay quota")
        pools[direction].append(row)

    ordered_directions = sorted(counts, key=lambda d: (d != current, d))
    chosen = {
        d: _draw(pools[d], counts[d], ctx.get(d, 0), slang.get(d, 0), seed)
        for d in ordered_directions
    }
    consumed: Counter[str] = Counter()
    total = sum(counts.values())
    selection = []
    # Deterministic deficit interleaving distributes low-quota replay across
    # the schedule rather than collecting it all at its start or end.
    for i in range(total):
        direction = max(
            (d for d in ordered_directions if consumed[d] < counts[d]),
            key=lambda d: (i + 1) * counts[d] - consumed[d] * total,
        )
        row = chosen[direction][consumed[direction]]
        consumed[direction] += 1
        selection.append({"direction": direction, "row_key": row_key(row)})
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "bridgecaption_continual_replay_plan",
        "current_direction": current,
        "input_manifest_sha256s": sorted(pins),
        "seed": seed,
        "purpose": purpose,
        "rows": total,
        "directions": {
            d: {
                "role": "current" if d == current else "replay",
                "available": len(pools[d]),
                "selected": len(chosen[d]),
                "context_minimum": ctx.get(d, 0),
                "context_required": sum(_flag(row)[0] for row in chosen[d]),
                "slang_minimum": slang.get(d, 0),
                "slang": sum(_flag(row)[1] for row in chosen[d]),
            } for d in ordered_directions
        },
        "selection": selection,
        "forgets_nothing_claim": False,
        "training_executed": False,
    }
    receipt["plan_sha256"] = _hash(receipt)
    return receipt
