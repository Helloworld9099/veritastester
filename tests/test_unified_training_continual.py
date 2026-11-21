"""Synthetic checks for opt-in train-only curriculum; no model or private data."""

import hashlib
import itertools
import json

import pytest

from research.unified_models.datasets import DatasetSpec, convert_dataset, load_bundle
from research.unified_training import continual


COLS = {
    "original_id": "id", "source": "source", "target": "target", "split": "split",
    "context": "context", "context_required": "ctx_label", "slang": "slang_label",
    "group_id": "group",
}


def fixture(tmp_path, name, language, flags, *, bundle_version="v1", dataset_id=None, rights="private_research"):
    source = tmp_path / f"{name}.jsonl"
    rows = []
    for i, item in enumerate(flags):
        ctx, slang, role = (*item, "train") if len(item) == 2 else item
        rows.append({
            "id": f"id-private-marker-{i}", "source": f"Source body hidden {name} {i} words.",
            "target": f"Target body hidden {name} {i} words.", "split": role,
            "context": [f"Prior body hidden {name} {i}"] if ctx is not False else [],
            "ctx_label": bool(ctx), "slang_label": slang,
            "group": f"group-{name}-{i}",
        })
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spec = DatasetSpec(
        dataset_id or name, bundle_version, "artificial fixture", "CC-BY-4.0",
        rights=rights, rights_evidence="local synthetic fixture rights",
        source_language=language, target_language="en",
    )
    folder = tmp_path / name
    convert_dataset(source, folder, spec, COLS)
    pin = hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
    return load_bundle(folder, expected_manifest_sha256=pin), pin


def plan(*pinned, current="es->en", counts=None, ctx=None, slang=None, seed=981, purpose="private_research"):
    return continual.plan_continual_replay(
        [b for b, _ in pinned], expected_manifest_sha256s=[p for _, p in pinned],
        current_direction=current, counts_by_direction=counts or {current: 1}, seed=seed,
        context_min_by_direction=ctx, slang_min_by_direction=slang, purpose=purpose,
    )


def test_deterministic_opt_in_replay_and_labels_with_no_payload(tmp_path):
    es = fixture(tmp_path, "es", "es", [(False, None)] * 8)
    zh = fixture(tmp_path, "zh", "zh", [(False, False), (True, True), (False, True), (True, False)])
    fr = fixture(tmp_path, "fr", "fr", [(False, None)] * 3)
    options = dict(counts={"zh->en": 2, "es->en": 5, "fr->en": 2}, ctx={"zh->en": 1}, slang={"zh->en": 1})
    result = plan(es, zh, fr, **options)
    assert result == plan(fr, es, zh, **options)
    assert result != plan(es, zh, fr, **options, seed=982)
    assert result["directions"]["es->en"]["role"] == "current"
    assert result["directions"]["zh->en"]["role"] == "replay"
    assert result["directions"]["fr->en"]["selected"] == 2
    assert result["directions"]["zh->en"]["context_required"] >= 1
    assert result["directions"]["zh->en"]["slang"] >= 1
    assert result["training_executed"] is False and result["forgets_nothing_claim"] is False
    picks = result["selection"]
    assert len(picks) == 9 == len({p["row_key"] for p in picks})
    assert any(p["direction"] == "es->en" for p in picks[:3])
    assert any(p["direction"] != "es->en" for p in picks[-5:])
    assert all(0 < len(p["row_key"]) == 64 for p in picks)
    held = json.dumps(result, ensure_ascii=False)
    assert all(v not in held for v in ("Source body", "Target body", "Prior body", "id-private-marker", "group-es"))
    raw_receipt = {k: v for k, v in result.items() if k != "plan_sha256"}
    assert hashlib.sha256(json.dumps(raw_receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == result["plan_sha256"]


def test_replay_needs_explicit_quota_and_minima_cannot_point_outside(tmp_path):
    es = fixture(tmp_path, "es", "es", [(False, False)])
    zh = fixture(tmp_path, "zh", "zh", [(False, False)])
    assert list(plan(es)["directions"]) == ["es->en"]
    with pytest.raises(continual.ContinualPlanError, match="explicit current or replay quota"):
        plan(es, zh)
    with pytest.raises(continual.ContinualPlanError, match="all annotation minima"):
        plan(es, ctx={"zh->en": 1})
    with pytest.raises(continual.ContinualPlanError, match="current direction"):
        plan(es, counts={"zh->en": 1})


def test_quotas_are_not_silently_relaxed_or_filled_by_repeating(tmp_path):
    es = fixture(tmp_path, "es", "es", [(True, True), (True, False), (False, True)])
    with pytest.raises(continual.ContinualPlanError, match="insufficient distinct"):
        plan(es, counts={"es->en": 4})
    with pytest.raises(continual.ContinualPlanError, match="jointly feasible"):
        plan(es, counts={"es->en": 2}, ctx={"es->en": 2}, slang={"es->en": 2})
    receipt = plan(es, counts={"es->en": 3}, ctx={"es->en": 2}, slang={"es->en": 2})
    assert receipt["directions"]["es->en"]["context_required"] == 2
    assert receipt["directions"]["es->en"]["slang"] == 2
    with pytest.raises(continual.ContinualPlanError, match="cannot exceed"):
        plan(es, counts={"es->en": 1}, ctx={"es->en": 2})


@pytest.mark.parametrize("split", ("dev", "test", "holdout"))
def test_existing_evaluation_roles_rejected_before_training_view(tmp_path, monkeypatch, split):
    mixed = fixture(tmp_path, "mixed", "es", [(False, False, split), (True, True)])
    monkeypatch.setattr(continual, "training_rows", lambda *_a, **_k: pytest.fail("role guard ran too late"))
    with pytest.raises(continual.ContinualPlanError, match="unsealed, train-only"):
        plan(mixed)


def test_mutating_role_to_reclassify_heldout_rows_breaks_digest(tmp_path):
    value = fixture(tmp_path, "mixed", "es", [(False, False, "holdout"), (True, True)])
    dev = value[0].records[0]
    dev["split"], dev["holdout"] = "train", False
    with pytest.raises(continual.ContinualPlanError, match="checksum"):
        plan(value)


def test_untrusted_pin_and_unreviewed_rights_fail_closed(tmp_path):
    good, pin = fixture(tmp_path, "good", "es", [(False, False)])
    unknown = fixture(tmp_path, "unknown", "es", [(False, False)], rights="unknown")
    with pytest.raises(continual.ContinualPlanError, match="checksum"):
        plan((good, "0" * 64))
    with pytest.raises(continual.ContinualPlanError, match="unknown or prohibited"):
        plan(unknown)
    with pytest.raises(continual.ContinualPlanError, match="commercial training"):
        plan((good, pin), purpose="commercial")


def test_identical_original_version_drift_not_counted_as_replay(tmp_path):
    older = fixture(tmp_path, "older", "es", [(False, False)], dataset_id="family", bundle_version="v1")
    newer = fixture(tmp_path, "newer", "es", [(False, False)], dataset_id="family", bundle_version="v2")
    with pytest.raises(continual.ContinualPlanError, match="same original row"):
        plan(older, newer, counts={"es->en": 2})


def test_four_cell_sampling_matches_exhaustive_small_set_feasibility():
    tags = ((False, False), (True, False), (False, True), (True, True))
    for counts in itertools.product(range(3), repeat=4):
        pool = [
            {"dataset_id": "checks", "dataset_version": "v1", "original_id": f"{i}-{j}",
             "context_required": tag[0], "slang": tag[1]}
            for i, (tag, size) in enumerate(zip(tags, counts)) for j in range(size)
        ]
        for wanted in range(1, len(pool) + 1):
            outcomes = {
                (sum(row["context_required"] for row in selected), sum(row["slang"] for row in selected))
                for selected in itertools.combinations(pool, wanted)
            }
            for cmin, smin in itertools.product(range(wanted + 1), repeat=2):
                can_draw = any(c >= cmin and s >= smin for c, s in outcomes)
                if not can_draw:
                    with pytest.raises(continual.ContinualPlanError):
                        continual._draw(pool, wanted, cmin, smin, 1)
                    continue
                sample = continual._draw(pool, wanted, cmin, smin, 1)
                assert len(sample) == wanted
                assert sum(row["context_required"] for row in sample) >= cmin
                assert sum(row["slang"] for row in sample) >= smin
