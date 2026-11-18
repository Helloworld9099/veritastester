#!/usr/bin/env python3
"""Stage-two MASSIVE LoRA training from the saved, verified stage-one adapter.

Excludes all previously consumed MASSIVE train/dev/test *identifiers and text*.
The new held-out official-test sample is scored ONCE, after dev model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "experiments/cpu_continuation_20261117"),
    str(REPO / "experiments/massive_20261117"),
    str(REPO / "experiments/opus_books_20261117"),
]
import train as base_train  # noqa: E402
import train_massive as massive  # noqa: E402
from train_books import evaluate_frozen_test  # noqa: E402

SEED = 202611182
STEPS = 320
TRAIN_COUNT = STEPS * 16
DEV_COUNT = 256
TEST_COUNT = 256
EXPECTED_STAGE_ONE_ADAPTERS = {
    "french": "5c5094a3f8dfc9d783f6e8482ae8ed2fad7e632b81b31f20f54fa8f288538ad8",
    "spanish": "e60fbadae2037686f9693af4be2e7e59b128aa44f4be8c287f40b7f893b0a004",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_previous_run(language: str, root: Path = REPO):
    """Verify the exact saved selected adapter and reconstruction manifest."""
    if language not in massive.LOCALE:
        raise ValueError("Language unsupported by saved MASSIVE adapters")
    directory = root / "experiments/massive_20261117/saved" / language
    selected = directory / "adapter_selected.safetensors"
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    evidence = json.loads((directory / "massive_evaluation.json").read_text(encoding="utf-8"))
    expected = EXPECTED_STAGE_ONE_ADAPTERS[language]
    assert report["language"] == evidence["language"] == language
    if report["selected"] != evidence["selected_on_dev"] or report["selected"] != "step256":
        raise RuntimeError("Historical frozen selection differs from expected checkpoint")
    if not (report["selected_candidate_sha256"] == expected == sha256(selected)):
        raise RuntimeError("Saved stage-one MASSIVE adapter SHA-256 mismatch")
    if evidence["revision"] != massive.REVISION:
        raise RuntimeError("MASSIVE author dataset revision changed")
    if evidence["source"] != massive.DATASET or not evidence["aligned_by_original_slurp_id"]:
        raise RuntimeError("Stage-one original corpus provenance invalid")
    return selected, evidence


def aligned_rows(english, translated, split):
    """Integrity-preserving official source/English pairing by immutable SLURP ID."""
    en_rows, src_rows = english[split], translated[split]
    en = {str(r["id"]): r for r in en_rows}
    src = {str(r["id"]): r for r in src_rows}
    if (len(en) != len(en_rows) or len(src) != len(src_rows)
            or set(en) != set(src)):
        raise ValueError("MASSIVE split does not have one-to-one original SLURP IDs")
    out = []
    for key in sorted(en):
        a, b = src[key], en[key]
        if a.get("intent") != b.get("intent"):
            raise ValueError("MASSIVE paired intents changed")
        allowed = {split} if split != "validation" else {"validation", "dev"}
        if a.get("partition") not in allowed or b.get("partition") not in allowed:
            raise ValueError("Incorrect original MASSIVE split labels")
        source, target = str(a.get("utt", "")).strip(), str(b.get("utt", "")).strip()
        if not (1 <= len(source.split()) <= 35 and 1 <= len(target.split()) <= 36):
            continue
        if not massive.canonical(source) or not massive.canonical(target):
            continue
        if massive.canonical(source) == massive.canonical(target):
            continue
        out.append((key, source, target))
    return out


def next_cohorts(english, translated, consumed, *,
                 train_n=TRAIN_COUNT, dev_n=DEV_COUNT, test_n=TEST_COUNT,
                 seed=SEED, include_ids=False):
    """Select previously UNUSED examples; never use prior test for fitting."""
    old_used_ids = {str(id) for ids in consumed["selected_ids"].values() for id in ids}
    old_source = {massive.canonical(src) for k in ("train", "dev", "test")
                  for src, _ in consumed[k]}
    old_target = {massive.canonical(en) for k in ("train", "dev", "test")
                  for _, en in consumed[k]}
    used_src, used_target = set(old_source), set(old_target)
    used_ids = set(old_used_ids)
    keep = {}
    # Test first, development second, fitting last; prevents leakage by priority.
    for split, need in (("test", test_n), ("validation", dev_n), ("train", train_n)):
        rows = aligned_rows(english, translated, split)
        random.Random(seed).shuffle(rows)
        chosen = []
        for key, src, en in rows:
            if key in used_ids:
                continue
            src_key, en_key = massive.canonical(src), massive.canonical(en)
            if src_key in used_src or en_key in used_target:
                continue
            chosen.append((key, src, en))
            used_ids.add(key)
            used_src.add(src_key)
            used_target.add(en_key)
            if len(chosen) >= need:
                break
        if len(chosen) != need:
            raise ValueError(f"Not enough fresh {split} pairs after removing old cohort")
        keep[split] = chosen
    hashes = {name: hashlib.sha256(
        "\n".join(item[0] for item in chosen).encode()).hexdigest()
        for name, chosen in keep.items()}
    result = {
        "train": [(src, en) for _, src, en in keep["train"]],
        "dev": [(src, en) for _, src, en in keep["validation"]],
        "test": [(src, en) for _, src, en in keep["test"]],
        "new_id_sha256": hashes,
        "no_previous_training_development_or_test_id_reused": True,
        "no_previous_exact_source_or_english_reference_reused": True,
    }
    if include_ids:
        result["selected_ids"] = {split: [row[0] for row in rows]
                                  for split, rows in keep.items()}
    return result


def run(language: str, output_dir: Path):
    if language not in massive.LOCALE:
        raise ValueError("Unsupported language")
    if output_dir.exists():
        raise ValueError("Output path exists; do not overwrite earlier research")
    warm_path, frozen_prior = verified_previous_run(language)
    english = massive.fetch_multipart("en-US")
    translated = massive.fetch_multipart(massive.LOCALE[language])
    # Recreate EXACT historical ID choices, verify against permanently saved hashes.
    old = massive.select_pairs(english, translated, include_ids=True)
    if old["ids_sha256"] != frozen_prior["split_ids_sha256"]:
        raise RuntimeError("Historical MASSIVE cohort cannot be reproduced identically")
    next_partitions = next_cohorts(english, translated, old)
    del english, translated, old
    print("FRESH_NONOVERLAPPING_MASSIVE_COHORT", language,
          next_partitions["new_id_sha256"], flush=True)

    with tempfile.TemporaryDirectory(prefix="bridgecaption-massive2-") as work:
        file = Path(work) / "train-dev.json"
        file.write_text(json.dumps({
            "train": next_partitions["train"],
            "dev": next_partitions["dev"]
        }, ensure_ascii=False), encoding="utf-8")

        spec = base_train.LANGUAGES[language].copy()
        base_train.LANGUAGES[language] = {
            **spec,
            "data_url": file.as_uri(),
            "archive_name": "MASSIVE-stage2-" + language,
            "warm_adapter_override": {
                "path": str(warm_path.relative_to(REPO)),
                "sha256": EXPECTED_STAGE_ONE_ADAPTERS[language],
            },
        }
        base_train.SEED = SEED
        base_train.TOTAL_UPDATES = STEPS
        base_train.TRAIN_COUNT = TRAIN_COUNT
        base_train.DEV_COUNT = DEV_COUNT
        base_train.CHECKPOINTS = (64, 128, 192, 256, 320)

        def parse(data: bytes, *, archive_name: str):
            if archive_name != "MASSIVE-stage2-" + language:
                raise RuntimeError("Wrong second-stage corpus")
            doc = json.loads(data)
            if (len(doc["train"]), len(doc["dev"])) != (TRAIN_COUNT, DEV_COUNT):
                raise RuntimeError("Prepared fresh split size is invalid")
            return doc["train"], doc["dev"], TRAIN_COUNT + DEV_COUNT

        base_train.parse_pairs = parse
        report = base_train.train_experiment(REPO, output_dir, language)
        report["dataset"] = massive.DATASET
        report["revision"] = massive.REVISION
        report["stage"] = 2
        report["prior_checkpoint_sha256"] = EXPECTED_STAGE_ONE_ADAPTERS[language]
        report["original_stage1_ids_sha256"] = frozen_prior["split_ids_sha256"]
        report["fresh_cohort_ids_sha256"] = next_partitions["new_id_sha256"]
        report["notes"] = ("Second-stage optimizer from saved MASSIVE stage-one weights. "
                           "No previous train, dev, or test example is reused in this run. "
                           "Proprietary user recordings are not collected. "
                           "chrF/BLEU are NOT percentage of correct translations.")
        (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    # First inference on fresh official test items, after dev checkpoint selection.
    heldout = evaluate_frozen_test(
        language, output_dir, next_partitions["test"], report["selected"],
        previous_adapter=warm_path,
    )
    result = {
        "language": language, "experiment": "MASSIVE second-stage unseen-data continuation",
        "stage": 2, "base_revision": report["base_revision"],
        "warm_sha256": EXPECTED_STAGE_ONE_ADAPTERS[language],
        "total_updates": report["completed_updates"],
        "train_n": TRAIN_COUNT, "dev_n": DEV_COUNT, "test_n": TEST_COUNT,
        "prior_stage_one_ids_sha256": frozen_prior["split_ids_sha256"],
        "second_stage_ids_sha256": next_partitions["new_id_sha256"],
        "new_cohorts_exclude_prior_train_dev_test_exact_text": True,
        "selected_based_on_dev_alone": report["selected"],
        "dev_scores": report["scores"],
        "fresh_unseen_official_test_scores": heldout,
        "limitations": (
            "Held-out from our adaptation steps, but dataset/domain identical and "
            "may overlap original pretrained OPUS. This is text translation only; "
            "automatic corpus metrics do not indicate 90% semantic adequacy. "
            "The newly consumed test cohort must not be reused for subsequent tuning."),
    }
    (output_dir / "stage2_evaluation.json").write_text(json.dumps(result, indent=2))
    print("STAGE2_FROZEN_RESULTS", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(massive.LOCALE), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.language, args.output_dir.resolve())
