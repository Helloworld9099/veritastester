#!/usr/bin/env python3
"""Research-only MASSIVE stage 3, from verified stage-2 weights on unused utterances.

Reconstruct BOTH prior cohorts and SHA-verify frozen split identifiers first.
Never train on previously consumed train, development OR held-out test text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "experiments/cpu_continuation_20261117"),
    str(REPO / "experiments/massive_20261117"),
    str(REPO / "experiments/massive_stage2_20261117"),
    str(REPO / "experiments/opus_books_20261117"),
]
import train as base_train  # noqa: E402
import train_massive as massive  # noqa: E402
import train_stage2 as stage2  # noqa: E402
from train_books import evaluate_frozen_test  # noqa: E402

SEED = 202611183
UPDATES = 96
TRAIN_COUNT = 96 * 16
DEV_COUNT = 256
TEST_COUNT = 256
SELECT_AT = (24, 48, 72, 96)
WARM_SHA256 = {
    "french": "0219365d7d51492f8319ebb8b6a8fccbf6da25ee3525505663ac52810b0cbcba",
    "spanish": "990fa755ab733a8a3a600615f2d9ed47c8ea24935cb88258e0f5a9f00888d791",
}


def prepare_new_round(english, translated, language: str, repo_root: Path = REPO,
                      train_n: int = TRAIN_COUNT, dev_n: int = DEV_COUNT,
                      test_n: int = TEST_COUNT):
    """Return a fresh cohort only after verifying both previous consumption logs."""
    if language not in WARM_SHA256:
        raise ValueError("Unsupported model language")
    _, previous = stage2.verified_previous_run(language, repo_root)
    first = massive.select_pairs(english, translated, include_ids=True)
    if first["ids_sha256"] != previous["split_ids_sha256"]:
        raise RuntimeError("Stage-one cohort IDs differ from saved experiment evidence")

    second = stage2.next_cohorts(
        english, translated, first, seed=stage2.SEED,
        train_n=stage2.TRAIN_COUNT, dev_n=stage2.DEV_COUNT,
        test_n=stage2.TEST_COUNT, include_ids=True)
    target = repo_root / "experiments/massive_stage2_20261117/saved" / language
    saved_report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    saved_test = json.loads((target / "stage2_evaluation.json").read_text(encoding="utf-8"))
    checkpoint = target / "adapter_selected.safetensors"
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected_sha = WARM_SHA256[language]
    if (not (checksum == expected_sha == saved_report["selected_candidate_sha256"])
            or saved_report["language"] != language or saved_test["language"] != language
            or saved_report["selected"] != saved_test["selected_based_on_dev_alone"]
            or saved_report["selected"] == "previous"
            or saved_report["completed_updates"] != stage2.STEPS
            or saved_test["warm_sha256"] != stage2.EXPECTED_STAGE_ONE_ADAPTERS[language]
            or saved_test["second_stage_ids_sha256"] != second["new_id_sha256"]
            or saved_report["fresh_cohort_ids_sha256"] != second["new_id_sha256"]
            or saved_report["original_stage1_ids_sha256"] != first["ids_sha256"]
            or saved_test["prior_stage_one_ids_sha256"] != first["ids_sha256"]):
        raise RuntimeError("Stage-two saved checkpoint or frozen cohort provenance mismatch")

    combined = {
        "train": first["train"] + second["train"],
        "dev": first["dev"] + second["dev"],
        "test": first["test"] + second["test"],
        "selected_ids": {
            key: first["selected_ids"][key] + second["selected_ids"][key]
            for key in ("train", "validation", "test")
        },
    }
    fresh = stage2.next_cohorts(
        english, translated, combined, seed=SEED,
        train_n=train_n, dev_n=dev_n, test_n=test_n, include_ids=True)
    return checkpoint, fresh, {
        "old_stage1_ids_sha256": first["ids_sha256"],
        "old_stage2_ids_sha256": second["new_id_sha256"],
    }


def run(language: str, output_dir: Path):
    if language not in WARM_SHA256:
        raise ValueError("Unsupported language")
    if output_dir.exists():
        raise ValueError("Output folder already exists; never overwrite past research")
    english = massive.fetch_multipart("en-US")
    translated = massive.fetch_multipart(massive.LOCALE[language])
    warm, fresh, provenance = prepare_new_round(english, translated, language)
    del english, translated
    print("STAGE3_NEW_OFFICIAL_SPLITS_SHA256", language, fresh["new_id_sha256"], flush=True)

    with tempfile.TemporaryDirectory(prefix="bridgecaption-massive3-") as temp:
        data = Path(temp) / "train-dev.json"
        data.write_text(json.dumps({"train": fresh["train"], "dev": fresh["dev"]},
                                   ensure_ascii=False), encoding="utf-8")
        spec = base_train.LANGUAGES[language].copy()
        base_train.LANGUAGES[language] = {
            **spec,
            "data_url": data.as_uri(),
            "archive_name": "MASSIVE-stage3-" + language,
            "warm_adapter_override": {
                "path": str(warm.relative_to(REPO)),
                "sha256": WARM_SHA256[language],
            },
        }
        base_train.SEED = SEED
        base_train.TOTAL_UPDATES = UPDATES
        base_train.TRAIN_COUNT = TRAIN_COUNT
        base_train.DEV_COUNT = DEV_COUNT
        base_train.CHECKPOINTS = SELECT_AT

        def parse(data_bytes: bytes, *, archive_name: str):
            if archive_name != "MASSIVE-stage3-" + language:
                raise RuntimeError("Unexpected stage-three training source")
            doc = json.loads(data_bytes)
            if (len(doc["train"]), len(doc["dev"])) != (TRAIN_COUNT, DEV_COUNT):
                raise RuntimeError("Prepared data changed before training")
            return doc["train"], doc["dev"], TRAIN_COUNT + DEV_COUNT

        base_train.parse_pairs = parse
        report = base_train.train_experiment(REPO, output_dir, language)
        report.update({
            "source": massive.DATASET, "dataset_revision": massive.REVISION,
            "stage": 3, "previous_stage2_checkpoint_sha256": WARM_SHA256[language],
            "old_cohort_sha256": provenance,
            "new_cohort_ids_sha256": fresh["new_id_sha256"],
            "stage3_utterances_from_original_author_partitions": True,
            "no_previous_train_dev_or_test_text_reused": True,
            "license": "CC BY 4.0 - AmazonScience MASSIVE",
            "caveat": "chrF/BLEU not translation-adequacy percentage; no bilingual human review.",
        })
        (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # This is the first time this stage's held-out test portion is scored.
    # Selection is already fixed on development scores.
    final_scores = evaluate_frozen_test(
        language, output_dir, fresh["test"], report["selected"], previous_adapter=warm)
    final = {
        "language": language, "stage": 3, "source": massive.DATASET,
        "dataset_revision": massive.REVISION,
        "stage2_warm_sha256": WARM_SHA256[language],
        "old_cohort_sha256": provenance,
        "new_ids_sha256": fresh["new_id_sha256"],
        "train_n": TRAIN_COUNT, "dev_n": DEV_COUNT, "test_n": TEST_COUNT,
        "no_previous_train_dev_or_test_text_reused": True,
        "selected_using_dev_only": report["selected"],
        "development_scores": report["scores"],
        "first_and_only_new_test_scores": final_scores,
        "warning": (
            "MASSIVE test is independent of previous adaptation examples, not "
            "necessarily independent of original Marian pretraining. Text only. "
            "Not a substitute for blind bilingual adequacy ratings / 90% evidence."),
    }
    (output_dir / "stage3_evaluation.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8")
    print("STAGE3_FROZEN_RESULTS", json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=tuple(WARM_SHA256), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run(args.language, args.output_dir.resolve())
