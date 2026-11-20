#!/usr/bin/env python3
"""Run one of 15 FULL, independently trained Spanish contextual LoRA experiments.

Run all 15 in parallel in GitHub Actions: each starts from the checksum-verified
240-step Spanish model, trains 240 NEW optimizer updates, performs development-only
checkpoint selection, and preserves every candidate checkpoint. This is a
hyperparameter/data-generalization sweep, NOT 15 sequential updates to one model.
The previous 256-video-heldout cohort is already consumed; it is never scored
again or used for model selection in this sweep. No independent 95% adequacy claim.

Samuel y Audrey subtitles: CC BY-NC 4.0, strictly NONCOMMERCIAL RESEARCH ONLY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for item in (
    "cpu_continuation_20261117", "spanish_context_20261120",
    "massive_20261117", "balanced_stage5_20261118",
):
    sys.path.insert(0, str(ROOT / "experiments" / item))
import train as core
import train_context as source
import train_stage5 as balanced
import train_massive as massive

N_ROUNDS = 15
UPDATES = 240
TOTAL_N = UPDATES * 16
DEV_N = 128
CHECKPOINTS = (80, 160, 240)
PRIOR_WARM_REL = "experiments/spanish_context_20261120/saved/adapter_selected.safetensors"
PRIOR_STAGE_DEV_HASH = "8a95faa2be8d84e4d24710601a714a98ce85a70d132570e9a71e03c158afd5f5"
PRIOR_STAGE_TEST_HASH = "012ae617df158c0ba2302fa07d8be723fd01dd6f229d760f5cf3e8a5e03f6138"

# Predeclared configurations, not adapted to any heldout test metric.
# Context ratio = fraction of video training utterances with earlier Spanish cues.
# Replay = historical Amazon MASSIVE publisher-TRAIN examples only.
CONFIGS = {
    1:  (1.5e-5, 0.75,  768,   0),
    2:  (2.5e-5, 0.75,  768,  64),
    3:  (3.5e-5, 0.75,  768,  64),
    4:  (5.0e-5, 0.75,  768,  64),
    5:  (2.5e-5, 0.50, 1024, 128),
    6:  (3.5e-5, 0.50, 1024, 128),
    7:  (5.0e-5, 0.50, 1024, 128),
    8:  (2.5e-5, 1.00,  512, 128),
    9:  (3.5e-5, 1.00,  512, 128),
    10: (5.0e-5, 1.00,  512, 128),
    11: (1.5e-5, 0.75, 1280,  64),
    12: (2.5e-5, 0.75, 1280,  64),
    13: (3.5e-5, 0.75, 1280,  64),
    14: (4.5e-5, 0.25,  768,  64),
    15: (3.5e-5, 0.00,  768,  64),
}


def verify_warm():
    """The runner downloads the ORIGINAL successful run artifact, not test data."""
    previous = ROOT / PRIOR_WARM_REL
    if not previous.is_file():
        raise FileNotFoundError("Stage-one Spanish checkpoint must be downloaded from pinned successful run")
    report = json.loads((previous.parent / "report.json").read_text(encoding="utf-8"))
    evidence = json.loads((previous.parent / "spanish_context_evaluation.json").read_text(encoding="utf-8"))
    if report["selected"] != "step240" or report["completed_updates"] != 240:
        raise RuntimeError("Unexpected initial Spanish checkpoint selection")
    if evidence["selected_on_contextual_development_only"] != "step240":
        raise RuntimeError("Initial Spanish evidence disagrees with checkpoint")
    if evidence["data_manifest"]["dev_pair_sha256"] != PRIOR_STAGE_DEV_HASH:
        raise RuntimeError("Historical Spanish development data changed")
    if evidence["data_manifest"]["frozen_test_pair_sha256"] != PRIOR_STAGE_TEST_HASH:
        raise RuntimeError("Historical Spanish frozen test changed")
    warm_hash = core.sha256(previous)
    if warm_hash != report["selected_candidate_sha256"]:
        raise RuntimeError("Downloaded stage-one Spanish checkpoint sha256 mismatch")
    return warm_hash


def choose_pool():
    """Pin first-round cohorts, and exclude all original heldout video GROUPS.

    The old heldout is reconstructed ONLY to verify its original manifest and
    exclude its exact sources/targets. It is NEVER scored or fed to a model.
    New development samples come exclusively from the publisher's old DEV
    group, never the publisher's old TEST group.
    """
    records, raw_sources = source.download_records()
    partitions = {"train": [], "dev": [], "test": []}
    for record in records:
        partition = source.group_split(record["video_id"])
        # Known video groups cannot move between partitions.
        es = source.parse_srt(record.get("srt_es"))
        en = source.parse_srt(record.get("srt_en"))
        partitions[partition].extend(source.align_video(record["video_id"], es, en))
    first_test = source.select_balanced(partitions["test"], 256, split="test")
    first_dev = source.select_balanced(partitions["dev"], 256, split="dev", forbidden=first_test)
    if source.sha_pairs([(r["source"], r["reference"]) for r in first_test]) != PRIOR_STAGE_TEST_HASH:
        raise RuntimeError("Original heldout cannot be reconstructed; refuse training")
    if source.sha_pairs([(r["source"], r["reference"]) for r in first_dev]) != PRIOR_STAGE_DEV_HASH:
        raise RuntimeError("Original development split cannot be reconstructed; refuse training")
    selected_dev = source.select_balanced(
        partitions["dev"], DEV_N, split="fifteen-rounds-development",
        forbidden=first_test + first_dev)
    # The original 256 test examples are already CONSUMED. Protect ALL of
    # their videos, not merely those 256 selected example texts.
    protected_videos = {r["video_id"] for r in partitions["test"]}
    protected_videos |= {r["video_id"] for r in partitions["dev"]}
    if any(r["video_id"] in protected_videos for r in partitions["train"]):
        raise RuntimeError("Train pool includes a development or heldout video")
    source.assert_video_and_text_separation(
        ("original frozen test", first_test),
        ("current train", partitions["train"][:0]))  # group-level check above
    return partitions["train"], first_dev, selected_dev, first_test, raw_sources, {
        key: len({r["video_id"] for r in rows}) for key, rows in partitions.items()
    }


def quality_filter(row: dict, strict: bool) -> bool:
    """Surface-only deterministic noise filters; no label-aware model scoring."""
    src, tgt = row["source"], row["reference"]
    if not (3 <= len(src.split()) <= 36 and 3 <= len(tgt.split()) <= 45):
        return False
    if strict and not (0.65 <= len(src) / max(len(tgt), 1) <= 1.65):
        return False
    # Digit strings are reliable anchors: mistranslated quantities are harmful.
    import re
    numeric_es = re.findall(r"\b\d+\b", src)
    numeric_en = re.findall(r"\b\d+\b", tgt)
    if numeric_es and numeric_en and numeric_es != numeric_en:
        return False
    return True


def build_data(round_id: int):
    if round_id not in CONFIGS:
        raise ValueError("Round must be between 1 and 15")
    lr, context_fraction, replay_n, bucket_n = CONFIGS[round_id]
    seed = 2026112100 + round_id * 101
    trainpool, prior_dev, dev_new, original_test, raw_sources, video_groups = choose_pool()
    forbidden = original_test + prior_dev + dev_new
    filtered = [r for r in trainpool if quality_filter(r, strict=(round_id % 3 == 0))]
    train_rows = source.select_balanced(
        filtered, TOTAL_N - replay_n,
        split=f"sweep-round-{round_id}-train",
        forbidden=forbidden)
    source.assert_video_and_text_separation(
        ("original test", original_test),
        ("new development", dev_new),
        ("new training", train_rows))
    chosen_train = [(r["source"], r["reference"]) for r in train_rows]
    chosen_dev = [(r["source"], r["reference"]) for r in dev_new]
    used = chosen_train + chosen_dev + [(r["source"], r["reference"]) for r in original_test + prior_dev]

    english = massive.fetch_multipart("en-US")
    spanish = massive.fetch_multipart("es-ES")
    original_massive = massive.select_pairs(english, spanish, include_ids=True)
    evidence = json.loads((ROOT / "experiments/massive_20261117/saved/spanish/"
                           "massive_evaluation.json").read_text(encoding="utf-8"))
    if original_massive["ids_sha256"] != evidence["split_ids_sha256"]:
        raise RuntimeError("Published historical MASSIVE splits have changed")
    replay = balanced.pick_replay(
        original_massive["train"],
        forbidden=used + original_massive["dev"] + original_massive["test"],
        count=replay_n, seed=seed)
    rng = random.Random(seed)
    contextual = [
        (r["input_with_context"] if rng.random() < context_fraction else r["source"],
         r["reference"])
        for r in train_rows
    ]
    train = contextual + replay
    rng.shuffle(train)
    dev = [(r["input_with_context"], r["reference"]) for r in dev_new]
    balanced.assert_no_overlap(
        ("old consumed test", [(r["source"], r["reference"]) for r in original_test]),
        ("new development", chosen_dev),
        ("new training", chosen_train),
        ("historical MASSIVE train-only replay", replay))
    if (len(train), len(dev)) != (TOTAL_N, DEV_N):
        raise RuntimeError("Incomplete actual training/development examples")
    manifest = {
        "round": round_id, "seed": seed, "updates": UPDATES,
        "learning_rate": lr, "context_fraction": context_fraction,
        "massive_train_only_replay_count": replay_n,
        "length_bucket_size": bucket_n,
        "strict_length_quality_filter": bool(round_id % 3 == 0),
        "selected_new_subtitle_train_n": len(train_rows),
        "current_development_n": len(dev),
        "train_pairs_sha256": source.sha_pairs(train),
        "dev_pairs_sha256": source.sha_pairs(dev),
        "original_stage_development_sha256": PRIOR_STAGE_DEV_HASH,
        "consumed_old_test_sha256_exclusion_only": PRIOR_STAGE_TEST_HASH,
        "source_revision": source.SOURCE_REV,
        "source_blob_hashes": raw_sources,
        "video_group_counts": video_groups,
        "all_heldout_test_videos_excluded_from_model": True,
        "test_never_used_for_model_selection": True,
        "independent_bilingual_human_95_percent_accuracy_verified": False,
        "usage": "Research only. Source CC BY-NC 4.0. No commercial/live use.",
        "limitations": "Automatic subtitle timestamp alignment is not human-verified; "
                       "15 runs share one development domain; downstream human "
                       "assessment on genuinely unseen material is still required.",
    }
    return train, dev, manifest


def run(round_id: int, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Never overwrite an existing training run")
    warm_hash = verify_warm()
    train, dev, meta = build_data(round_id)
    print("SPANISH_ROUND_START", json.dumps(meta), flush=True)
    with tempfile.TemporaryDirectory(prefix="spanish-15fullrounds-") as temporary:
        prepared = Path(temporary) / "train_dev.json"
        prepared.write_text(json.dumps({"train": train, "dev": dev}, ensure_ascii=False),
                            encoding="utf-8")
        core.LANGUAGES["spanish"] = {
            **core.LANGUAGES["spanish"],
            "data_url": prepared.as_uri(),
            "archive_name": "fifteen-full-rounds-spanish",
            "warm_adapter_override": {"path": PRIOR_WARM_REL, "sha256": warm_hash},
        }
        core.SEED = meta["seed"]
        core.TRAIN_COUNT = TOTAL_N
        core.DEV_COUNT = DEV_N
        core.TOTAL_UPDATES = UPDATES
        core.CHECKPOINTS = CHECKPOINTS
        core.LEARNING_RATE = meta["learning_rate"]
        core.LENGTH_BUCKET_SIZE = meta["length_bucket_size"]

        def parse_pairs(blob: bytes, *, archive_name: str):
            if archive_name != "fifteen-full-rounds-spanish":
                raise RuntimeError("Unexpected training source")
            pairs = json.loads(blob)
            if (len(pairs["train"]), len(pairs["dev"])) != (TOTAL_N, DEV_N):
                raise RuntimeError("Frozen train/dev count changed")
            if (source.sha_pairs(pairs["train"]) != meta["train_pairs_sha256"] or
                    source.sha_pairs(pairs["dev"]) != meta["dev_pairs_sha256"]):
                raise RuntimeError("Frozen training or development cases altered")
            return pairs["train"], pairs["dev"], TOTAL_N + DEV_N

        core.parse_pairs = parse_pairs
        report = core.train_experiment(ROOT, output, "spanish")
        report.update({
            "fifteen_rounds_version": "parallel-independent-v1",
            "sweep_round": round_id, "completed_optimizer_updates": report["completed_updates"],
            "data_manifest": meta, "initial_selected_warm_sha256": warm_hash,
            "research_only": True, "commercial_deployment_allowed": False,
            "human_semantic_accuracy_95_percent_verified": False,
            "final_test_was_not_evaluated_in_sweep": True,
        })
        (output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    if report["completed_updates"] != UPDATES or not all(
        "step" + str(step) in report["scores"] for step in CHECKPOINTS
    ):
        raise RuntimeError("Round did not complete all optimizer updates and evaluations")
    print("SPANISH_FULL_ROUND_DONE", json.dumps({
        "round": round_id, "updates": report["completed_updates"],
        "selected_on_development": report["selected"], "scores": report["scores"],
        "elapsed_seconds": report["elapsed_seconds"],
        "frozen_external_test_used": False,
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, choices=tuple(CONFIGS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.round, args.output_dir.resolve())
