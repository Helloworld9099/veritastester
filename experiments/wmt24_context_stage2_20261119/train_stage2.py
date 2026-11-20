#!/usr/bin/env python3
"""Second real WMT24 context-conditioned French LoRA training round.

- Warm start from SHA-pinned, trained stage-one French context adapter.
- Use NEW source/reference turns from official WMT24 TRAIN only, never previous
  stage-one TRAIN examples or publisher TEST.
- Select NEW development turns from official WMT24 VALIDATION, excluding every
  earlier selected validation source/reference (dialogue groups can recur).
- No publisher TEST files are downloaded, opened, scored, or used for selection.
- CC BY-NC 4.0: research ONLY, never integrate into commercial application.

Our research example-level dev overlaps original validation *conversation groups*;
do not claim a new independent group-level test or 90% human adequacy.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import random
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "experiments/wmt24_context_20261119"),
    str(REPO / "experiments/cpu_continuation_20261117"),
    str(REPO / "experiments/balanced_stage5_20261118"),
    str(REPO / "experiments/opus_books_20261117"),
    str(REPO / "experiments/massive_20261117"),
]
import train_context as first  # noqa: E402
import train as core  # noqa: E402
import train_stage5 as balanced  # noqa: E402
import train_massive as massive  # noqa: E402
from train_books import evaluate_frozen_test  # noqa: E402

SOURCE_SHA = {
    "train": "eff96c25a2b5b5fc436ab85d573d4b2186fb8354a31763445c02e1cc80cc76ad",
    "dev": "09f41828543e8ea13c6fd86a07a44316100c7b4f4763ee816489206b4050d442",
}
WARM_REL = "experiments/wmt24_context_20261119/saved/adapter_selected.safetensors"
WARM_SHA = "c858eb0a369baccbc93c5289b700886df896e15535cdf0222354e25f415a8b44"
SEED = 2026111902
UPDATES = 96
CHAT_TRAIN = 384
WMT_REPLAY = 768
MASSIVE_REPLAY = 384
DEV_COUNT = 256
TOTAL_TRAIN = CHAT_TRAIN + WMT_REPLAY + MASSIVE_REPLAY
STEPS = (24, 48, 72, 96)


def digest(items) -> str:
    return hashlib.sha256(
        "\n".join(hashlib.sha256(
            (x["source"] + "\x00" + x["reference"]).encode()).hexdigest()
                  for x in items).encode()
    ).hexdigest()


def new_cohort(items, old, size, *, seed):
    """Select new source/reference examples with no exact earlier overlap."""
    excluded_src = {first.canonical(r["source"]) for r in old}
    excluded_eng = {first.canonical(r["reference"]) for r in old}
    filtered = [
        item for item in items
        if first.canonical(item["source"]) not in excluded_src
        and first.canonical(item["reference"]) not in excluded_eng
    ]
    selected = first.select_group_aware(filtered, size, seed)
    if len(selected) != size:
        raise RuntimeError("Short or altered frozen cohort")
    if ({first.canonical(r["source"]) for r in selected} & excluded_src or
            {first.canonical(r["reference"]) for r in selected} & excluded_eng):
        raise RuntimeError("Previously trained/tested texts entered new cohort")
    return selected


def make_cohorts():
    history = json.loads((REPO / "experiments/wmt24_context_20261119/saved/"
                          "context_research_evaluation.json").read_text())
    assert history["language"] == "french"
    assert history["selected_based_on_contextual_DEVELOPMENT_only"] == "step240"
    assert history["official_publisher_TEST_was_never_scored"] is True
    assert history["commercial_deployment_allowed"] is False
    first_meta = history["cohorts"]
    assert first_meta["source_SHA256"] == SOURCE_SHA
    raw_train, raw_dev = first.download_frozen("train"), first.download_frozen("dev")
    if first.file_digest(raw_train) != SOURCE_SHA["train"] or first.file_digest(raw_dev) != SOURCE_SHA["dev"]:
        raise RuntimeError("Original WMT official source digest changed")
    train_all = first.parse_chat(raw_train)
    dev_all = first.parse_chat(raw_dev)
    train_groups = {r["doc_id"] for r in train_all}
    dev_groups = {r["doc_id"] for r in dev_all}
    if train_groups & dev_groups:
        raise RuntimeError("Publisher train/validation dialogue groups overlap")
    train_items = first.add_context(train_all)
    dev_items = first.add_context(dev_all)
    old_dev = first.select_group_aware(dev_items, first.DEV_N, first.SEED + 1)
    old_dev_src = {first.canonical(x["source"]) for x in old_dev}
    old_dev_ref = {first.canonical(x["reference"]) for x in old_dev}
    old_filtered = [r for r in train_items
                    if first.canonical(r["source"]) not in old_dev_src
                    and first.canonical(r["reference"]) not in old_dev_ref]
    old_train = first.select_group_aware(old_filtered, first.TRAIN_CHAT, first.SEED)
    if (len({r["doc_id"] for r in old_train}) != first_meta["distinct_training_groups"]
            or len({r["doc_id"] for r in old_dev}) != first_meta["distinct_dev_groups"]
            or len(old_train) != first_meta["selected_chat_training_n"]):
        raise RuntimeError("Stage-one WMT cohort failed deterministic reconstruction")

    new_dev = new_cohort(dev_items, old_dev + old_train, DEV_COUNT, seed=SEED + 1)
    new_chat = new_cohort(
        train_items, old_train + old_dev + new_dev, CHAT_TRAIN, seed=SEED)

    old_and_frozen = old_train + old_dev + new_dev + new_chat
    # Rehearse historical WMT24 TRAIN conversations, never old/new
    # development. Distinct dialogue groups among these training examples
    # are allowed because the same groups were legitimately trained before.
    excluded_src = {first.canonical(x["source"]) for x in new_dev + new_chat}
    excluded_en = {first.canonical(x["reference"]) for x in new_dev + new_chat}
    old_replay_eligible = [
        row for row in old_train
        if first.canonical(row["source"]) not in excluded_src
        and first.canonical(row["reference"]) not in excluded_en
    ]
    wmt_replay = first.select_group_aware(old_replay_eligible, WMT_REPLAY, SEED + 2)
    french = massive.fetch_multipart("fr-FR")
    english = massive.fetch_multipart("en-US")
    old_massive = massive.select_pairs(english, french, include_ids=True)
    saved_massive = json.loads(
        (REPO / "experiments/massive_20261117/saved/french/massive_evaluation.json").read_text())
    if old_massive["ids_sha256"] != saved_massive["split_ids_sha256"]:
        raise RuntimeError("Historical MASSIVE selection hashes changed")
    blocked = [(r["source"], r["reference"]) for r in old_and_frozen]
    blocked += old_massive["dev"] + old_massive["test"]
    replay = balanced.pick_replay(
        old_massive["train"], blocked, count=MASSIVE_REPLAY, seed=SEED)
    train_pairs = ([(x["input_with_context"], x["reference"]) for x in new_chat]
                   + [(x["input_with_context"], x["reference"]) for x in wmt_replay]
                   + replay)
    dev_pairs = [(x["input_with_context"], x["reference"]) for x in new_dev]
    plain_dev = [(x["source"], x["reference"]) for x in new_dev]
    random.Random(SEED).shuffle(train_pairs)
    meta = {
        "source_sha256": SOURCE_SHA,
        "official_train_dialogue_groups": len(train_groups),
        "official_validation_dialogue_groups": len(dev_groups),
        "old_training_pairs_excluded_from_new_chat": len(old_train),
        "old_dev_pairs_excluded_from_new_dev": len(old_dev),
        "new_chat_train_n": len(new_chat),
        "new_train_history_turn_n": sum(bool(x["prior_turns"]) for x in new_chat),
        "new_chat_train_hash": digest(new_chat),
        "old_WMT24_official_TRAIN_only_replay": len(wmt_replay),
        "old_WMT24_replay_hash": digest(wmt_replay),
        "new_dev_n": len(new_dev),
        "new_dev_history_turn_n": sum(bool(x["prior_turns"]) for x in new_dev),
        "new_dev_hash": digest(new_dev),
        "MASSIVE_old_train_only_replay": len(replay),
        "stage_one_context_warm_sha256": WARM_SHA,
        "new_training_utterances_distinct_from_prior_train_dev": True,
        "official_publisher_TEST_still_completely_sealed": True,
        "dev_dialogue_groups_can_overlap_prior_validation_groups": True,
        "license": "WMT24 Chat CC BY-NC 4.0; NONCOMMERCIAL research only",
    }
    return train_pairs, dev_pairs, plain_dev, meta


def run(output_dir):
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite a previous real training checkpoint")
    warm = REPO / WARM_REL
    if core.sha256(warm) != WARM_SHA:
        raise RuntimeError("Previous trained model SHA-256 mismatch")
    train_rows, dev_rows, plain_dev, meta = make_cohorts()
    print("STAGE_TWO_FRESH_CONTEXT_COHORTS", json.dumps(meta), flush=True)

    with tempfile.TemporaryDirectory(prefix="wmt24-context-v2-") as tmp:
        path = Path(tmp) / "train-dev.json"
        path.write_text(json.dumps({"train": train_rows, "dev": dev_rows},
                                   ensure_ascii=False))
        spec = core.LANGUAGES["french"].copy()
        core.LANGUAGES["french"] = {
            **spec,
            "data_url": path.as_uri(),
            "archive_name": "WMT24-french-context-stage2-NC",
            "warm_adapter_override": {"path": WARM_REL, "sha256": WARM_SHA},
        }
        core.SEED = SEED
        core.TOTAL_UPDATES = UPDATES
        core.TRAIN_COUNT = TOTAL_TRAIN
        core.DEV_COUNT = DEV_COUNT
        core.CHECKPOINTS = STEPS

        def parse(data: bytes, *, archive_name: str):
            if archive_name != "WMT24-french-context-stage2-NC":
                raise RuntimeError("Wrong WMT training archive")
            obj = json.loads(data)
            if (len(obj["train"]), len(obj["dev"])) != (TOTAL_TRAIN, DEV_COUNT):
                raise RuntimeError("Frozen training or validation counts changed")
            return obj["train"], obj["dev"], TOTAL_TRAIN + DEV_COUNT

        core.parse_pairs = parse
        report = core.train_experiment(REPO, output_dir, "french")
        report.update({
            "source": "Real WMT24 Chat v2, different official train/validation utterances",
            "research_only": True,
            "commercial_deployment_allowed": False,
            "warm_sha256": WARM_SHA,
            "cohorts": meta,
            "official_WMT24_TEST_never_scored": True,
        })
        (output_dir / "report.json").write_text(json.dumps(report, indent=2))

    # Same publisher validation conversation population, NEW example rows.
    # This is NOT an unbiased final test nor independent conversational rating.
    ablation = evaluate_frozen_test(
        "french", output_dir, plain_dev, report["selected"],
        previous_adapter=warm)
    evidence = {
        "language": "french",
        "stage": "context-stage-two",
        "optimizer_updates": report["completed_updates"],
        "selected_using_new_development_only": report["selected"],
        "new_cohorts": meta,
        "contextual_dev_scores": report["scores"],
        "plain_input_new_dev_ablation": ablation,
        "official_WMT24_TEST_still_sealed": True,
        "human_semantic_adequacy_90_percent_verified": False,
        "commercial_use_prohibited": True,
        "caveat": "Development observations are on the SAME publisher validation "
                  "conversation groups as prior stage, but different source/English "
                  "utterances; do not treat as independent final assessment.",
    }
    (output_dir / "stage2_evaluation.json").write_text(json.dumps(evidence, indent=2))
    print("WMT24_CONTEXT_STAGE2_RESULTS", json.dumps(evidence), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run(args.output_dir.resolve())
