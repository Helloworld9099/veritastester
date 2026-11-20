#!/usr/bin/env python3
"""ONE final bounded, review-directed ES->EN CPU LoRA continuation.

Checkpoint starts from the *accepted* v2 Spanish learned weights (not zero).
Train solely on new aligned MASSIVE publisher TRAIN examples, excluding all
original 500 reviewed prompts and reconstructed three earlier v2 train rounds.
Choose a never-trained official VALIDATION development cohort different from
the earlier 128 dev examples and from the reviewer workbook. Never open TEST.
The reviewer-reported failure categories guide weighting, NOT target strings.

This round is a *research* candidate; there are no human meaning labels to
authorize product promotion or 95% claims. No French or app files are edited.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import resource
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"research/es_en_v2")]
from research.es_en_v2 import train_three as previous
def load_review():
    """Load the already generated 500 real outputs without introducing new QA dependencies."""
    path = ROOT / "research/es_en_v2/reviewer_examples_500/BridgeCaption_500_Real_Model_Translations_Review.csv"
    manifest_path = ROOT / "research/es_en_v2/reviewer_examples_500/manifest.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = meta["files_sha256"]["BridgeCaption_500_Real_Model_Translations_Review.csv"]
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if (len(rows) != 500
        or len({row["Case ID"] for row in rows}) != 500
        or {row["Direction"] for row in rows} != {"es-en", "en-es"}
        or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        or meta["actual_inference_executed"] is not True):
        raise RuntimeError("Original 500 real model outputs changed or missing")
    return rows


core = previous.core
PARENT = "experiments/es_en_v2_runs/candidate_after_three.safetensors"
PARENT_SHA = "3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71"
SAVED = ROOT / "research/es_en_v2/review_directed_round_20261120"
UPDATES = 64
N_TRAIN = UPDATES * 16
N_DEV = 128
SEED = 2026112098
LR = 1.0e-5
PRIORITY_SCENARIOS = {
    "alarm", "calendar", "datetime", "email", "iot", "lists", "music",
    "qa", "recommendation", "social", "transport",
}
PRIORITY_INTENT_TERMS = (
    "set", "remove", "delete", "alarm", "schedule", "calendar", "remind",
    "clear", "modify", "replace", "skip", "play", "time", "date", "what",
)


def hashed_ids(items: list[str]) -> str:
    h = hashlib.sha256()
    for key in sorted(items):
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def corpus():
    """Prepare source/target-disjoint MASSIVE pairs with pinned publisher splits."""
    print("LOADING_OFFICIAL_TRAIN_AND_VALIDATION_ONLY", flush=True)
    en_train = previous.aligned("en-US", "train")
    es_train = previous.aligned("es-ES", "train")
    en_dev = previous.aligned("en-US", "validation")
    es_dev = previous.aligned("es-ES", "validation")

    reviewed = load_review()
    review_sources = {
        previous.norm(str(row["Original spoken text"]))
        for row in reviewed if row["Direction"] == "es-en"
    }
    # EN->ES source utterances must not be used for ES->EN model training
    # either, where they correspond to English publisher reference text.
    review_english = {
        previous.norm(str(row["Original spoken text"]))
        for row in reviewed if row["Direction"] == "en-es"
    }
    valid, validation_stats = previous.vetted_rows(es_dev, en_dev, 2.5)
    old_dev = sorted(valid, key=lambda x: hashlib.sha256(
        ("v2dev" + x[0]).encode()).hexdigest())[:128]
    if len(old_dev) != 128:
        raise RuntimeError("Historic development split cannot be reproduced")
    old_dev_ids = {key for key, _, _ in old_dev}
    new_dev_pool = [
        row for row in valid
        if row[0] not in old_dev_ids
        and previous.norm(row[1]) not in review_sources
        and previous.norm(row[2]) not in review_english
    ]
    new_dev_pool.sort(key=lambda x: hashlib.sha256(
        ("review-directed-fresh-dev-" + x[0]).encode()).hexdigest())
    chosen_dev = new_dev_pool[:N_DEV]
    if len(chosen_dev) != N_DEV:
        raise RuntimeError("Not enough unused official validation examples")
    dev = [(src, tgt) for _, src, tgt in chosen_dev]
    dev_ids = {key for key, _, _ in chosen_dev}
    if dev_ids & old_dev_ids:
        raise RuntimeError("Old development examples leaked into new model selection")

    # Reconstruct previous allowed v2 training examples (IDs not in saved
    # manifests), to avoid simply training on the same 3 x 1024 pairs again.
    # This does NOT claim older unrelated projects never saw the same base text.
    used_src, used_tgt = set(), set()
    for exp in previous.EXPERIMENTS:
        old_train, _, _ = previous.build_data(
            exp, en_train, es_train, en_dev, es_dev)
        used_src.update(previous.norm(src) for src, _ in old_train)
        used_tgt.update(previous.norm(tgt) for _, tgt in old_train)
    valid_sources = {previous.norm(src) for _, src, _ in valid}
    valid_targets = {previous.norm(tgt) for _, _, tgt in valid}
    train_candidates, train_stats = previous.vetted_rows(es_train, en_train, 2.5)
    filtered = [
        row for row in train_candidates
        if previous.norm(row[1]) not in used_src
        and previous.norm(row[2]) not in used_tgt
        and previous.norm(row[1]) not in review_sources
        and previous.norm(row[2]) not in review_english
        and previous.norm(row[1]) not in valid_sources
        and previous.norm(row[2]) not in valid_targets
    ]
    if len(filtered) < N_TRAIN:
        raise RuntimeError("Insufficient previously unused eligible train examples")

    by_id = {str(row["id"]): row for row in es_train}
    focus, general = [], []
    for row in filtered:
        item_id, src, tgt = row
        attrs = by_id[item_id]
        intent = str(attrs.get("intent", "")).casefold()
        scenario = str(attrs.get("scenario", "")).casefold()
        length = len(src.split())
        is_relevant = (
            scenario in PRIORITY_SCENARIOS
            or any(term in intent for term in PRIORITY_INTENT_TERMS)
            or length >= 10
            or any(c.isdigit() for c in src)
        )
        (focus if is_relevant else general).append(row)
    rng = random.Random(SEED)
    rng.shuffle(focus)
    rng.shuffle(general)
    target_focus = 704
    selected = focus[:target_focus]
    remaining_pool = general + focus[target_focus:]
    rng.shuffle(remaining_pool)
    selected += remaining_pool[:N_TRAIN - len(selected)]
    if len(selected) != N_TRAIN:
        raise RuntimeError("Not enough selected examples for 64 complete updates")
    rng.shuffle(selected)
    training = [(src, tgt) for _, src, tgt in selected]
    if len({previous.norm(s) for s, _ in training}) != N_TRAIN:
        raise RuntimeError("Duplicate selected source speech")
    if len({previous.norm(t) for _, t in training}) != N_TRAIN:
        raise RuntimeError("Duplicate selected translation reference")
    if ({previous.norm(s) for s, _ in training} &
        ({previous.norm(s) for s, _ in dev} | review_sources)):
        raise RuntimeError("Reviewed or development source contaminated training")
    if ({previous.norm(t) for _, t in training} &
        ({previous.norm(t) for _, t in dev} | review_english)):
        raise RuntimeError("Reviewed or development reference contaminated training")

    scenario_counts = Counter(str(by_id[item_id].get("scenario", ""))
                              for item_id, _, _ in selected)
    manifest = {
        "status": "official_train_development_only_corpus_prep",
        "source": previous.SOURCE,
        "revision": previous.REV,
        "training_license": "CC-BY-4.0 (AmazonScience MASSIVE)",
        "training_split": "official TRAIN",
        "dev_split": "official VALIDATION, previous 128 and reviewed sources excluded",
        "author_test_split_never_downloaded": True,
        "training_count": len(training),
        "development_count": len(dev),
        "previous_v2_train_pair_text_excluded": True,
        "historical_v2_dev_ids_excluded": True,
        "all_500_reviewed_prompt_text_excluded_from_training": True,
        "all_500_reviewed_prompt_text_excluded_from_new_dev": True,
        "license_and_publisher_id_alignment_heuristically_verified": True,
        "no_independent_pairwise_human_semantic_validation": True,
        "no_reviewer_source_corrections_as_targets": True,
        "train_focused_examples": min(len(focus), target_focus),
        "training_scenarios": dict(sorted(scenario_counts.items())),
        "eligible_training_count_after_exclusions": len(filtered),
        "source_original_id_set_sha256": hashed_ids([item_id for item_id, _, _ in selected]),
        "development_id_set_sha256": hashed_ids([item_id for item_id, _, _ in chosen_dev]),
        "training_pair_sha256": previous.sha_pairs(training),
        "development_pair_sha256": previous.sha_pairs(dev),
        "original_training_filter": train_stats,
        "original_validation_filter": validation_stats,
        "seed": SEED,
        "reason_for_focus": "Human review flagged wrong actions, times, names, numbers and omitted clauses; "
                            "source domains weighted without using the reviewed text as training data.",
        "limitation": "MASSIVE utterances are mostly isolated task utterances; this round does "
                      "NOT train conversation history, regional slang or implicit interpersonal meaning.",
    }
    return training, dev, manifest


def run():
    if core.sha256(ROOT / PARENT) != PARENT_SHA:
        raise RuntimeError("Refusing training: learned Spanish v2 parent SHA changed")
    if SAVED.exists():
        raise FileExistsError("Refuse to overwrite any previous training/checkpoints")
    training, dev, manifest = corpus()
    if len(training) != N_TRAIN or len(dev) != N_DEV:
        raise RuntimeError("Miscounted dataset sizes")

    original_spec = dict(core.LANGUAGES["spanish"])
    core.LANGUAGES["spanish"] = {
        **original_spec,
        "archive_name": "review-feedback-new-MASSIVE-TRAIN-only",
        "warm_adapter_override": {"path": PARENT, "sha256": PARENT_SHA},
    }
    core.SEED = SEED
    core.TRAIN_COUNT = N_TRAIN
    core.DEV_COUNT = N_DEV
    core.TOTAL_UPDATES = UPDATES
    core.CHECKPOINTS = (32, 64)
    core.LEARNING_RATE = LR
    core.LENGTH_BUCKET_SIZE = 64

    with TemporaryDirectory(prefix="bridgecaption-review-directed-") as temporary:
        data_path = Path(temporary) / "source-licensed-pairs.json"
        data_path.write_text(json.dumps({"train": training, "dev": dev},
                                        ensure_ascii=False), encoding="utf-8")
        core.LANGUAGES["spanish"]["data_url"] = data_path.as_uri()

        def pinned_parser(blob: bytes, *, archive_name: str):
            if archive_name != "review-feedback-new-MASSIVE-TRAIN-only":
                raise RuntimeError("Unexpected dataset family")
            stored = json.loads(blob)
            if previous.sha_pairs(stored["train"]) != manifest["training_pair_sha256"]:
                raise RuntimeError("Training pair checksum changed")
            if previous.sha_pairs(stored["dev"]) != manifest["development_pair_sha256"]:
                raise RuntimeError("Development pair checksum changed")
            return stored["train"], stored["dev"], N_TRAIN + N_DEV
        core.parse_pairs = pinned_parser

        elapsed_start = time.monotonic()
        result = core.train_experiment(ROOT, SAVED / "weights", "spanish")
        if (result["completed_updates"] != UPDATES or
            result["warm_adapter_sha256"] != PARENT_SHA):
            raise RuntimeError("Training did not finish or reset weights")
        if result["base_model"] != "Helsinki-NLP/opus-mt-es-en":
            raise RuntimeError("Unknown base translation model")
        prior = result["scores"]["previous"]
        selected_label = result["selected"]
        if selected_label == "previous":
            candidate = PARENT
            selected_sha = PARENT_SHA
            accepted_dev_only = False
        else:
            if selected_label not in ("step32", "step64"):
                raise RuntimeError("Unexpected selected optimizer checkpoint")
            better = result["scores"][selected_label]
            if not core.accepted_candidate(better, prior):
                raise RuntimeError("Invalid automated development gate")
            candidate = str((SAVED / "weights/candidate_selected.safetensors").relative_to(ROOT))
            selected_sha = core.sha256(ROOT / candidate)
            accepted_dev_only = True

    # Pinned original pretrained base can be re-downloaded from exact revision
    # and hashed; NEVER store a new 300 MB duplicate in source control.
    shutil.rmtree(SAVED / "weights/pinned_base")
    manifest["original_trained_checkpoint_sha256"] = PARENT_SHA
    manifest["selected_research_checkpoint_sha256"] = selected_sha
    manifest["source_references_not_published"] = True
    (SAVED / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "completed_one_real_spanish_continuation_training_round",
        "checkpoint_ancestry": "WARM_STARTED from accepted v2 learned weights; fresh optimizer state",
        "base_checkpoint_sha256": PARENT_SHA,
        "parent_unchanged": core.sha256(ROOT / PARENT) == PARENT_SHA,
        "selected_for_further_research": candidate,
        "selected_sha256": selected_sha,
        "dev_only_candidate_accepted": accepted_dev_only,
        "training_updates_completed": UPDATES,
        "training_examples": N_TRAIN,
        "new_development_examples": N_DEV,
        "dev_scores_are_automatic_similarity_not_accuracy": result["scores"],
        "train_dataset_sha256": manifest["training_pair_sha256"],
        "dev_dataset_sha256": manifest["development_pair_sha256"],
        "model_weights_restart_from_zero": False,
        "optimizer_state_continued": False,
        "reviewed_500_examples_not_used_for_training": True,
        "official_final_holdout_never_accessed": True,
        "no_human_95_percent_claim": True,
        "no_context_or_slang_quality_claim": True,
        "french_training_changed": False,
        "production_default_activated": False,
        "paid_spend_usd": 0,
        "elapsed_seconds": time.monotonic() - elapsed_start,
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "next_actions_automatically_started": False,
        "message": "One bounded real Spanish training round finished. Stop after preserving its verified artifacts.",
    }
    (SAVED / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sha_records = [
        {"path": str(f.relative_to(SAVED)), "sha256": core.sha256(f), "bytes": f.stat().st_size}
        for f in sorted(SAVED.rglob("*")) if f.is_file()
    ]
    (SAVED / "sha256_manifest.json").write_text(
        json.dumps(sha_records, indent=2) + "\n", encoding="utf-8")
    print("ONE_SPANISH_REVIEW_TRAINING_ROUND_COMPLETED", json.dumps({
        "status": summary["status"],
        "parent_sha256": PARENT_SHA,
        "selected_sha256": selected_sha,
        "selected": result["selected"],
        "dev_accepted": accepted_dev_only,
        "dev_scores": result["scores"],
        "training_updates": UPDATES,
        "training_n": N_TRAIN,
        "dev_n": N_DEV,
        "train_examples_from_reviewed_500": 0,
        "elapsed_seconds": summary["elapsed_seconds"],
        "paid_spend": 0,
        "independent_human_accuracy_verified": False,
    }), flush=True)


if __name__ == "__main__":
    run()
