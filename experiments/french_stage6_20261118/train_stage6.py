#!/usr/bin/env python3
"""TRAIN another French LoRA round on real, disjoint news + historical train replay.

New dev/test cohorts are chosen before optimization. No prior MASSIVE author
test or old GlobalVoices train/dev/test pair is allowed in the new news training
set. Legitimate rehearsal is from historical MASSIVE TRAIN only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments" / directory) for directory in (
    "cpu_continuation_20261117",
    "globalvoices_20261117",
    "globalvoices_stage4_20261118",
    "balanced_stage5_20261118",
    "massive_20261117",
    "massive_stage2_20261117",
    "massive_stage3_20261117",
    "opus_books_20261117",
)]
import train as base_train
import train_globalvoices as gv
import train_stage4 as stage4
import train_stage5 as stage5
import train_massive as massive
import train_stage2 as stage2
import train_stage3 as stage3
from train_books import evaluate_frozen_test

LANGUAGE = "french"
SEED = 202611186
UPDATES = 192
MASSIVE_REPLAY = 1536
FRESH_NEWS = 1536
MASSIVE_DEV = 128
NEWS_DEV = 128
TEST_N = 256
WARM_SHA = "fba2bd78958bf0bf8a4c14dc55e601e08663633718c4c9c17f91ef9b52dcfa1c"
WARM_REL = "experiments/balanced_stage5_20261118/saved/french/adapter_selected.safetensors"


def digest_pairs(rows):
    return stage4.cohort_digest(rows)


def historical_globalvoices(rows):
    """Reproduce exact stage1/4/5 splits and verify permanently saved hashes."""
    original = sum(gv.select_cohorts(rows), [])
    mass4, _ = stage4.untouched_massive_test(LANGUAGE)
    fourth = stage4.choose_new_globalvoices(rows, original, mass4)
    evidence4 = json.loads(
        (ROOT / "experiments/globalvoices_stage4_20261118/saved/french/"
         "stage4_evaluation.json").read_text())
    for key in ("train", "dev", "test"):
        if (digest_pairs(fourth[key]) !=
                evidence4["training_and_dev"][key]["pair_sha256"]):
            raise RuntimeError("GlobalVoices stage-four cohort provenance mismatch")
    previous = original + sum(fourth.values(), [])
    old_replay, old_mass_dev, old_mass_test, _, old_massive = stage5.select_massive(
        LANGUAGE, previous)
    fifth = stage4.choose_new_globalvoices(
        rows, previous, old_massive + old_replay + old_mass_dev + old_mass_test,
        train_n=stage5.NEW_NEWS_COUNT, dev_n=stage5.GV_DEV,
        test_n=stage5.TEST_COUNT, seed=stage5.SEED)
    evidence5 = json.loads(
        (ROOT / "experiments/balanced_stage5_20261118/saved/french/"
         "stage5_evaluation.json").read_text())
    if (digest_pairs(fifth["test"]) !=
            evidence5["cohorts"]["test"]["GlobalVoices"]["pair_sha256"] or
            digest_pairs(fifth["train"] + []) == ""):
        raise RuntimeError("GlobalVoices stage-five frozen test mismatch")
    old_train = old_replay + fifth["train"]
    random.Random(stage5.SEED).shuffle(old_train)
    old_dev = old_mass_dev + fifth["dev"]
    random.Random(stage5.SEED + 1).shuffle(old_dev)
    if (digest_pairs(old_train) != evidence5["cohorts"]["train"]["sha256"] or
            digest_pairs(old_dev) != evidence5["cohorts"]["dev"]["sha256"]):
        raise RuntimeError("GlobalVoices stage-five previous train/dev manifest changed")
    return previous + sum(fifth.values(), [])


def fresh_massive(previous_gv, *, train_n=MASSIVE_REPLAY,
                  dev_n=MASSIVE_DEV, test_n=TEST_N, seed=SEED):
    english = massive.fetch_multipart("en-US")
    french = massive.fetch_multipart("fr-FR")
    first = massive.select_pairs(english, french, include_ids=True)
    stage1_report = json.loads(
        (ROOT / "experiments/massive_20261117/saved/french/massive_evaluation.json").read_text())
    if first["ids_sha256"] != stage1_report["split_ids_sha256"]:
        raise RuntimeError("Stage one MASSIVE historical split provenance mismatch")
    second = stage2.next_cohorts(
        english, french, first, include_ids=True,
        train_n=stage2.TRAIN_COUNT, dev_n=stage2.DEV_COUNT,
        test_n=stage2.TEST_COUNT, seed=stage2.SEED)
    third_warm, third, _ = stage3.prepare_new_round(english, french, LANGUAGE)
    combined = stage5.combine_massive(first, second, third)
    fourth = stage2.next_cohorts(
        english, french, combined, include_ids=True,
        train_n=1, dev_n=1, test_n=stage4.MASSIVE_TEST_COUNT, seed=stage4.SEED)
    consumed = stage5.combine_massive(first, second, third, fourth)
    fifth = stage2.next_cohorts(
        english, french,
        {**consumed, "train": consumed["train"] + list(previous_gv[:-stage5.TRAIN_COUNT])},
        include_ids=True, train_n=1, dev_n=stage5.MASSIVE_DEV,
        test_n=stage5.TEST_COUNT, seed=stage5.SEED)
    saved5 = json.loads(
        (ROOT / "experiments/balanced_stage5_20261118/saved/french/"
         "stage5_evaluation.json").read_text())
    actual_hashes = fifth["new_id_sha256"]
    if (saved5["cohorts"]["MASSIVE"]["new_test_ids_sha256"] != actual_hashes["test"]
            or saved5["cohorts"]["MASSIVE"]["new_dev_ids_sha256"] !=
               actual_hashes["validation"]):
        raise RuntimeError("Consumed fifth-stage MASSIVE cohort IDs changed")
    old_rounds = stage5.combine_massive(first, second, third, fourth, fifth)
    sixth = stage2.next_cohorts(
        english, french,
        {**old_rounds, "train": old_rounds["train"] + list(previous_gv)},
        include_ids=True, train_n=1, dev_n=dev_n, test_n=test_n, seed=seed)
    historical_train = first["train"] + second["train"] + third["train"]
    replay = stage5.pick_replay(
        historical_train, forbidden=previous_gv + sixth["test"] + sixth["dev"],
        count=train_n, seed=seed)
    if (set(sixth["selected_ids"]["test"]) &
            set(old_rounds["selected_ids"]["test"])):
        raise RuntimeError("Previously consumed official MASSIVE test ID reused")
    meta = {
        "previous_stage5_test_id_sha256": actual_hashes["test"],
        "new_frozen_test_ids_sha256": sixth["new_id_sha256"]["test"],
        "new_development_ids_sha256": sixth["new_id_sha256"]["validation"],
        "n": test_n,
        "no_prior_stage_train_dev_test_id_used_in_test": True,
        "official_publisher_test_split": "MASSIVE test",
    }
    all_prev = old_rounds["train"] + old_rounds["dev"] + old_rounds["test"]
    return replay, sixth["dev"], sixth["test"], meta, all_prev


def build_cohorts():
    spec = gv.GLOBALVOICES[LANGUAGE]
    with urlopen(Request(spec["url"],
                         headers={"User-Agent": "BridgeCaption-training-research"}),
                 timeout=90) as response:
        data = response.read(150 * 1024 * 1024 + 1)
    if len(data) > 150 * 1024 * 1024:
        raise ValueError("GlobalVoices exceeded limit")
    digest = hashlib.sha256(data).hexdigest()
    if digest != stage4.GV_SHA256[LANGUAGE]:
        raise RuntimeError("Unexpected changed GlobalVoices archive")
    gv.GLOBALVOICES[LANGUAGE]["sha256"] = digest
    rows, _, eligible = gv.aligned_subtitles(data, LANGUAGE)
    del data

    older = historical_globalvoices(rows)
    replay, mdev, mtest, mmeta, all_mass_old = fresh_massive(older)
    sixth = stage4.choose_new_globalvoices(
        rows, older, all_mass_old + replay + mdev + mtest,
        train_n=FRESH_NEWS, dev_n=NEWS_DEV, test_n=TEST_N, seed=SEED)
    stage5.assert_no_overlap(
        ("new untouched MASSIVE test", mtest),
        ("new untouched GlobalVoices test", sixth["test"]),
        ("new MASSIVE development", mdev),
        ("new GlobalVoices development", sixth["dev"]),
        ("historical MASSIVE TRAIN rehearsal", replay),
        ("new GlobalVoices training", sixth["train"]))
    training = replay + sixth["train"]
    development = mdev + sixth["dev"]
    random.Random(SEED).shuffle(training)
    random.Random(SEED + 1).shuffle(development)
    metadata = {
        "source": "human-translated GlobalVoices + publisher-partitioned MASSIVE",
        "globalvoices_source_sha256": digest,
        "globalvoices_eligible": eligible,
        "all_earlier_globalvoices_train_dev_test_units_excluded_from_new_news": True,
        "massive": mmeta,
        "train": {"total": len(training), "historical_mas_train_replay": MASSIVE_REPLAY,
                  "new_gv": FRESH_NEWS, "sha256": digest_pairs(training)},
        "dev": {"total": len(development), "massive": MASSIVE_DEV,
                "globalvoices": NEWS_DEV, "sha256": digest_pairs(development)},
        "tests": {
            "massive": {"n": len(mtest), "sha256": digest_pairs(mtest)},
            "globalvoices": {"n": len(sixth["test"]), "sha256": digest_pairs(sixth["test"])},
        },
        "new_groups_source_and_reference_disjoint": True,
    }
    return training, development, sixth["test"], mtest, metadata


def experiment(output: Path):
    if output.exists():
        raise FileExistsError("Never overwrite older experiment weights")
    warm = ROOT / WARM_REL
    if base_train.sha256(warm) != WARM_SHA:
        raise RuntimeError("Previously selected stage-five French adapter changed")
    train_pairs, dev_pairs, gv_test, mas_test, meta = build_cohorts()
    print("STAGE6_FRESH_GROUPS", json.dumps(meta), flush=True)

    with tempfile.TemporaryDirectory(prefix="bridgecaption-stage6-") as directory:
        corpus = Path(directory) / "train-dev.json"
        corpus.write_text(json.dumps({"train": train_pairs, "dev": dev_pairs},
                                     ensure_ascii=False), encoding="utf-8")
        spec = base_train.LANGUAGES[LANGUAGE].copy()
        base_train.LANGUAGES[LANGUAGE] = {
            **spec, "data_url": corpus.as_uri(), "archive_name": "stage6-fr",
            "warm_adapter_override": {"path": WARM_REL, "sha256": WARM_SHA},
        }
        base_train.SEED = SEED
        base_train.TRAIN_COUNT = MASSIVE_REPLAY + FRESH_NEWS
        base_train.DEV_COUNT = MASSIVE_DEV + NEWS_DEV
        base_train.TOTAL_UPDATES = UPDATES
        base_train.CHECKPOINTS = (48, 96, 144, 192)

        def parse(bytes_: bytes, *, archive_name: str):
            if archive_name != "stage6-fr":
                raise ValueError("Unexpected research corpus source")
            rows = json.loads(bytes_)
            if (len(rows["train"]) != MASSIVE_REPLAY + FRESH_NEWS or
                    len(rows["dev"]) != MASSIVE_DEV + NEWS_DEV):
                raise ValueError("Prepared train/development record count changed")
            return rows["train"], rows["dev"], len(rows["train"]) + len(rows["dev"])

        base_train.parse_pairs = parse
        report = base_train.train_experiment(ROOT, output, LANGUAGE)
        report.update({
            "stage": 6, "experiment": "French-only conversational replay + fresh news",
            "warm_adapter_sha256": WARM_SHA, "cohorts": meta,
            "source_credits": ["MASSIVE by AmazonScience, CC BY 4.0",
                               "OPUS GlobalVoices, original article author attribution"],
            "limitations": "No context-conditioned Marian model; no human semantic adequacy ratings",
        })
        (output / "report.json").write_text(json.dumps(report, indent=2))

    # Test is only seen after the trained candidate is chosen using dev.
    scores_gv = evaluate_frozen_test(
        LANGUAGE, output, gv_test, report["selected"], previous_adapter=warm)
    scores_mas = evaluate_frozen_test(
        LANGUAGE, output, mas_test, report["selected"], previous_adapter=warm)
    evaluation = {
        "language": LANGUAGE, "stage": 6,
        "selected_using_dev_only": report["selected"],
        "completed_updates": report["completed_updates"],
        "warm_adapter_sha256": WARM_SHA,
        "cohorts": meta,
        "fresh_GlobalVoices_test": scores_gv,
        "fresh_MASSIVE_official_test": scores_mas,
        "historical_test_not_reused_for_finetuning": True,
        "no_bilingual_human_90_accuracy_claim": True,
        "limitations": "Only text similarity, original Marian pretraining exposure unknown; "
                       "no independent bilingual human judge evidence for slang/context.",
    }
    (output / "stage6_evaluation.json").write_text(json.dumps(evaluation, indent=2))
    print("STAGE6_FROZEN_RESULTS", json.dumps(evaluation), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment(args.output_dir.resolve())
