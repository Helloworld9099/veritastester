#!/usr/bin/env python3
"""Stage five: ACTUALLY train balanced voice-command + news translation LoRA.

Preserve stage-3 conversational performance while gaining news vocabulary.
Use only previously published MASSIVE TRAIN examples for replay. Reject all
consumed train/dev/test texts from earlier GlobalVoices and all MAS test IDs.
New official MASSIVE test and new GlobalVoices test selected before training.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "experiments/cpu_continuation_20261117"),
    str(REPO / "experiments/globalvoices_20261117"),
    str(REPO / "experiments/globalvoices_stage4_20261118"),
    str(REPO / "experiments/opus_books_20261117"),
    str(REPO / "experiments/massive_20261117"),
    str(REPO / "experiments/massive_stage2_20261117"),
    str(REPO / "experiments/massive_stage3_20261117"),
]
import train as base_train  # noqa: E402
import train_globalvoices as voices  # noqa: E402
import train_stage4 as stage4  # noqa: E402
import train_massive as massive  # noqa: E402
import train_stage2 as stage2  # noqa: E402
import train_stage3 as stage3  # noqa: E402
from train_books import evaluate_frozen_test  # noqa: E402

SEED = 202611185
UPDATES = 192
REPLAY_COUNT = 1536
NEW_NEWS_COUNT = 1536
TRAIN_COUNT = REPLAY_COUNT + NEW_NEWS_COUNT
MASSIVE_DEV = 128
GV_DEV = 128
DEV_COUNT = MASSIVE_DEV + GV_DEV
TEST_COUNT = 256
CHECKPOINTS = (48, 96, 144, 192)
SOURCE_SHA = stage4.GV_SHA256
WARM_SHA = stage4.STAGE3_SHA256


def normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def assert_no_overlap(*cohorts):
    """Reject duplicate exact source OR target texts anywhere across groups."""
    used_source, used_target = set(), set()
    for name, pairs in cohorts:
        source = {normalize(s) for s, _ in pairs}
        target = {normalize(en) for _, en in pairs}
        if len(source) != len(pairs) or len(target) != len(pairs):
            raise RuntimeError(f"Repeated source/English target within {name}")
        if used_source & source or used_target & target:
            raise RuntimeError(f"Cross-cohort source/reference collision at {name}")
        used_source.update(source)
        used_target.update(target)


def pick_replay(eligible, forbidden, count=REPLAY_COUNT, seed=SEED):
    """Replay OLD MASSIVE TRAIN only, excluding every GV and future test pair."""
    blocked_src = {normalize(s) for s, _ in forbidden}
    blocked_en = {normalize(en) for _, en in forbidden}
    used_src, used_en, chosen = set(), set(), []
    candidates = list(eligible)
    random.Random(seed).shuffle(candidates)
    for s, en in candidates:
        a, b = normalize(s), normalize(en)
        if not a or not b or a in blocked_src or b in blocked_en:
            continue
        if a in used_src or b in used_en:
            continue
        used_src.add(a)
        used_en.add(b)
        chosen.append((s, en))
        if len(chosen) >= count:
            break
    if len(chosen) != count:
        raise ValueError(f"Insufficient safe MASSIVE train replay examples: {len(chosen)}")
    return chosen


def combine_massive(*rounds):
    return {
        "selected_ids": {
            key: sum((round["selected_ids"][key] for round in rounds), [])
            for key in ("train", "validation", "test")
        },
        "train": sum((round["train"] for round in rounds), []),
        "dev": sum((round["dev"] for round in rounds), []),
        "test": sum((round["test"] for round in rounds), []),
    }


def select_massive(language, gv_old):
    """Reconstruct ALL four consumed MASSIVE cohort splits and reserve new test."""
    en = massive.fetch_multipart("en-US")
    source = massive.fetch_multipart(massive.LOCALE[language])
    m1 = massive.select_pairs(en, source, include_ids=True)
    report1 = json.loads((REPO / "experiments/massive_20261117/saved" /
                          language / "massive_evaluation.json").read_text())
    if m1["ids_sha256"] != report1["split_ids_sha256"]:
        raise RuntimeError("Historical stage-one MASSIVE ID manifest changed")
    m2 = stage2.next_cohorts(
        en, source, m1, seed=stage2.SEED, train_n=stage2.TRAIN_COUNT,
        dev_n=stage2.DEV_COUNT, test_n=stage2.TEST_COUNT, include_ids=True)
    report2 = json.loads((REPO / "experiments/massive_stage2_20261117/saved" /
                          language / "stage2_evaluation.json").read_text())
    if m2["new_id_sha256"] != report2["second_stage_ids_sha256"]:
        raise RuntimeError("Historical stage-two MASSIVE ID manifest changed")
    _, m3, _ = stage3.prepare_new_round(en, source, language)
    report3 = json.loads((REPO / "experiments/massive_stage3_20261117/saved" /
                          language / "stage3_evaluation.json").read_text())
    if m3["new_id_sha256"] != report3["new_ids_sha256"]:
        raise RuntimeError("Historical stage-three MASSIVE ID manifest changed")
    first_three = combine_massive(m1, m2, m3)
    m4 = stage2.next_cohorts(
        en, source, first_three, seed=stage4.SEED,
        train_n=1, dev_n=1, test_n=stage4.MASSIVE_TEST_COUNT, include_ids=True)
    report4 = json.loads((REPO / "experiments/globalvoices_stage4_20261118/saved" /
                          language / "stage4_evaluation.json").read_text())
    if (m4["new_id_sha256"]["test"] !=
            report4["fresh_heldout_MASSIVE_author_test"]["manifest"]["test_id_sha256"]):
        raise RuntimeError("Historical stage-four MASSIVE frozen test ID manifest changed")
    # Also reserve against all earlier GlobalVoices train, dev and test text.
    consumed = combine_massive(m1, m2, m3, m4)
    consumed_with_news = {
        **consumed,
        "train": consumed["train"] + list(gv_old),
    }
    m5 = stage2.next_cohorts(
        en, source, consumed_with_news,
        seed=SEED, train_n=1, dev_n=MASSIVE_DEV,
        test_n=TEST_COUNT, include_ids=True)
    future = m5["test"] + m5["dev"]
    replay = pick_replay(m1["train"] + m2["train"] + m3["train"],
                         forbidden=gv_old + future + m4["test"],
                         count=REPLAY_COUNT)
    manifest = {
        "previous_test_id_hashes": [
            m1["ids_sha256"]["test"], m2["new_id_sha256"]["test"],
            m3["new_id_sha256"]["test"], m4["new_id_sha256"]["test"],
        ],
        "new_test_ids_sha256": m5["new_id_sha256"]["test"],
        "new_dev_ids_sha256": m5["new_id_sha256"]["validation"],
        "author_split": "official MASSIVE test", "source_revision": massive.REVISION,
        "all_previously_evaluated_and_trained_ids_excluded": True,
        "previous_train_rows_replayed_without_reusing_any_test_row": True,
    }
    # Return all historical MASSIVE rows to prevent new news data overlap.
    all_used = consumed["train"] + consumed["dev"] + consumed["test"]
    return replay, m5["dev"], m5["test"], manifest, all_used


def prepare(language: str):
    """All cohort selection completed before loading any model or seeing scores."""
    if language not in SOURCE_SHA:
        raise ValueError("Unsupported French/Spanish research language")
    spec = voices.GLOBALVOICES[language]
    with urlopen(Request(spec["url"],
                         headers={"User-Agent": "BridgeCaption-research/1.0"}),
                 timeout=90) as stream:
        content = stream.read(150 * 1024 * 1024 + 1)
    if len(content) > 150 * 1024 * 1024:
        raise ValueError("Source TMX exceeds secure size bound")
    sha = hashlib.sha256(content).hexdigest()
    if sha != SOURCE_SHA[language]:
        raise RuntimeError("Official historical GlobalVoices archive bytes changed")
    voices.GLOBALVOICES[language]["sha256"] = sha
    rows, _, eligible = voices.aligned_subtitles(content, language)
    del content
    gv1_parts = voices.select_cohorts(rows)
    gv1_old = sum(gv1_parts, [])
    stage4_mass_test, meta4 = stage4.untouched_massive_test(language)
    gv4_parts = stage4.choose_new_globalvoices(
        rows, gv1_old, stage4_mass_test)
    gv4_old = sum(gv4_parts.values(), [])
    stage4_evidence = json.loads(
        (REPO / "experiments/globalvoices_stage4_20261118/saved" /
         language / "stage4_evaluation.json").read_text())
    if (stage4_evidence["source_archive_sha256"] != sha or
            stage4_evidence["fresh_heldout_MASSIVE_author_test"]["manifest"]
            ["test_id_sha256"] != meta4["test_id_sha256"]):
        raise RuntimeError("Cannot reconstruct previously consumed stage-four data")
    for name, selected in gv4_parts.items():
        if stage4_evidence["training_and_dev"][name]["pair_sha256"] != stage4.cohort_digest(selected):
            raise RuntimeError("Previously consumed GlobalVoices cohort hash differs: "+name)

    gv_used = gv1_old + gv4_old
    replay, mass_dev, mass_test, mass_meta, mass_all = select_massive(language, gv_used)
    # Exclude all old MASSIVE training/development/testing text, and the new
    # MAS development/test/replay examples, from ALL new news data.
    gv5 = stage4.choose_new_globalvoices(
        rows, gv_used, mass_all + replay + mass_dev + mass_test,
        train_n=NEW_NEWS_COUNT, dev_n=GV_DEV,
        test_n=TEST_COUNT, seed=SEED)
    assert_no_overlap(
        ("new MASSIVE test", mass_test),
        ("new GlobalVoices test", gv5["test"]),
        ("MASSIVE development", mass_dev),
        ("GlobalVoices development", gv5["dev"]),
        ("MASSIVE old-train replay", replay),
        ("new GlobalVoices train", gv5["train"]),
    )
    train_rows = replay + gv5["train"]
    dev_rows = mass_dev + gv5["dev"]
    random.Random(SEED).shuffle(train_rows)
    random.Random(SEED + 1).shuffle(dev_rows)
    meta = {
        "GlobalVoices_source_sha256": sha,
        "GlobalVoices_original_eligible_pairs": eligible,
        "old_GlobalVoices_round1_count": len(gv1_old),
        "old_GlobalVoices_round4_count": len(gv4_old),
        "MASSIVE": mass_meta,
        "train": {
            "n": len(train_rows), "MASSIVE_replay": REPLAY_COUNT,
            "new_GlobalVoices": NEW_NEWS_COUNT,
            "sha256": stage4.cohort_digest(train_rows),
        },
        "dev": {
            "n": len(dev_rows), "MASSIVE": MASSIVE_DEV, "GlobalVoices": GV_DEV,
            "sha256": stage4.cohort_digest(dev_rows),
        },
        "test": {
            "MASSIVE": {"n": len(mass_test), "pair_sha256": stage4.cohort_digest(mass_test)},
            "GlobalVoices": {"n": len(gv5["test"]),
                             "pair_sha256": stage4.cohort_digest(gv5["test"])},
        },
        "cross_domain_source_and_english_reference_text_disjoint": True,
        "all_previous_MASSIVE_and_GlobalVoices_tests_excluded": True,
        "heldout_selected_before_any_model_scoring": True,
    }
    return train_rows, dev_rows, gv5["test"], mass_test, meta


def run(language: str, output_dir: Path):
    if language not in WARM_SHA:
        raise ValueError("Unsupported language")
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite old experiment")
    warm = REPO / "experiments/massive_stage3_20261117/saved" / language / "adapter_selected.safetensors"
    if base_train.sha256(warm) != WARM_SHA[language]:
        raise RuntimeError("Frozen MASSIVE stage-three warm weights altered")
    train_rows, dev_rows, gv_test, mass_test, meta = prepare(language)
    print("BALANCED_STAGE5_NEW_COHORTS", language, json.dumps(meta), flush=True)
    with tempfile.TemporaryDirectory(prefix="bridgecaption-balanced5-") as temp:
        file = Path(temp) / "train-dev.json"
        file.write_text(json.dumps({"train": train_rows, "dev": dev_rows},
                                   ensure_ascii=False), encoding="utf-8")
        spec = base_train.LANGUAGES[language].copy()
        base_train.LANGUAGES[language] = {
            **spec, "data_url": file.as_uri(),
            "archive_name": "balanced-stage5-" + language,
            "warm_adapter_override": {
                "path": str(warm.relative_to(REPO)), "sha256": WARM_SHA[language],
            },
        }
        base_train.SEED = SEED
        base_train.TRAIN_COUNT = TRAIN_COUNT
        base_train.DEV_COUNT = DEV_COUNT
        base_train.TOTAL_UPDATES = UPDATES
        base_train.CHECKPOINTS = CHECKPOINTS

        def parse(data: bytes, *, archive_name: str):
            if archive_name != "balanced-stage5-" + language:
                raise ValueError("Unexpected source data for stage-five training")
            content = json.loads(data)
            if (len(content["train"]), len(content["dev"])) != (TRAIN_COUNT, DEV_COUNT):
                raise ValueError("Prepared train/dev cohorts changed")
            return content["train"], content["dev"], TRAIN_COUNT + DEV_COUNT

        base_train.parse_pairs = parse
        report = base_train.train_experiment(REPO, output_dir, language)
        report.update({
            "experiment": "Balanced 50-50 old-MASSIVE-TRAIN replay + NEW GlobalVoices",
            "warm_sha256": WARM_SHA[language],
            "cohorts": meta,
            "source_licenses": ["AmazonScience MASSIVE CC BY 4.0",
                                "Global Voices citizen-journalism attribution required"],
            "quality_goal": (
                "Reduce news-only catastrophic forgetting, not claim 90% correctness."
            ),
        })
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

    # Two entirely new heldout cohorts; neither was used for model selection.
    gv_scores = evaluate_frozen_test(
        language, output_dir, gv_test, report["selected"], previous_adapter=warm)
    mass_scores = evaluate_frozen_test(
        language, output_dir, mass_test, report["selected"], previous_adapter=warm)
    evidence = {
        "language": language, "selected_using_combined_dev_only": report["selected"],
        "completed_updates": report["completed_updates"], "warm_sha256": WARM_SHA[language],
        "cohorts": meta,
        "first_and_only_new_GlobalVoices_test_scores": gv_scores,
        "first_and_only_new_MASSIVE_test_scores": mass_scores,
        "test_cases_excluded_from_all_local_finetuning": True,
        "did_not_use_any_previously_evaluated_test_for_training": True,
        "no_independent_bilingual_human_adequacy_judgments": True,
        "limitations": (
            "Reference overlap with Marian's original OPUS pretraining remains "
            "unknown. GlobalVoices does not provide reliable source article IDs. "
            "MASSIVE replay uses past TRAIN examples only, not any test or dev. "
            "ChrF/BLEU are corpus text-similarity diagnostics, not translation "
            "accuracy or fluent bilingual semantic adequacy on slang/context."
        ),
    }
    (output_dir / "stage5_evaluation.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8")
    print("STAGE5_FROZEN_RESULTS", json.dumps(evidence), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(WARM_SHA), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.language, args.output_dir.resolve())
