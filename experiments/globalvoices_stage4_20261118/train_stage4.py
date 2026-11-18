#!/usr/bin/env python3
"""ACTUAL CPU LoRA continuation from saved MASSIVE stage-3 adapters.

Use a NEW OPUS GlobalVoices train/dev sample, excluding EVERY source/reference
used in the earlier GlobalVoices training, development and test cohorts. Cross-
domain evaluation uses a NEW official MASSIVE test sample, excluding ALL
stage-1/2/3 MASSIVE train/dev/test sample IDs and exact source/reference text.

The 90% human-adequacy milestone requires independent bilingual raters;
automatic chrF and BLEU are strictly exploratory metrics.
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
    str(REPO / "experiments/opus_books_20261117"),
    str(REPO / "experiments/massive_20261117"),
    str(REPO / "experiments/massive_stage2_20261117"),
    str(REPO / "experiments/massive_stage3_20261117"),
]
import train as base_train  # noqa: E402
import train_globalvoices as voices  # noqa: E402
import train_massive as massive  # noqa: E402
import train_stage2 as stage2  # noqa: E402
import train_stage3 as stage3  # noqa: E402
from train_books import evaluate_frozen_test  # noqa: E402

SEED = 202611184
UPDATES = 192
TRAIN_COUNT = UPDATES * 16
DEV_COUNT = 128
GV_TEST_COUNT = 256
MASSIVE_TEST_COUNT = 256
CHECKPOINTS = (48, 96, 144, 192)
GV_SHA256 = {
    "french": "7f5018e2d7aca3bed1e427957bb4a9bcae80e397f2eab4cadd1659d54315da5d",
    "spanish": "680eb6975a09e797ae10e9f32590a540752333b9c81477e9e1bb0b5436f72e44",
}
STAGE3_SHA256 = {
    "french": "072f7f5d2d90b26991c5c62bf8ddb40f686f65e66be52717e7b121d507008ad9",
    "spanish": "14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b",
}


def norm(s: str) -> str:
    return " ".join(s.strip().casefold().split())


def cohort_digest(rows) -> str:
    return hashlib.sha256("\n".join(
        hashlib.sha256((s + "\x00" + e).encode("utf-8")).hexdigest()
        for s, e in rows).encode("ascii")).hexdigest()


def choose_new_globalvoices(rows, old_used, mass_test,
                            *, train_n=TRAIN_COUNT, dev_n=DEV_COUNT,
                            test_n=GV_TEST_COUNT, seed=SEED):
    """Strictly distinct source and reference strings, with test chosen FIRST."""
    consumed_src = {norm(s) for s, _ in old_used}
    consumed_en = {norm(e) for _, e in old_used}
    consumed_src.update(norm(s) for s, _ in mass_test)
    consumed_en.update(norm(e) for _, e in mass_test)

    original = list(rows)
    random.Random(seed).shuffle(original)
    cohorts = {}
    for name, needed in (("test", test_n), ("dev", dev_n), ("train", train_n)):
        chosen = []
        for src, en in original:
            a, b = norm(src), norm(en)
            if not a or not b or a in consumed_src or b in consumed_en:
                continue
            chosen.append((src, en))
            consumed_src.add(a)
            consumed_en.add(b)
            if len(chosen) == needed:
                break
        if len(chosen) != needed:
            raise ValueError(f"Insufficient genuinely new {name} GlobalVoices pairs: {len(chosen)}")
        cohorts[name] = chosen
    assert not ({norm(s) for s, _ in cohorts["train"]} &
                {norm(s) for s, _ in old_used})
    for a, b in (("test", "dev"), ("test", "train"), ("dev", "train")):
        assert not {norm(s) for s, _ in cohorts[a]} & {norm(s) for s, _ in cohorts[b]}
        assert not {norm(e) for _, e in cohorts[a]} & {norm(e) for _, e in cohorts[b]}
    return cohorts


def _combine(old1, old2, old3):
    return {
        "selected_ids": {key: sum((round["selected_ids"][key]
                                   for round in (old1, old2, old3)), [])
                         for key in ("train", "validation", "test")},
        "train": sum((round["train"] for round in (old1, old2, old3)), []),
        "dev": sum((round["dev"] for round in (old1, old2, old3)), []),
        "test": sum((round["test"] for round in (old1, old2, old3)), []),
    }


def untouched_massive_test(language: str, *,
                           size=MASSIVE_TEST_COUNT, repo_root: Path = REPO):
    """Audit all prior official MASSIVE cohorts BEFORE selecting new test cases."""
    if language not in STAGE3_SHA256:
        raise ValueError("Unsupported language")
    english = massive.fetch_multipart("en-US")
    translated = massive.fetch_multipart(massive.LOCALE[language])
    first = massive.select_pairs(english, translated, include_ids=True)
    stage1 = json.loads((repo_root / "experiments/massive_20261117/saved" /
                         language / "massive_evaluation.json").read_text(encoding="utf-8"))
    if first["ids_sha256"] != stage1["split_ids_sha256"]:
        raise RuntimeError("Previously consumed stage-one MASSIVE cases changed")
    second = stage2.next_cohorts(
        english, translated, first, seed=stage2.SEED,
        train_n=stage2.TRAIN_COUNT, dev_n=stage2.DEV_COUNT,
        test_n=stage2.TEST_COUNT, include_ids=True)
    stage2_evidence = json.loads((repo_root / "experiments/massive_stage2_20261117/saved" /
                                  language / "stage2_evaluation.json").read_text(encoding="utf-8"))
    if second["new_id_sha256"] != stage2_evidence["second_stage_ids_sha256"]:
        raise RuntimeError("Previously consumed stage-two MASSIVE cases changed")
    _, third, _ = stage3.prepare_new_round(
        english, translated, language, repo_root=repo_root)
    stage3_evidence = json.loads((repo_root / "experiments/massive_stage3_20261117/saved" /
                                  language / "stage3_evaluation.json").read_text(encoding="utf-8"))
    if third["new_id_sha256"] != stage3_evidence["new_ids_sha256"]:
        raise RuntimeError("Previously consumed stage-three MASSIVE cases changed")

    used = _combine(first, second, third)
    new = stage2.next_cohorts(english, translated, used, seed=SEED,
                              train_n=1, dev_n=1, test_n=size, include_ids=True)
    previous_ids = set(used["selected_ids"]["test"] +
                       used["selected_ids"]["train"] +
                       used["selected_ids"]["validation"])
    if previous_ids & set(new["selected_ids"]["test"]):
        raise RuntimeError("New test overlaps previously consumed utterance IDs")
    return new["test"], {
        "official_partition": "test",
        "author_revision": massive.REVISION,
        "n": size,
        "test_id_sha256": new["new_id_sha256"]["test"],
        "all_stage1_stage2_stage3_IDs_excluded": True,
        "all_stage1_stage2_stage3_exact_source_and_english_text_excluded": True,
    }


def train(language: str, output: Path):
    if language not in STAGE3_SHA256:
        raise ValueError("Unsupported language")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an earlier research run")
    checkpoint = REPO / ("experiments/massive_stage3_20261117/saved/" +
                         language + "/adapter_selected.safetensors")
    if base_train.sha256(checkpoint) != STAGE3_SHA256[language]:
        raise RuntimeError("Warm MASSIVE stage-three checkpoint changed")

    mass_test, mass_meta = untouched_massive_test(language)
    spec = voices.GLOBALVOICES[language]
    with urlopen(Request(spec["url"], headers={"User-Agent": "BridgeCaption-research/1.0"}),
                 timeout=90) as stream:
        corpus = stream.read(150 * 1024 * 1024 + 1)
    if len(corpus) > 150 * 1024 * 1024:
        raise RuntimeError("Oversized GlobalVoices corpus")
    rawsha = hashlib.sha256(corpus).hexdigest()
    if rawsha != GV_SHA256[language]:
        raise RuntimeError("GlobalVoices corpus archive changed; no blind reuse")
    voices.GLOBALVOICES[language]["sha256"] = GV_SHA256[language]
    rows, same_sha, eligible = voices.aligned_subtitles(corpus, language)
    del corpus
    old_train, old_dev, old_test = voices.select_cohorts(rows)
    old_used = old_train + old_dev + old_test
    selected = choose_new_globalvoices(rows, old_used, mass_test)
    del rows
    manifests = {key: {"n": len(cohort), "pair_sha256": cohort_digest(cohort)}
                 for key, cohort in selected.items()}
    print("FRESH_GLOBALVOICES", language, "source_sha", same_sha,
          "old_excluded", len(old_used), "new_splits", manifests,
          "MASSIVE_UNSEEN", mass_meta, flush=True)

    # Raw human data exists only in a self-deleting temporary directory.
    with tempfile.TemporaryDirectory(prefix="bridgecaption-gv4-") as temp:
        raw = Path(temp) / "train_dev.json"
        raw.write_text(json.dumps({
            "train": selected["train"], "dev": selected["dev"]},
            ensure_ascii=False), encoding="utf-8")
        original = base_train.LANGUAGES[language].copy()
        base_train.LANGUAGES[language] = {
            **original,
            "data_url": raw.as_uri(), "archive_name": "GLOBALVOICES-STAGE4-" + language,
            "warm_adapter_override": {
                "path": str(checkpoint.relative_to(REPO)),
                "sha256": STAGE3_SHA256[language],
            },
        }
        base_train.SEED = SEED
        base_train.TOTAL_UPDATES = UPDATES
        base_train.TRAIN_COUNT = TRAIN_COUNT
        base_train.DEV_COUNT = DEV_COUNT
        base_train.CHECKPOINTS = CHECKPOINTS

        def parse(payload: bytes, *, archive_name: str):
            if archive_name != "GLOBALVOICES-STAGE4-" + language:
                raise RuntimeError("Unexpected training source")
            record = json.loads(payload)
            if (len(record["train"]), len(record["dev"])) != (TRAIN_COUNT, DEV_COUNT):
                raise RuntimeError("Mutated frozen training/development data")
            return record["train"], record["dev"], TRAIN_COUNT + DEV_COUNT

        base_train.parse_pairs = parse
        report = base_train.train_experiment(REPO, output, language)
        report.update({
            "experiment": "Stage4 fresh GlobalVoices training from saved MASSIVE3 adapter",
            "source": "OPUS-GlobalVoices/v2018q4",
            "source_sha256": rawsha,
            "old_GlobalVoices_train_dev_test_pair_count_excluded": len(old_used),
            "new_GlobalVoices_cohort_hashes": manifests,
            "previous_adapter_sha256": STAGE3_SHA256[language],
            "rights": "Global Voices original article attribution required (usually CC BY)",
            "known_limitations": (
                "TMX preserves translation-unit pairing, not article-group IDs. "
                "Original Marian may contain related text. Strict literal "
                "text separation is not proof of contextual generalization. "
                "No live app promotion; chrF/BLEU are not human correctness."),
        })
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ONLY AFTER frozen dev checkpoint selection: test on unexposed samples.
    gv_score = evaluate_frozen_test(
        language, output, selected["test"], report["selected"],
        previous_adapter=checkpoint)
    massive_score = evaluate_frozen_test(
        language, output, mass_test, report["selected"],
        previous_adapter=checkpoint)
    evidence = {
        "language": language, "selected_on_dev_only": report["selected"],
        "completed_updates": report["completed_updates"],
        "source_archive_sha256": rawsha,
        "training_and_dev": manifests,
        "fresh_heldout_GlobalVoices_test": {
            "manifest": manifests["test"], "scores": gv_score},
        "fresh_heldout_MASSIVE_author_test": {
            "manifest": mass_meta, "scores": massive_score},
        "was_any_previous_GV_or_MASSIVE_test_used_for_stage4_train": False,
        "not_validated_90_percent_human_accuracy": True,
        "limitations": (
            "GlobalVoices TMX lacks article IDs. MASSIVE author test was excluded "
            "from all four local training stages by ID, exact source and reference; "
            "unknown Marian upstream pretraining overlap is still possible. "
            "No real bilingual human judgments of slang/context/meaning. "
            "These test sets are now consumed and cannot tune further rounds."),
    }
    (output / "stage4_evaluation.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8")
    print("STAGE4_FROZEN_RESULTS", json.dumps(evidence), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=tuple(STAGE3_SHA256), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    x = p.parse_args()
    train(x.language, x.output_dir.resolve())


if __name__ == "__main__":
    main()
