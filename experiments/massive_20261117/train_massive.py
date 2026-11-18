#!/usr/bin/env python3
"""French/Spanish->English voice-command LoRA continuation with Amazon MASSIVE.

Uses author-published, predefined train/validation/test splits with ID alignment.
No audio/ASR, no personally identifiable data and no production activation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unicodedata

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "experiments/cpu_continuation_20261117"),
    str(REPO / "experiments/opus_books_20261117"),
]
import train as base_train  # noqa: E402
from train_books import evaluate_frozen_test  # noqa: E402

DATASET = "AmazonScience/massive"
REVISION = "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
LOCALE = {"french": "fr-FR", "spanish": "es-ES"}
SEED = 202611179
UPDATES = 256
TRAIN_SIZE = UPDATES * 16
DEV_SIZE = 128
TEST_SIZE = 256
SPLITS = (("train", "train"), ("validation", "dev"), ("test", "test"))


def canonical(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def select_pairs(english: dict, translated: dict,
                 *, train_size: int = TRAIN_SIZE,
                 dev_size: int = DEV_SIZE, test_size: int = TEST_SIZE,
                 seed: int = SEED, include_ids: bool = False):
    """Select parallel utterances, never moving a test/dev example into training.

    english/translated: {split_name: [MASSIVE row dict, ...]}. Must align on SLURP ID.
    """
    candidates = {}
    for split, _ in SPLITS:
        source_rows, reference_rows = translated.get(split), english.get(split)
        if source_rows is None or reference_rows is None:
            raise ValueError("Missing official MASSIVE data partition")
        en = {str(row["id"]): row for row in reference_rows}
        src = {str(row["id"]): row for row in source_rows}
        if len(en) != len(reference_rows) or len(src) != len(source_rows):
            raise ValueError("Duplicate sample ID inside original MASSIVE partition")
        if set(en) != set(src):
            raise ValueError("Misaligned translated and English MASSIVE IDs")
        pool = []
        for key in sorted(en):
            a, b = src[key], en[key]
            if a.get("intent") != b.get("intent"):
                raise ValueError("Translated and English intent annotations disagree")
            if a.get("partition") not in (split, "dev" if split == "validation" else split):
                raise ValueError("Incorrect target original partition label")
            if b.get("partition") not in (split, "dev" if split == "validation" else split):
                raise ValueError("Incorrect English original partition label")
            source, target = str(a.get("utt", "")).strip(), str(b.get("utt", "")).strip()
            if not (1 <= len(source.split()) <= 35 and 1 <= len(target.split()) <= 36):
                continue
            if not canonical(source) or not canonical(target) or canonical(source) == canonical(target):
                continue
            pool.append((key, source, target))
        random.Random(seed).shuffle(pool)
        candidates[split] = pool

    # Reserve the untouched original test first. Duplicates crossing official splits
    # are excluded from other partitions, never copied into a fitting split.
    selected, seen_src, seen_en, seen_ids = {}, set(), set(), set()
    for split, need in (("test", test_size), ("validation", dev_size), ("train", train_size)):
        keep = []
        for key, source, target in candidates[split]:
            ss, tt = canonical(source), canonical(target)
            if key in seen_ids or ss in seen_src or tt in seen_en:
                continue
            keep.append((key, source, target))
            seen_ids.add(key)
            seen_src.add(ss)
            seen_en.add(tt)
            if len(keep) >= need:
                break
        if len(keep) != need:
            raise ValueError("Not enough separate MASSIVE examples in " + split)
        selected[split] = keep

    def pairs(split):
        return [(src, en) for _, src, en in selected[split]]
    result = {
        "train": pairs("train"), "dev": pairs("validation"), "test": pairs("test"),
        "ids_sha256": {split: hashlib.sha256(
            "\n".join(str(row[0]) for row in selected[split]).encode()).hexdigest()
            for split in selected},
        "original_splits_preserved": True,
    }
    if include_ids:
        result["selected_ids"] = {key: [row[0] for row in selection]
                                  for key, selection in selected.items()}
    return result


def fetch_multipart(locale: str):
    """Load public parquet-backed or official author-script MASSIVE at pinned revision."""
    from datasets import load_dataset
    dataset = load_dataset(DATASET, locale, revision=REVISION, trust_remote_code=True)
    data = {}
    for key, _ in SPLITS:
        if key not in dataset:
            raise RuntimeError("Dataset missing registered split " + key)
        data[key] = [dict(row) for row in dataset[key]]
    return data


def run(language: str, output: Path):
    if language not in LOCALE:
        raise ValueError("Only French and Spanish are supported by these saved adapters")
    if output.exists():
        raise ValueError("Refusing to overwrite existing research output")
    english = fetch_multipart("en-US")
    translated = fetch_multipart(LOCALE[language])
    splits = select_pairs(english, translated)
    del english, translated  # Training only uses text pairs, not worker IDs/ratings.

    with tempfile.TemporaryDirectory(prefix="bridgecaption-massive-") as tmp:
        raw = Path(tmp) / "train-dev-only.json"
        raw.write_text(json.dumps({"train": splits["train"], "dev": splits["dev"]},
                                  ensure_ascii=False), encoding="utf-8")
        old = base_train.LANGUAGES[language].copy()
        base_train.LANGUAGES[language] = {
            **old, "archive_name": "MASSIVE-1.1-" + language,
            "data_url": raw.as_uri()
        }
        base_train.SEED = SEED
        base_train.TOTAL_UPDATES = UPDATES
        base_train.TRAIN_COUNT = TRAIN_SIZE
        base_train.CHECKPOINTS = (64, 128, 192, 256)

        def parse(data: bytes, *, archive_name: str):
            if archive_name != "MASSIVE-1.1-" + language:
                raise RuntimeError("Unexpected corpus selected for MASSIVE experiment")
            decoded = json.loads(data)
            if (len(decoded["train"]), len(decoded["dev"])) != (TRAIN_SIZE, DEV_SIZE):
                raise RuntimeError("Prepared data changed unexpectedly")
            return decoded["train"], decoded["dev"], TRAIN_SIZE + DEV_SIZE

        base_train.parse_pairs = parse
        report = base_train.train_experiment(REPO, output, language)
        report["corpus_source"] = DATASET
        report["corpus_revision"] = REVISION
        report["license"] = "CC BY 4.0 — MASSIVE Amazon Science / SLURP"
        report["split_ids_sha256"] = splits["ids_sha256"]
        report["warning"] = (
            "Official MASSIVE splits and shared SLURP IDs; 90% correctness still "
            "requires independent bilingual human adequacy review. Some localization "
            "differences are intentional rather than literal translations.")
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Frozen test is only opened for inference after dev selection is finalized.
    comparison = evaluate_frozen_test(language, output, splits["test"], report["selected"])
    final = {
        "language": language, "source": DATASET, "revision": REVISION,
        "license": "CC BY 4.0", "data_splits": "official MASSIVE train/dev/test",
        "train_n": TRAIN_SIZE, "dev_n": DEV_SIZE, "test_n": TEST_SIZE,
        "aligned_by_original_slurp_id": True,
        "split_ids_sha256": splits["ids_sha256"],
        "dev_selection_frozen_before_test": True,
        "selected_on_dev": report["selected"], "dev_scores": report["scores"],
        "independent_split_test_scores": comparison,
        "limitations": (
            "This test is held out from our adaptation but could overlap original "
            "Marian pretraining. MASSIVE contains localized, not always literal "
            "utterances. chrF and BLEU do not indicate percent correct translations; "
            "these are text-only utterances, not an end-to-end speech or dialog test."),
    }
    (output / "massive_evaluation.json").write_text(json.dumps(final, indent=2))
    print("MASSIVE_FROZEN_TEST", json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(LOCALE), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arg = parser.parse_args()
    run(arg.language, arg.output_dir.resolve())
