#!/usr/bin/env python3
"""Research-only French->English dialogue adaptation with protected test dialogues.

Multi3WOZ has CULTURALLY ADAPTED French rewrites, NOT guaranteed literal
translations. Conservatively filter obvious mismatches; no deployment or
independent semantic-adequacy claim is permitted from this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "experiments/cpu_continuation_20261117"),
    str(REPO / "experiments/opus_books_20261117"),
]
import train as base_train  # noqa: E402
from train_books import evaluate_frozen_test  # noqa: E402

SOURCE_COMMIT = "c65e80fe120704125255de5cf582a51ebaa285cd"
SOURCE_ZIP_SHA256 = "da238995eed56ed76551c19d0d3fc58e039b62197514ca0d7b705880664ca65e"
WARM_ADAPTER_SHA256 = "072f7f5d2d90b26991c5c62bf8ddb40f686f65e66be52717e7b121d507008ad9"
WARM_ADAPTER_REL = "experiments/massive_stage3_20261117/saved/french/adapter_selected.safetensors"
SEED = 2026111831
UPDATES = 96
TRAIN_COUNT = UPDATES * 16
DEV_COUNT = 128
TEST_COUNT = 256


def normalized(s: str) -> str:
    return " ".join(s.casefold().strip().split())


def eligible_pair(turn: dict) -> tuple[str, str] | None:
    """Conservative surface checks, not proof of semantic equivalence."""
    source, english = turn.get("text"), turn.get("en_text")
    if not isinstance(source, str) or not isinstance(english, str):
        return None
    source, english = source.strip(), english.strip()
    sw, tw = len(source.split()), len(english.split())
    if not (5 <= sw <= 36 and 5 <= tw <= 40):
        return None
    if not (0.75 <= sw / tw <= 1.35):
        return None
    if not (0.55 <= len(source) / max(len(english), 1) <= 1.65):
        return None
    if normalized(source) == normalized(english):
        return None
    # Preserve simple numeric facts; different amounts/times are not translations.
    nums_src = sorted(re.findall(r"\d+(?:[.,]\d+)?", source))
    nums_en = sorted(re.findall(r"\d+(?:[.,]\d+)?", english))
    if nums_src != nums_en:
        return None
    if turn.get("intent_with_errors"):
        return None
    if turn.get("speaker") not in ("USER", "SYSTEM"):
        return None
    if not isinstance(turn.get("turn_id"), int):
        return None
    return (source, english)


def read_dialogues(source_zip: Path):
    """Only inspect official speaker-dialogue records, never audio or external PII."""
    if not source_zip.is_file() or hashlib.sha256(source_zip.read_bytes()).hexdigest() != SOURCE_ZIP_SHA256:
        raise ValueError("Original Multi3WOZ archive SHA-256 mismatch")
    with zipfile.ZipFile(source_zip) as z:
        french = json.loads(z.read("data/French/data.json"))
        valid = {
            line.strip() for line in z.read("data/French/valListFile.txt").decode().splitlines()
            if line.strip()
        }
        test = {
            line.strip() for line in z.read("data/French/testListFile.txt").decode().splitlines()
            if line.strip()
        }
    if not isinstance(french, dict) or not valid or not test:
        raise ValueError("Official dialogue partitions are absent")
    aliases = {key.casefold(): key for key in french}
    if len(aliases) != len(french):
        raise ValueError("Ambiguous dialogue IDs differing only by case")
    def canonical_ids(original: set[str], kind: str):
        converted, missing = set(), []
        for value in original:
            filename = Path(value).name
            if not filename.casefold().endswith(".json"):
                filename += ".json"
            chosen = aliases.get(filename.casefold())
            if chosen is None:
                missing.append(value)
            else:
                converted.add(chosen)
        if missing:
            print("OFFICIAL_PARTITION_ID_MISMATCH", kind,
                  "missing", len(missing), "examples", sorted(missing)[:10],
                  "available_ids_examples", sorted(french)[:5], flush=True)
            raise ValueError("Official " + kind + " IDs do not match source dialogues")
        return converted
    valid = canonical_ids(valid, "validation")
    test = canonical_ids(test, "test")
    if valid & test:
        raise ValueError("Official dialogue partitions overlap")
    return french, valid, test


def choose_dialogue_cohorts(dialogues, official_valid: set[str], official_test: set[str],
                            *, train_n=TRAIN_COUNT, dev_n=DEV_COUNT,
                            test_n=TEST_COUNT, seed=SEED):
    """Disjoint dialogue IDs and exact text; official test NEVER enters training."""
    identifiers = set(dialogues)
    if (not official_valid <= identifiers or not official_test <= identifiers
            or official_valid & official_test):
        raise ValueError("Missing/overlapping official dialogue IDs")
    train_ids = identifiers - official_valid - official_test
    all_parts = {"test": official_test, "dev": official_valid, "train": train_ids}
    used_source, used_target, rows, chosen_metadata = set(), set(), {}, {}
    for partition, size in (("test", test_n), ("dev", dev_n), ("train", train_n)):
        ids = sorted(all_parts[partition],
                     key=lambda key: hashlib.sha256(f"{seed}|{partition}|{key}".encode()).hexdigest())
        kept, used_ids, samples = [], set(), 0
        for dialogue_id in ids:
            conversation = dialogues[dialogue_id]
            if not isinstance(conversation, dict):
                continue
            turns = conversation.get("log", [])
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                aligned = eligible_pair(turn)
                if aligned is None:
                    continue
                src, tgt = aligned
                a, b = normalized(src), normalized(tgt)
                if a in used_source or b in used_target:
                    continue
                used_source.add(a)
                used_target.add(b)
                used_ids.add(dialogue_id)
                kept.append(aligned)
                samples += 1
                if len(kept) >= size:
                    break
            if len(kept) >= size:
                break
        if len(kept) < size:
            raise ValueError(f"Only {len(kept)} filtered {partition} pairs, expected {size}")
        rows[partition] = kept
        chosen_metadata[partition] = {
            "n_pairs": len(kept),
            "n_dialogues": len(used_ids),
            "dialogue_ids_sha256": hashlib.sha256(
                "\n".join(sorted(used_ids)).encode()).hexdigest(),
            "pair_digests_sha256": hashlib.sha256(
                "\n".join(hashlib.sha256((src+"\x00"+en).encode()).hexdigest()
                          for src,en in kept).encode()).hexdigest(),
        }
    assert all(not (set(official_valid) & train_ids or set(official_test) & train_ids)
               for _ in (0,))
    return rows, chosen_metadata


def experiment(source_zip: Path, output_dir: Path):
    if output_dir.exists():
        raise ValueError("Refusing to overwrite research experiment output")
    dialogues, valid, test = read_dialogues(source_zip)
    rows, groups = choose_dialogue_cohorts(dialogues, valid, test)
    del dialogues
    print("MULTI3WOZ_SPLITS", json.dumps(groups), flush=True)
    warm = REPO / WARM_ADAPTER_REL
    if hashlib.sha256(warm.read_bytes()).hexdigest() != WARM_ADAPTER_SHA256:
        raise RuntimeError("Stage-three MASSIVE warm model SHA-256 mismatch")

    with tempfile.TemporaryDirectory(prefix="bridgecaption-multi3woz-") as work:
        data = Path(work) / "train-and-dev-only.json"
        data.write_text(json.dumps({"train":rows["train"],"dev":rows["dev"]},
                                   ensure_ascii=False),encoding="utf-8")
        spec = base_train.LANGUAGES["french"].copy()
        base_train.LANGUAGES["french"] = {
            **spec, "data_url": data.as_uri(), "archive_name":"Multi3WOZ-dialogue-fr",
            "warm_adapter_override": {
                "path":WARM_ADAPTER_REL, "sha256":WARM_ADAPTER_SHA256,
            },
        }
        base_train.SEED=SEED
        base_train.TOTAL_UPDATES=UPDATES
        base_train.TRAIN_COUNT=TRAIN_COUNT
        base_train.DEV_COUNT=DEV_COUNT
        base_train.CHECKPOINTS=(24,48,72,96)

        def parse(data_bytes: bytes, *, archive_name: str):
            if archive_name != "Multi3WOZ-dialogue-fr":
                raise RuntimeError("Wrong corpus source")
            doc=json.loads(data_bytes)
            if (len(doc["train"]),len(doc["dev"])) != (TRAIN_COUNT,DEV_COUNT):
                raise RuntimeError("Prepared cohort differs from frozen manifest")
            return doc["train"],doc["dev"],TRAIN_COUNT+DEV_COUNT

        base_train.parse_pairs=parse
        report=base_train.train_experiment(REPO,output_dir,"french")
        report.update({
            "data_source":"cambridgeltl/multi3woz",
            "source_commit":SOURCE_COMMIT,
            "source_zip_sha256":SOURCE_ZIP_SHA256,
            "warm_adapter_sha256":WARM_ADAPTER_SHA256,
            "split_source":"official Multi3WOZ train/validation/test dialogues",
            "selected_pair_groups":groups,
            "license":"Source repository MIT, paper CC BY 4.0; credit original authors",
            "not_proven_direct_translations":True,
            "limit":"French turns are culturally adapted; length/number filter "
                     "does NOT certify semantic equivalence or 90% adequacy.",
        })
        (output_dir/"report.json").write_text(json.dumps(report,indent=2))
    # Test cohort is sealed off from training and development selection.
    score=evaluate_frozen_test("french",output_dir,rows["test"],
                               report["selected"],previous_adapter=warm)
    result={
        "language":"french", "dataset":"Multi3WOZ human-human conversations",
        "training_updates":UPDATES,
        "train_n":TRAIN_COUNT,"dev_n":DEV_COUNT,"test_n":TEST_COUNT,
        "split_manifest":groups,
        "test_dialogues_separate_from_train_and_dev":True,
        "selected_on_dev_only":report["selected"],
        "frozen_test_scores":score,
        "no_human_semantic_quality_ratings":True,
        "warnings":(
            "Culturally adapted bilingual turns may not be exact translations, "
            "even after filtering. This is a research-only experiment and no "
            "candidate must replace previous best based on this dataset alone. "
            "Never reuse consumed official test dialogues for later tuning."
        ),
    }
    (output_dir/"multi3woz_evaluation.json").write_text(json.dumps(result,indent=2))
    print("MULTI3WOZ_FROZEN_RESULTS",json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    experiment(args.source_zip.resolve(),args.output_dir.resolve())
