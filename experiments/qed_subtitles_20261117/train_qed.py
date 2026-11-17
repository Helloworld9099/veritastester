#!/usr/bin/env python3
"""Research-only French/Spanish LoRA training on paired QED TMX subtitle segments.

TMX translation units guarantee source/reference pairing where Moses line splits
were not reliable. Translation unit split is NOT speaker/video-independent.
QED is explicitly RESEARCH USE ONLY; no production promotion.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "experiments/cpu_continuation_20261117"),
    str(REPO / "experiments/opus_books_20261117"),
]
import train as base_train  # noqa: E402
from train_books import evaluate_frozen_test  # noqa: E402

QED = {
    "french": {
        "pair": "en-fr", "source": "fr",
        "url": "https://object.pouta.csc.fi/OPUS-QED/v2.0a/tmx/en-fr.tmx.gz",
        "sha256": None,  # Record checksum on first verified run, then pin it.
    },
    "spanish": {
        "pair": "en-es", "source": "es",
        "url": "https://object.pouta.csc.fi/OPUS-QED/v2.0a/tmx/en-es.tmx.gz",
        "sha256": None,
    },
}
SEED = 202611173
TOTAL_UPDATES = 320
TRAIN_COUNT = TOTAL_UPDATES * 16
DEV_COUNT = 128
TEST_COUNT = 256
RESERVOIR_CAP = 50000
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def norm(s: str) -> str:
    return " ".join(s.casefold().strip().split())


def local_tag(element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _tuv_contents(unit, source_language):
    """Only accept a bilingual unit with one source and one English segment."""
    values = {}
    for tuv in unit:
        if local_tag(tuv) != "tuv":
            continue
        language = (tuv.attrib.get(XML_LANG) or tuv.attrib.get("lang") or "").split("-")[0].lower()
        if language not in (source_language, "en") or language in values:
            return None
        segment = next((child for child in tuv if local_tag(child) == "seg"), None)
        if segment is None:
            return None
        values[language] = "".join(segment.itertext()).strip()
    if not values.get(source_language) or not values.get("en"):
        return None
    source, target = values[source_language], values["en"]
    if not (3 <= len(source.split()) <= 32 and 3 <= len(target.split()) <= 36):
        return None
    if not (0.3 <= len(source) / max(1, len(target)) <= 3.0):
        return None
    if norm(source) == norm(target):
        return None
    return source, target


def aligned_subtitles(data: bytes, language: str, *, reservoir_cap=RESERVOIR_CAP):
    """Return bounded random sample of actual paired TMX translation units."""
    if language not in QED:
        raise ValueError("Unknown language")
    spec = QED[language]
    digest = hashlib.sha256(data).hexdigest()
    if spec["sha256"] and digest != spec["sha256"]:
        raise RuntimeError("QED source archive changed since recorded checksum")
    if len(data) > 150 * 1024 * 1024:
        raise RuntimeError("QED archive exceeds budget")
    if not data.startswith(b"\x1f\x8b"):
        raise RuntimeError("Expected a compressed QED TMX archive")
    rng = random.Random(SEED)
    sampled = []
    eligible = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as source:
            iterator = ET.iterparse(source, events=("start", "end"))
            _, root = next(iterator)
            for event, elem in iterator:
                if event != "end" or local_tag(elem) != "tu":
                    continue
                pair = _tuv_contents(elem, spec["source"])
                if pair is not None:
                    eligible += 1
                    if len(sampled) < reservoir_cap:
                        sampled.append(pair)
                    else:
                        idx = rng.randrange(eligible)
                        if idx < reservoir_cap:
                            sampled[idx] = pair
                elem.clear()
                root.clear()
    except (ET.ParseError, OSError, EOFError) as exc:
        raise RuntimeError("Invalid or truncated QED TMX dataset") from exc
    seen_src, seen_en, unique = set(), set(), []
    for src, en in sampled:
        s, t = norm(src), norm(en)
        if s in seen_src or t in seen_en:
            continue
        unique.append((src, en))
        seen_src.add(s)
        seen_en.add(t)
    print("QED_TMX_SHA256", language, digest, "eligible", eligible,
          "sampled_unique", len(unique), flush=True)
    return unique, digest, eligible


def select_cohorts(rows, seed=SEED, train_count=TRAIN_COUNT,
                   dev_count=DEV_COUNT, test_count=TEST_COUNT):
    """Exact-source/reference-disjoint, NOT video-independent cohorts."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if len(rows) < train_count + dev_count + test_count:
        raise ValueError("Too few distinct translation units")
    train = rows[:train_count]
    dev = rows[train_count:train_count + dev_count]
    test = rows[train_count + dev_count:train_count + dev_count + test_count]
    for column in (0, 1):
        sets = [{norm(row[column]) for row in cohort} for cohort in (train, dev, test)]
        if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise RuntimeError("Exact text duplicated across train/dev/test")
    return train, dev, test


def experiment(lang: str, out: Path):
    if out.exists():
        raise ValueError("Research output directory must not exist; never overwrite")
    if lang not in QED:
        raise ValueError("Unknown language")
    spec = QED[lang]
    original = base_train.LANGUAGES[lang].copy()
    base_train.LANGUAGES[lang] = {
        **original, "data_url": spec["url"], "archive_name": "QED-TMX-" + lang,
        "max_dataset_bytes": 150 * 1024 * 1024,
    }
    base_train.SEED = SEED
    base_train.TOTAL_UPDATES = TOTAL_UPDATES
    base_train.TRAIN_COUNT = TRAIN_COUNT
    base_train.CHECKPOINTS = (64, 128, 192, 256, 320)
    holder = {}

    def parse(data: bytes, *, archive_name: str):
        if archive_name != "QED-TMX-" + lang:
            raise RuntimeError("Unexpected QED source archive")
        rows, checksum, eligible = aligned_subtitles(data, lang)
        train, dev, test = select_cohorts(rows)
        holder.update(test=test, sha256=checksum, eligible=eligible,
                      unique=len(rows))
        return train, dev, len(rows)

    base_train.parse_pairs = parse
    report = base_train.train_experiment(REPO, out, lang)
    report["attribution"] = (
        "QCRI Educational Domain Corpus (QED v2.0a), "
        "Abdelali, Guzman, Sajjad, Vogel; OPUS/Tiedemann. Research use ONLY.")
    report["warning"] = (
        "TMX has paired bilingual segments but no trustworthy video-group IDs; "
        "sample-level exact disjointness only. Pretraining contamination unknown. "
        "Neither chrF/BLEU nor reference exact-match equals human adequacy.")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Only now open the reserved evaluation cohort: model selection is frozen.
    scores = evaluate_frozen_test(lang, out, holder["test"], report["selected"])
    summary = {
        "language": lang, "corpus": "QED v2.0a TMX educational subtitles",
        "corpus_sha256": holder["sha256"], "source_pre_pinned": bool(spec["sha256"]),
        "eligible_translation_units": holder["eligible"],
        "sampled_unique_units": holder["unique"],
        "train_rows": TRAIN_COUNT, "dev_rows": DEV_COUNT, "test_rows": TEST_COUNT,
        "cohorts_exact_text_disjoint": True,
        "cohorts_video_document_disjoint": False,
        "selection_frozen_before_test": True,
        "selected_on_dev_only": report["selected"],
        "development_scores": report["scores"], "heldout_tmx_test": scores,
        "limitations": (
            "Research-only dataset. Held-out translation units may be from the "
            "same video or present in pretrained OPUS. No blinded bilingual "
            "judgments, no 90% accuracy verification, no production integration."),
    }
    (out / "qed_evaluation.json").write_text(json.dumps(summary, indent=2))
    print("QED_FROZEN_RESULT", json.dumps(summary), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--language", choices=tuple(QED), required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    experiment(args.language, args.output_dir.resolve())


if __name__ == "__main__":
    main()
