#!/usr/bin/env python3
"""Research-only QED subtitle/lecture-domain LoRA continuation (fr/es -> en).

QED is explicitly licensed for research only by QCRI. Do not integrate its
trained candidates into a commercial or production translation provider.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [
    str(REPO / "experiments/cpu_continuation_20261117"),
    str(REPO / "experiments/opus_books_20261117"),
]
import train as base_train  # noqa: E402
from books import split_books  # noqa: E402
from train_books import evaluate_frozen_test  # noqa: E402

QED = {
    "french": {
        "pair": "en-fr", "source": "fr",
        "url": "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/en-fr.txt.zip",
        "sha256": None,  # First-ever run records a digest, then freeze it.
    },
    "spanish": {
        "pair": "en-es", "source": "es",
        "url": "https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/en-es.txt.zip",
        "sha256": None,
    },
}
SEED = 202611173
TOTAL_UPDATES = 320
TRAIN_COUNT = TOTAL_UPDATES * 16


def norm(s: str) -> str:
    return " ".join(s.casefold().strip().split())


def aligned_subtitles(data: bytes, language: str):
    spec = QED[language]
    digest = hashlib.sha256(data).hexdigest()
    expected = spec["sha256"]
    if expected and digest != expected:
        raise RuntimeError("QED archive changed unexpectedly; aborting")
    prefix = "QED." + spec["pair"]
    wanted = [prefix + "." + extension for extension in (spec["source"], "en", "ids")]
    if len(data) > 150 * 1024 * 1024:
        raise RuntimeError("Unusually large QED source archive")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        lookup = {}
        for info in z.infolist():
            name = Path(info.filename).name
            if name in wanted:
                if name in lookup or info.file_size > 180_000_000:
                    raise RuntimeError("Unexpected QED archive member")
                lookup[name] = info.filename
        if set(wanted) != set(lookup):
            available = sorted(Path(item.filename).name for item in z.infolist())
            raise RuntimeError("QED archive missing source, English, or subtitle video IDs. ZIP filenames: " + repr(available[:20]))
        source, english, metadata = [
            z.read(lookup[name]).decode("utf-8-sig").splitlines() for name in wanted
        ]
    if not (len(source) == len(english) == len(metadata)):
        raise RuntimeError("QED subtitle parallel rows misaligned")
    groups, seen_source, seen_target = {}, set(), set()
    for src, tgt, meta in zip(source, english, metadata):
        src, tgt = src.strip(), tgt.strip()
        if not (3 <= len(src.split()) <= 32 and 3 <= len(tgt.split()) <= 36):
            continue
        if not (0.3 <= len(src) / max(1, len(tgt)) <= 3.0):
            continue
        s, t = norm(src), norm(tgt)
        if s == t or s in seen_source or t in seen_target:
            continue
        ids = meta.split()
        if len(ids) < 2:
            continue
        doc = ids[0] + "|" + ids[1]
        groups.setdefault(doc, []).append((src, tgt))
        seen_source.add(s)
        seen_target.add(t)
    if len(groups) < 10:
        raise RuntimeError("QED source lacks enough independently identifiable videos")
    return groups, digest


def experiment(lang: str, out: Path):
    if out.exists():
        raise ValueError("Refusing to overwrite previous research output")
    if lang not in QED:
        raise ValueError("Invalid language")
    spec = QED[lang]
    source = base_train.LANGUAGES[lang].copy()
    base_train.LANGUAGES[lang] = {
        **source, "data_url": spec["url"], "archive_name": "QED." + spec["pair"],
        "max_dataset_bytes": 150 * 1024 * 1024,
    }
    base_train.SEED = SEED
    base_train.TOTAL_UPDATES = TOTAL_UPDATES
    base_train.TRAIN_COUNT = TRAIN_COUNT
    base_train.CHECKPOINTS = (64, 128, 192, 256, 320)
    split_holder = {}

    def parse(data: bytes, *, archive_name: str):
        if archive_name != "QED." + spec["pair"]:
            raise RuntimeError("Incorrect subtitle corpus")
        grouped, digest = aligned_subtitles(data, lang)
        selected = split_books(grouped, seed=SEED, train_count=TRAIN_COUNT,
                               dev_count=128, test_count=256)
        split_holder.update(selected)
        split_holder["sha256"] = digest
        return selected["train"], selected["dev"], sum(map(len, grouped.values()))

    base_train.parse_pairs = parse
    report = base_train.train_experiment(REPO, out, lang)
    report["attribution"] = (
        "QCRI Educational Domain Corpus (QED v2.0a), Ahmed Abdelali, Paco Guzman, "
        "Hassan Sajjad and Stephan Vogel; distributed by OPUS/Jorg Tiedemann. "
        "RESEARCH USE ONLY (Qatar Computing Research Institute copyright).")
    report["warning"] = (
        "Unreviewed subtitle alignment and possible pretrained model corpus overlap. "
        "Development-only model selection does not establish true translation adequacy.")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    test = evaluate_frozen_test(lang, out, split_holder["test"], report["selected"])
    final = {
        "language": lang,
        "corpus": "QED v2.0a subtitles and educational lectures",
        "corpus_sha256": split_holder["sha256"],
        "sha_pinned_before_run": bool(spec["sha256"]),
        "train_rows": TRAIN_COUNT, "dev_rows": 128, "test_rows": 256,
        "subtitle_document_groups": split_holder["book_groups_total"],
        "heldout_group_hashes": split_holder["doc_group_sha256"],
        "selection_frozen_before_external_test": True,
        "selected_on_dev_only": report["selected"],
        "development_scores": report["scores"],
        "frozen_subtitle_test": test,
        "limitations": (
            "QED is research-only; some aligned subtitle pairs may be noisy. "
            "The test uses separate subtitle/video document IDs within QED "
            "but may be represented in the pretrained OPUS model. "
            "No bilingual human review or evidence of 90% correctness."),
    }
    (out / "qed_evaluation.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("FROZEN QED RESULTS", json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--language", choices=tuple(QED), required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    experiment(args.language, args.output_dir.resolve())
