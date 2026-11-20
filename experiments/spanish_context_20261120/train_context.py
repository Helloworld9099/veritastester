#!/usr/bin/env python3
"""Spanish->English contextual LoRA continuation on real, time-aligned travel dialogue.

Noncommercial research ONLY (Samuel & Audrey CC BY-NC 4.0). Never activate
these weights in BridgeCaption's commercial/live app. Video-disjoint test is
selected before optimization and scored once, after development-only selection.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import random
import re
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "experiments/cpu_continuation_20261117"),
    str(ROOT / "experiments/balanced_stage5_20261118"),
    str(ROOT / "experiments/massive_20261117"),
    str(ROOT / "experiments/opus_books_20261117"),
]
import train as core
import train_stage5 as balanced
import train_massive as massive
from train_books import evaluate_frozen_test

SOURCE_REPO = "samuelandaudreymedianetwork/samuel-y-audrey-youtube-transcripts-es-en"
SOURCE_REV = "48a47d4af401b44624975d8b5e940abd70879937"
PREFIX = "samuel-y-audrey-youtube-transcripts-es-en"
# Git blob SHA-1 and byte lengths are pinned from the publisher's revision.
SOURCE_PARTS = (
    (PREFIX + ".part-01.jsonl.gz", "ad7345d16f37631e00a6c8ff7f4c318a57ed1dbe", 19902245),
    (PREFIX + ".part-02.jsonl.gz", "72c49c690dcf5d7f590b9b5e846d84a8e256d1bc", 12903665),
)
SEED = 2026112006
LANGUAGE = "spanish"
UPDATES = 240
CHECKPOINTS = (60, 120, 180, 240)
TRAIN_DIALOGUE = 3072
TRAIN_REPLAY = 768
DEV_N = 256
HELDOUT_N = 256
WARM_REL = "experiments/massive_stage3_20261117/saved/spanish/adapter_selected.safetensors"
WARM_SHA = "14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b"
TIMESTAMP = re.compile(r"(?:(?:\d{1,2}):)?\d{2}:\d{2}[,.]\d{3}")
TAGS = re.compile(r"<[^>]*>")
SPACES = re.compile(r"\s+")


def norm(value: str) -> str:
    return SPACES.sub(" ", value.strip().casefold())


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_pairs(rows) -> str:
    return sha_text("\n".join(sha_text(a + "\0" + b) for a, b in rows))


def srt_seconds(token: str) -> float:
    bits = token.replace(",", ".").split(":")
    if len(bits) == 2:
        minutes, seconds = bits
        return int(minutes) * 60 + float(seconds)
    if len(bits) == 3:
        hours, minutes, seconds = bits
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError("Unsupported SRT timestamp")


def parse_srt(payload: str) -> list[dict]:
    """Parse original timed subtitles, not an inferred translation alignment."""
    if not isinstance(payload, str) or not payload or len(payload) > 4_000_000:
        return []
    srt = payload.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    segments = []
    for block in re.split(r"\n\s*\n", srt):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        stamp = next((j for j, line in enumerate(lines) if "-->" in line), None)
        if stamp is None or stamp + 1 >= len(lines):
            continue
        times = TIMESTAMP.findall(lines[stamp])
        if len(times) < 2:
            continue
        start, end = srt_seconds(times[0]), srt_seconds(times[1])
        if not (0 <= start < end <= 86400 and 0.25 <= end - start <= 20):
            continue
        surface = html.unescape(TAGS.sub("", " ".join(lines[stamp + 1:])))
        surface = SPACES.sub(" ", surface).strip()
        if (not surface or len(surface) > 270 or
                surface.startswith(("[", "(", "♪")) or
                len(surface.split()) < 2 or
                not any(ch.isalpha() for ch in surface)):
            continue
        if segments and norm(segments[-1]["text"]) == norm(surface) and start - segments[-1]["end"] < 6:
            continue
        segments.append({"start": start, "end": end, "text": surface})
    return sorted(segments, key=lambda row: (row["start"], row["end"]))


def align_video(video_id: str, spanish: list[dict], english: list[dict]) -> list[dict]:
    """Conservative exclusive 1:1 temporal alignment; discard uncertain cues.

    Cue times provide candidate matches, NOT proof that the sentences mean
    the same thing. Frozen human review is still required for 95% adequacy.
    """
    if not video_id or not spanish or not english:
        return []
    rows = []
    cursor = 0
    last_used = -1
    prior_source = []
    for es in spanish:
        while cursor < len(english) and english[cursor]["end"] < es["start"] - 1.2:
            cursor += 1
        best = None
        for j in range(max(cursor, last_used + 1), min(cursor + 8, len(english))):
            en = english[j]
            if en["start"] > es["end"] + 1.2:
                break
            overlap = max(0.0, min(es["end"], en["end"]) - max(es["start"], en["start"]))
            if (overlap / (es["end"] - es["start"]) < 0.72 or
                    overlap / (en["end"] - en["start"]) < 0.72 or
                    abs(es["start"] - en["start"]) > 1.2 or
                    abs(es["end"] - en["end"]) > 1.2):
                continue
            source, target = es["text"], en["text"]
            if (not 3 <= len(source.split()) <= 36 or
                    not 3 <= len(target.split()) <= 45 or
                    not 0.45 <= len(source) / len(target) <= 2.3 or
                    norm(source) == norm(target)):
                continue
            goodness = overlap / max(es["end"] - es["start"], en["end"] - en["start"])
            if best is None or goodness > best[0]:
                best = (goodness, j, target)
        if best is not None:
            _, chosen_index, target = best
            recent = [text[:75] for text in prior_source[-2:]]
            suffix = " | contexto previo: " + " / ".join(recent) if recent else ""
            rows.append({
                "video_id": video_id, "source": es["text"],
                "reference": target, "input_with_context": es["text"] + suffix,
                "prior_turns": len(recent), "cue_start": es["start"],
            })
            last_used = chosen_index
        # Only original prior SPANISH transcript text. No reference, gold
        # English, future subtitle or cross-video history enters the input.
        prior_source.append(es["text"])
    return rows


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def download_records() -> tuple[list[dict], dict]:
    records, checksums = [], {}
    for name, expected_sha, expected_size in SOURCE_PARTS:
        url = "https://raw.githubusercontent.com/" + SOURCE_REPO + "/" + SOURCE_REV + "/" + name
        with urlopen(Request(url, headers={"User-Agent": "BridgeCaption-noncommercial-research/1.0"}),
                     timeout=90) as response:
            raw = response.read(expected_size + 1)
        if len(raw) != expected_size or git_blob_sha(raw) != expected_sha:
            raise RuntimeError("Pinned transcript source changed: " + name)
        decompressed = gzip.decompress(raw)
        if len(decompressed) > 90_000_000:
            raise RuntimeError("Unexpectedly large decoded transcript part")
        for line in decompressed.splitlines():
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Source record is not an object")
                records.append(record)
        checksums[name] = {"bytes": len(raw), "git_blob_sha1": expected_sha,
                           "sha256": hashlib.sha256(raw).hexdigest()}
    if len(records) != 643:
        raise RuntimeError("Unexpected number of source video records: " + str(len(records)))
    ids = [r.get("video_id") for r in records]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate or missing source video IDs")
    return records, checksums


def group_split(video_id: str) -> str:
    bucket = int(sha_text(str(SEED) + "|" + video_id)[:8], 16) % 100
    return "train" if bucket < 82 else ("dev" if bucket < 92 else "test")


def select_balanced(rows: list[dict], size: int, *, split: str, forbidden=()) -> list[dict]:
    """Distinct video groups first, unique Spanish and English surface across picks."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["video_id"], []).append(row)
    ordered = sorted(groups, key=lambda g: sha_text(str(SEED) + split + g))
    for vid in ordered:
        groups[vid].sort(key=lambda r: sha_text(str(SEED) + str(r["cue_start"]) +
                                                     r["source"] + r["reference"]))
    chosen, used_src, used_ref = [], set(), set()
    blocked_src = {norm(r["source"]) for r in forbidden}
    blocked_ref = {norm(r["reference"]) for r in forbidden}
    for index in range(max(map(len, groups.values()), default=0)):
        for vid in ordered:
            if index >= len(groups[vid]):
                continue
            row = groups[vid][index]
            s, e = norm(row["source"]), norm(row["reference"])
            if s in used_src or e in used_ref or s in blocked_src or e in blocked_ref:
                continue
            used_src.add(s)
            used_ref.add(e)
            chosen.append(row)
            if len(chosen) == size:
                return chosen
    raise ValueError(f"Insufficient exclusive {split} examples: {len(chosen)}/{size}")


def assert_video_and_text_separation(*cohorts) -> None:
    used_video, used_es, used_en = set(), set(), set()
    for label, rows in cohorts:
        videos = {row["video_id"] for row in rows}
        if used_video.intersection(videos):
            raise RuntimeError("Video group leaked into " + label)
        for row in rows:
            es, en = norm(row["source"]), norm(row["reference"])
            if es in used_es or en in used_en:
                raise RuntimeError("Exact source/reference overlap in " + label)
            used_es.add(es)
            used_en.add(en)
        used_video.update(videos)


def make_cohorts():
    records, sources = download_records()
    all_rows = []
    stats = {"videos_total": len(records), "videos_with_matching_cues": 0,
             "Spanish_cues": 0, "English_cues": 0, "aligned_candidates": 0}
    for entry in records:
        es, en = parse_srt(entry.get("srt_es")), parse_srt(entry.get("srt_en"))
        stats["Spanish_cues"] += len(es)
        stats["English_cues"] += len(en)
        aligned = align_video(entry["video_id"], es, en)
        if aligned:
            stats["videos_with_matching_cues"] += 1
            stats["aligned_candidates"] += len(aligned)
            all_rows.extend(aligned)
    print("SPANISH_ALIGNMENT_AUDIT", json.dumps(stats), flush=True)
    if stats["videos_with_matching_cues"] < 120 or stats["aligned_candidates"] < 5000:
        raise RuntimeError("Insufficient trustworthy SRT alignment for training: " + json.dumps(stats))
    split_rows = {key: [r for r in all_rows if group_split(r["video_id"]) == key]
                  for key in ("train", "dev", "test")}
    # Select ALL splits before a model can access even development references.
    test = select_balanced(split_rows["test"], HELDOUT_N, split="test")
    dev = select_balanced(split_rows["dev"], DEV_N, split="dev", forbidden=test)
    train = select_balanced(split_rows["train"], TRAIN_DIALOGUE, split="train", forbidden=test+dev)
    assert_video_and_text_separation(("heldout", test), ("development", dev), ("train", train))

    en_us = massive.fetch_multipart("en-US")
    es_es = massive.fetch_multipart("es-ES")
    first = massive.select_pairs(en_us, es_es, include_ids=True)
    prior = json.loads((ROOT / "experiments/massive_20261117/saved/spanish/"
                        "massive_evaluation.json").read_text())
    if first["ids_sha256"] != prior["split_ids_sha256"]:
        raise RuntimeError("Historical MASSIVE split changed")
    all_chosen = [(r["source"], r["reference"]) for r in test + dev + train]
    rehearsal = balanced.pick_replay(
        first["train"],
        forbidden=all_chosen + first["dev"] + first["test"],
        count=TRAIN_REPLAY, seed=SEED,
    )
    contextual_train = [
        (row["source"] if i % 4 == 0 else row["input_with_context"], row["reference"])
        for i, row in enumerate(train)
    ]
    train_pairs = contextual_train + rehearsal
    dev_context = [(r["input_with_context"], r["reference"]) for r in dev]
    dev_plain = [(r["source"], r["reference"]) for r in dev]
    test_context = [(r["input_with_context"], r["reference"]) for r in test]
    test_plain = [(r["source"], r["reference"]) for r in test]
    balanced.assert_no_overlap(
        ("frozen test", test_plain), ("development", dev_plain),
        ("new conversational training", [(r["source"], r["reference"]) for r in train]),
        ("historic MASSIVE train-only rehearsal", rehearsal),
    )
    random.Random(SEED).shuffle(train_pairs)
    group_counts = {label: len({r["video_id"] for r in rows})
                    for label, rows in (("train", train), ("dev", dev), ("test", test))}
    if group_counts["test"] < 10 or group_counts["dev"] < 10 or group_counts["train"] < 40:
        raise RuntimeError("Too few independent video groups per split")
    manifest = {
        "dataset": "Samuel y Audrey bilingual creator-authored travel subtitles ES/EN",
        "license": "CC BY-NC 4.0 -- NONCOMMERCIAL research ONLY",
        "source_revision": SOURCE_REV, "source_files": sources, "counts": stats,
        "selected_rows": {"train_conversation": len(train), "prior_MASSIVE_TRAIN_replay": len(rehearsal),
                          "dev": len(dev), "frozen_test": len(test)},
        "disjoint_video_group_counts": group_counts,
        "train_pair_sha256": sha_pairs(train_pairs),
        "dev_pair_sha256": sha_pairs(dev_plain),
        "frozen_test_pair_sha256": sha_pairs(test_plain),
        "contextual_train_rows": sum("| contexto previo:" in s for s, _ in contextual_train),
        "contextual_dev_rows": sum(r["prior_turns"] > 0 for r in dev),
        "contextual_test_rows": sum(r["prior_turns"] > 0 for r in test),
        "selection_frozen_before_training": True,
        "no_cross_video_or_exact_source_and_target_overlap": True,
        "no_future_turn_or_English_reference_in_context": True,
        "upstream_Marian_pretraining_overlap_unverifiable": True,
        "warning": "Timestamp proximity is not verified translation equivalence; "
                   "publisher caption quality may vary; bilingual human review required.",
    }
    return train_pairs, dev_context, dev_plain, test_context, test_plain, manifest


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError("Research experiments must never overwrite earlier checkpoints")
    if core.sha256(ROOT / WARM_REL) != WARM_SHA:
        raise RuntimeError("Protected earlier Spanish stage-three model hash changed")
    train, dev, dev_plain, test, test_plain, manifest = make_cohorts()
    if (len(train), len(dev), len(test)) != (UPDATES * 16, DEV_N, HELDOUT_N):
        raise RuntimeError("Frozen training/development/test counts changed")
    print("SPANISH_FROZEN_COHORTS_BEFORE_TRAIN", json.dumps(manifest), flush=True)
    with tempfile.TemporaryDirectory(prefix="bridgecaption-spanish-context-") as scratch:
        prepared = Path(scratch) / "train-dev.json"
        prepared.write_text(json.dumps({"train": train, "dev": dev}, ensure_ascii=False),
                            encoding="utf-8")
        spec = core.LANGUAGES[LANGUAGE].copy()
        core.LANGUAGES[LANGUAGE] = {
            **spec, "data_url": prepared.as_uri(),
            "archive_name": "es-conversation-timestamp-aligned",
            "warm_adapter_override": {"path": WARM_REL, "sha256": WARM_SHA},
        }
        core.SEED = SEED
        core.TRAIN_COUNT = len(train)
        core.DEV_COUNT = DEV_N
        core.TOTAL_UPDATES = UPDATES
        core.CHECKPOINTS = CHECKPOINTS

        def parser(payload: bytes, *, archive_name: str):
            if archive_name != "es-conversation-timestamp-aligned":
                raise RuntimeError("Unexpected corpus")
            decoded = json.loads(payload)
            if (len(decoded["train"]), len(decoded["dev"])) != (len(train), DEV_N):
                raise RuntimeError("Prepared corpus changed")
            if sha_pairs(decoded["train"]) != sha_pairs(train):
                raise RuntimeError("Training corpus changed after split freeze")
            return decoded["train"], decoded["dev"], len(train) + DEV_N

        core.parse_pairs = parser
        report = core.train_experiment(ROOT, output, LANGUAGE)
        report.update({
            "experiment": "Spanish video-disjoint genuine conversational subtitle LoRA",
            "research_only": True, "commercial_deployment_allowed": False,
            "source_license": "CC BY-NC 4.0",
            "real_optimizer_updates": report["completed_updates"],
            "video_disjoint_data": manifest,
            "selection_using_dev_only": True,
        })
        (output / "report.json").write_text(json.dumps(report, indent=2))

    # Model selection is frozen. Once-only exploratory scores on video-disjoint
    # heldout source videos; NO subsequent model selection on this test cohort.
    plain_dev_scores = evaluate_frozen_test(
        LANGUAGE, output, dev_plain, report["selected"],
        previous_adapter=ROOT / WARM_REL)
    test_scores = evaluate_frozen_test(
        LANGUAGE, output, test, report["selected"],
        previous_adapter=ROOT / WARM_REL)
    # Do NOT score the frozen test a second time to tune for plain-input output.
    summary = {
        "language": "spanish", "completed_updates": report["completed_updates"],
        "selected_on_contextual_development_only": report["selected"],
        "data_manifest": manifest,
        "contextual_development_scores": report["scores"],
        "plain_input_development_ablation": plain_dev_scores,
        "once_only_contextual_video_disjoint_test_scores": test_scores,
        "plain_test_not_scored_or_used_for_selection": True,
        "heldout_test_not_used_for_training_or_checkpoint_selection": True,
        "human_semantic_accuracy_95_percent_verified": False,
        "commercial_deployment_allowed": False,
        "limitations": "Video-disjoint transcript test is an exploratory reference-similarity "
                       "diagnostic, not independent blinded bilingual adequacy. One-to-one "
                       "timestamp alignment may contain misaligned captions; upstream Marian "
                       "pretraining exposure cannot be ruled out. Need real slang, code-switching, "
                       "pronoun, idiom and context human review before 95% claims.",
    }
    (output / "spanish_context_evaluation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("SPANISH_CONTEXT_REAL_TRAINING_RESULTS", json.dumps(summary), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", required=False, type=Path)
    cli.add_argument("--inspect-alignment", action="store_true", help="Audit source alignment only; do not train")
    opts = cli.parse_args()
    if opts.inspect_alignment:
        records, files = download_records()
        counts = {"source_records": len(records), "srt_es_records": 0, "srt_en_records": 0,
                  "paired_videos": 0, "aligned_cues": 0, "spanish_cues": 0, "english_cues": 0}
        for record in records:
            es, en = parse_srt(record.get("srt_es")), parse_srt(record.get("srt_en"))
            counts["srt_es_records"] += bool(es)
            counts["srt_en_records"] += bool(en)
            counts["spanish_cues"] += len(es)
            counts["english_cues"] += len(en)
            matches = align_video(record["video_id"], es, en)
            counts["paired_videos"] += bool(matches)
            counts["aligned_cues"] += len(matches)
        print("SPANISH_ALIGNMENT_INSPECTION", json.dumps({"counts": counts, "source_files": files}), flush=True)
    else:
        if opts.output_dir is None:
            cli.error("--output-dir is required without --inspect-alignment")
        run(opts.output_dir.resolve())
