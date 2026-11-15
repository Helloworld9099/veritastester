"""Permissioned audio replay for real ASR and translation evaluation.

No network requests unless both --translate and --execute are supplied.
Cloud uploads also require --allow-cloud-transcripts. Raw clips never leave
the computer; cloud services receive recognized speech text only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .asr_eval import decode_audio, edit_distance, score_units
from .pipeline import Pipeline
from .providers import PROVIDERS, create_translator, needs_cloud


REQUIRED = {"file", "language", "transcript"}


def read_manifest(path: Path, limit: int = 30):
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")
    base = path.resolve().parent
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError("CSV must have file,language,transcript columns")
        rows = []
        for index, row in enumerate(reader, start=2):
            if len(rows) >= limit:
                break
            if not all((row.get(key) or "").strip() for key in REQUIRED):
                raise ValueError(f"Missing required value on CSV line {index}")
            relative = Path(row["file"])
            if relative.is_absolute():
                raise ValueError(f"Use audio paths relative to CSV at line {index}")
            clip = (base / relative).resolve()
            if not clip.is_relative_to(base) or not clip.is_file():
                raise ValueError(f"Audio file unavailable or outside dataset folder at line {index}")
            previous_text = row.get("previous_json") or "[]"
            try:
                previous = json.loads(previous_text)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid previous_json at line {index}") from exc
            if (not isinstance(previous, list) or len(previous) > 4 or
                    any(not isinstance(turn, str) or len(turn) > 500 for turn in previous)):
                raise ValueError(f"previous_json must contain up to four short utterances at line {index}")
            rows.append({"file": clip, "language": row["language"].strip(),
                         "transcript": row["transcript"].strip(),
                         "translation": (row.get("translation") or "").strip(),
                         "previous": previous,
                         "topic": (row.get("topic") or "general").strip()[:100]})
    if not rows:
        raise ValueError("No audio clips in manifest")
    return rows


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    before, after = int(index), min(len(ordered) - 1, int(index) + 1)
    return round(ordered[before] + (ordered[after] - ordered[before]) * (index - before), 3)


def summarize(cases):
    by_language = defaultdict(list)
    for result in cases:
        by_language[result["language"]].append(result)
    report = {}
    for language, examples in sorted(by_language.items()):
        unit_count = sum(x["reference_units"] for x in examples)
        error_count = sum(x["asr_errors"] for x in examples)
        item = {"clips": len(examples),
                "asr_metric": "CER" if language in {"zh", "yue", "ja", "th", "lo", "km", "my"} else "WER",
                "asr_error_rate": round(error_count / unit_count, 4) if unit_count else None,
                "asr_median_seconds": percentile([x["asr_seconds"] for x in examples], .5),
                "asr_p95_seconds": percentile([x["asr_seconds"] for x in examples], .95)}
        translations = [x for x in examples if x.get("translated") is not None]
        if translations:
            item["translation_median_seconds"] = percentile(
                [x["translation_seconds"] for x in translations], .5)
            item["translation_p95_seconds"] = percentile(
                [x["translation_seconds"] for x in translations], .95)
            paired = [x for x in translations if x["reference_translation"]]
            if paired:
                try:
                    from sacrebleu.metrics import CHRF
                    scorer = CHRF()
                    item["translation_chrF"] = round(scorer.corpus_score(
                        [x["translated"] for x in paired],
                        [[x["reference_translation"] for x in paired]]).score, 2)
                    item["translation_reference_clips"] = len(paired)
                except ImportError:
                    item["translation_chrF"] = None
        report[language] = item
    return report


def run_replay(rows, recognizer, *, translator=None, contextual=False,
               context_mode="selective", target="en"):
    outputs = []
    for row in rows:
        raw = decode_audio(row["file"])
        start = time.monotonic()
        predicted, detected = recognizer.transcribe(raw, row["language"])
        asr_seconds = round(time.monotonic() - start, 3)
        reference = score_units(row["transcript"], row["language"])
        candidate = score_units(predicted, row["language"])
        output = {
            "language": row["language"], "reference_units": len(reference),
            "asr_errors": edit_distance(reference, candidate),
            "asr_seconds": asr_seconds, "translated": None,
            "reference_translation": row["translation"], "translation_seconds": None,
            "detected_language": detected,
        }
        if translator is not None and predicted.strip():
            # New isolated history per clip. Only the explicit preceding turns
            # belong to this clip; no other sample's audio/text is reused.
            pipeline = Pipeline(translator=translator, context_mode=context_mode)
            if contextual:
                for earlier in row["previous"]:
                    pipeline.history.add(earlier)
            begin = time.monotonic()
            result = pipeline.translate(predicted, row["language"], target,
                                        row["topic"], contextual=contextual)
            output["translated"] = result.translated
            output["translation_seconds"] = round(time.monotonic() - begin, 3)
            pipeline.reset()
        outputs.append(output)
    return outputs


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Replay consented clips through local ASR")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--execute", action="store_true",
                        help="Actually decode and transcribe authorized audio")
    parser.add_argument("--translate", action="store_true",
                        help="Also run translation after recognition")
    parser.add_argument("--allow-cloud-transcripts", action="store_true",
                        help="Confirm consent to send recognized text to selected cloud provider")
    parser.add_argument("--engine", choices=PROVIDERS, default="ollama")
    parser.add_argument("--contextual", action="store_true")
    parser.add_argument("--context-mode", choices=("all", "selective"), default="selective")
    parser.add_argument("--target", default="en")
    parser.add_argument("--out", type=Path, default=Path("results/audio_replay_metrics.json"))
    args = parser.parse_args()
    try:
        rows = read_manifest(args.manifest, args.limit)
    except (OSError, ValueError) as exc:
        parser.error(f"Invalid manifest: {exc}")
    print(f"Valid manifest: {len(rows)} clips (source audio remains local)")
    if not args.execute:
        print("DRY RUN only. Pass --execute to transcribe; --translate to include translation.")
        return
    if args.translate and needs_cloud(args.engine) and not args.allow_cloud_transcripts:
        parser.error("Cloud translation requires --allow-cloud-transcripts and participant permission")
    from .audio import SpeechRecognizer
    import os
    recognizer = SpeechRecognizer(os.getenv("WHISPER_MODEL", "base"))
    translator = create_translator(args.engine) if args.translate else None
    results = run_replay(rows, recognizer, translator=translator,
                         contextual=args.contextual, context_mode=args.context_mode,
                         target=args.target)
    report = {"engine": args.engine if args.translate else "none",
              "contextual": bool(args.contextual), "clips": len(results),
              "by_language": summarize(results),
              "limits": "No continuous-live capture tested; ASR evaluation uses isolated clips."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved aggregate metrics (no transcript text) to {args.out}")


if __name__ == "__main__":
    main()
