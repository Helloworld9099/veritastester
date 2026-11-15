"""Evaluate local ASR on held-out permitted speech clips, by language."""
import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

from .audio import SpeechRecognizer

def edit_distance(reference: list[str], predicted: list[str]) -> int:
    old = list(range(len(predicted) + 1))
    for i, word in enumerate(reference, 1):
        row = [i]
        for j, guess in enumerate(predicted, 1):
            row.append(min(row[-1] + 1, old[j] + 1, old[j - 1] + (word != guess)))
        old = row
    return old[-1]

def decode_audio(path: Path) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(path), "-ac", "1",
         "-ar", "16000", "-f", "s16le", "pipe:1"],
        capture_output=True, check=True)
    return proc.stdout

def main():
    parser = argparse.ArgumentParser(description="Report per-language word error rate")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV columns file,language,transcript; file paths relative to the manifest")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    recognizer = SpeechRecognizer()
    errors = defaultdict(int)
    lengths = defaultdict(int)
    with args.manifest.open(encoding="utf-8", newline="") as file:
        for index, row in enumerate(csv.DictReader(file)):
            if index >= args.limit:
                break
            raw = decode_audio((args.manifest.parent / row["file"]).resolve())
            predicted, _ = recognizer.transcribe(raw, row["language"])
            reference = row["transcript"].casefold().split()
            hypothesis = predicted.casefold().split()
            errors[row["language"]] += edit_distance(reference, hypothesis)
            lengths[row["language"]] += len(reference)
            print(f"{row['language']}: reference={row['transcript']!r} prediction={predicted!r}")
    for language in sorted(lengths):
        words = lengths[language]
        print(f"{language}: WER={errors[language] / words:.3f} reference_words={words}" if words else
              f"{language}: reference has zero words")

if __name__ == "__main__":
    main()
