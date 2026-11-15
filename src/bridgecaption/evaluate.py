"""Compare the SAME model and source sentences with and without context.

Requires GEMINI_API_KEY and appropriate independent held-out human references.
Do not present synthetic-demo chrF as a competition-grade result.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from .provider import GeminiTranslator, TranslationRequest

def run_pair(example, translator):
    row = {"language": example["language"], "source": example["source"],
           "reference": example["reference"]}
    for name, enabled in (("baseline", False), ("context", True)):
        start = time.monotonic()
        row[name] = translator.translate(TranslationRequest(
            text=example["source"], source=example["language"],
            target=example.get("target", "en"), topic=example.get("topic", "general"),
            previous=example.get("previous", []), glossary=example.get("glossary", {}),
            context_enabled=enabled))
        row[name + "_seconds"] = round(time.monotonic() - start, 3)
    return row

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/context_demo.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/translation_comparison.csv"))
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args()
    if not 1 <= args.limit <= 1000:
        p.error("limit must be between 1 and 1000")
    translator = GeminiTranslator()
    results = []
    with args.data.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                results.append(run_pair(json.loads(line), translator))
            if len(results) >= args.limit:
                break
    if not results:
        p.error("no examples")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} paired examples to {args.out}")
    try:
        from sacrebleu.metrics import CHRF
        scorer = CHRF()
        for language in sorted({r["language"] for r in results}):
            subset = [r for r in results if r["language"] == language]
            refs = [r["reference"] for r in subset]
            scores = [scorer.corpus_score([r[k] for r in subset], [refs]).score for k in ("baseline", "context")]
            print(f"{language}: n={len(subset)} baseline_chrF={scores[0]:.2f}, context_chrF={scores[1]:.2f}")
    except ImportError:
        print("Install optional eval dependencies to calculate chrF.")
    print("Review errors and meaningful slang in both outputs manually.")

if __name__ == "__main__":
    main()
