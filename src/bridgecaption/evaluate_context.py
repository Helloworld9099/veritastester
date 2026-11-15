"""Controlled four-arm translation comparison on held-out conversation groups.

Network/API calls only with --execute. Requires separately trained predictor
weights and independently reviewed reference translations.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from .context_engine import ContextSelector
from .provider import TranslationRequest
from .providers import create_translator
from .research_training import load_examples, split_groups

ARMS = ("A_no_context", "B_all_context_two_view",
        "C_selected_single_view", "D_selected_two_view")


def build_requests(row, artifact: Path, selectors=None):
    """Build controlled ablations without using withheld reference text."""
    if not artifact:
        raise ValueError("A trained memory artifact is required for C and D")
    previous = [row.previous, row.candidate_memory]
    if selectors is None:
        selectors = (ContextSelector(artifact, variant="single_view"),
                     ContextSelector(artifact, variant="two_view"))
    single = selectors[0].select(
        row.utterance, [row.candidate_memory], row.language, "general",
        background=row.previous)
    multiple = selectors[1].select(
        row.utterance, [row.candidate_memory], row.language, row.situation,
        background=row.previous)
    conditions = (
        ((), "general", False),
        (tuple(previous), row.situation, True),
        (single.memories, "general", True),
        (multiple.memories, row.situation, True),
    )
    return {name: TranslationRequest(text=row.utterance, source=row.language, target="en",
                                     topic=topic, previous=list(memory),
                                     context_enabled=enabled)
            for name, (memory, topic, enabled) in zip(ARMS, conditions)}


def evaluate(rows, translator, artifact: Path):
    results = []
    selectors = (ContextSelector(artifact, variant="single_view"),
                 ContextSelector(artifact, variant="two_view"))
    for row in rows:
        requests = build_requests(row, artifact, selectors)
        item = {"case_id": row.case_id, "group_id": row.group_id, "language": row.language,
                "reference": row.reference_translation, "judgement": row.human_judgement}
        for name, req in requests.items():
            begin = time.monotonic()
            item[name] = translator.translate(req)
            item[name + "_seconds"] = round(time.monotonic() - begin, 3)
            item[name + "_included_history"] = bool(req.previous)
        results.append(item)
    return results


def summary(rows):
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        return {"warning": "pip install sacrebleu to score chrF"}
    scorer = CHRF()
    outcomes = {}
    for language in sorted({row["language"] for row in rows}):
        subset = [row for row in rows if row["language"] == language]
        refs = [row["reference"] for row in subset]
        outcomes[language] = {
            arm: {"chrF": round(scorer.corpus_score([r[arm] for r in subset], [refs]).score, 2),
                  "n": len(subset),
                  "mean_translation_seconds": round(
                      sum(r[arm + "_seconds"] for r in subset) / len(subset), 3)}
            for arm in ARMS
        }
    return outcomes


def main():
    parser = argparse.ArgumentParser(description="4-arm held-out context comparison")
    parser.add_argument("--data", type=Path, required=True,
                        help="Human-annotated contextual examples JSONL")
    parser.add_argument("--artifact", type=Path, required=True,
                        help="Opt-in trained memory predictor")
    parser.add_argument("--engine", choices=("groq", "ollama", "gemini"),
                        default="ollama")
    parser.add_argument("--execute", action="store_true",
                        help="Authorize real translation API requests")
    parser.add_argument("--out", type=Path, default=Path("results/four_arm.csv"))
    args = parser.parse_args()
    records = load_examples(args.data)
    _, _, test = split_groups(records)
    build_requests(test[0], args.artifact)
    print(f"Held-out test: {len(test)} cases from {len({r.group_id for r in test})} groups")
    if not args.execute:
        print("Validation only. Run with --execute to call a real translator.")
        return
    translator = create_translator(args.engine)
    items = evaluate(test, translator, args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    print(json.dumps(summary(items), ensure_ascii=False, indent=2))
    print("Manually review slang, harmful transfers, consent and actual meanings.")


if __name__ == "__main__":
    main()
