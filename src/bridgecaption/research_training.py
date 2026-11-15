"""Offline experiment scaffold for selective meeting-context transfer.

The original two-view/transfer-value hypotheses were proposed for RL. This
module tests a smaller analogous question for translation; it is not JEPA or
reinforcement learning. No model is trained unless --train is explicitly set.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .research_features import context_features


@dataclass(frozen=True)
class MemoryExample:
    case_id: str
    group_id: str  # Entire conversation/speaker family, never a single sentence.
    language: str
    utterance: str
    previous: str
    situation: str  # Complementary grounded view; NO reference translation.
    candidate_memory: str
    reference_translation: str
    human_judgement: str  # helpful, neutral, harmful; independent fluent review.
    provenance: str  # Licensed dataset or consented human example.

    @classmethod
    def from_dict(cls, obj: dict) -> MemoryExample:
        required = set(cls.__dataclass_fields__)
        missing = required - set(obj)
        if missing:
            raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")
        values = {key: obj[key] for key in required}
        if any(not isinstance(value, str) or not value.strip() or len(value) > 4000
               for value in values.values()):
            raise ValueError("All example fields must be nonempty bounded strings")
        if values["human_judgement"] not in {"helpful", "neutral", "harmful"}:
            raise ValueError("human_judgement must be helpful, neutral or harmful")
        if len(values["language"]) > 16:
            raise ValueError("Invalid language code")
        return cls(**values)


def load_examples(path: Path) -> list[MemoryExample]:
    seen = set()
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = MemoryExample.from_dict(json.loads(raw))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid example on line {line_no}: {exc}") from exc
        if row.case_id in seen:
            raise ValueError(f"Duplicate case_id: {row.case_id}")
        seen.add(row.case_id)
        rows.append(row)
    if not rows:
        raise ValueError("No examples")
    return rows


def split_groups(rows: list[MemoryExample], seed: int = 2026):
    """Keep related utterances in the SAME split to prevent test leakage."""
    groups = sorted({row.group_id for row in rows})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    if n < 6:
        raise ValueError("Need at least six independent conversation/speaker groups")
    train_end = max(1, round(n * .6))
    dev_end = max(train_end + 1, round(n * .8))
    bins = [set(groups[:train_end]), set(groups[train_end:dev_end]), set(groups[dev_end:])]
    if any(not b for b in bins):
        raise ValueError("Each train/dev/test split needs independent groups")
    split = [[row for row in rows if row.group_id in b] for b in bins]
    assert not (bins[0] & bins[1] or bins[0] & bins[2] or bins[1] & bins[2])
    return tuple(split)


def features(row: MemoryExample, multiview: bool) -> tuple[float, ...]:
    """Exactly the same eight features used for opt-in live context inference."""
    return context_features(row.utterance, row.previous, row.candidate_memory,
                            row.situation, multiview)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    exponent = math.exp(x)
    return exponent / (1 + exponent)


def probability(weights: list[float], row: MemoryExample, multiview: bool) -> float:
    return sigmoid(sum(w * x for w, x in zip(weights, features(row, multiview))))


def train_predictor(rows: list[MemoryExample], multiview: bool, epochs=250, lr=.08) -> list[float]:
    """Deterministic logistic baseline to learn when past utterances help.

    Not a learned JEPA embedding. Labels require independently scored examples.
    """
    weights = [0.0] * 8
    prepared = [(features(row, multiview), float(row.human_judgement == "helpful"))
                for row in sorted(rows, key=lambda r: r.case_id)]
    for _ in range(epochs):
        for vector, target in prepared:
            error = sigmoid(sum(w * x for w, x in zip(weights, vector))) - target
            weights = [w - lr * (error * x + .001 * w) for w, x in zip(weights, vector)]
    return weights


def score(rows: list[MemoryExample], weights: list[float], multiview: bool, threshold: float):
    selected = [row for row in rows if probability(weights, row, multiview) >= threshold]
    selected_ids = {row.case_id for row in selected}
    return {
        "count": len(rows), "selected": len(selected),
        "helpful_selected": sum(row.human_judgement == "helpful" for row in selected),
        "harmful_selected": sum(row.human_judgement == "harmful" for row in selected),
        "neutral_selected": sum(row.human_judgement == "neutral" for row in selected),
        "helpful_missed": sum(row.human_judgement == "helpful" for row in rows if row.case_id not in selected_ids),
    }


def choose_threshold(rows: list[MemoryExample], weights: list[float], multiview: bool):
    """Tune on DEV only; penalize negative transfer twice as strongly."""
    candidates = (.5, .6, .7, .8, .9, 1.0)  # 1.0 means abstain everywhere.
    def utility(threshold):
        result = score(rows, weights, multiview, threshold)
        return result["helpful_selected"] - 2 * result["harmful_selected"] - .2 * result["neutral_selected"]
    return max(candidates, key=lambda t: (utility(t), t))


def run_experiment(rows: list[MemoryExample], *, min_examples_per_language: int = 30):
    language_counts = {code: sum(row.language == code for row in rows) for code in {r.language for r in rows}}
    too_small = {code: n for code, n in language_counts.items() if n < min_examples_per_language}
    if too_small:
        raise ValueError(f"Insufficient independently annotated examples per language: {too_small}")
    train, dev, test = split_groups(rows)
    for lang in language_counts:
        if not all(any(row.language == lang for row in part) for part in (train, dev, test)):
            raise ValueError(f"Language {lang!r} missing from an independent train/dev/test split")
    if not {"helpful", "harmful"}.issubset({r.human_judgement for r in train}):
        raise ValueError("Training split must include both helpful and harmful memories")
    outcomes = {}
    for name, use_multiview in (("single_view", False), ("two_view", True)):
        weights = train_predictor(train, use_multiview)
        threshold = choose_threshold(dev, weights, use_multiview)
        outcomes[name] = {"weights": weights, "threshold": threshold,
                          "test": score(test, weights, use_multiview, threshold),
                          "by_language": {lang: score([r for r in test if r.language == lang],
                                                      weights, use_multiview, threshold)
                                          for lang in sorted(language_counts)}}
    return {"method": "supervised_utility_predictor_not_RL_or_JEPA",
            "seed": 2026, "train_n": len(train), "dev_n": len(dev), "test_n": len(test),
            "languages": language_counts, "models": outcomes}


def main():
    parser = argparse.ArgumentParser(description="Validate/prepare selective-context research; training is opt-in")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train", action="store_true", help="Explicitly authorize offline training")
    parser.add_argument("--preflight-report", type=Path,
                        default=Path("results/foundation_preflight.json"))
    parser.add_argument("--confirm-live-verified", action="store_true",
                        help="Confirm real desktop, audio, API, privacy and latency gates were checked")
    parser.add_argument("--out", type=Path, default=Path("results/memory_research.json"))
    args = parser.parse_args()
    rows = load_examples(args.data)
    # Run the same minimum data/label gate whether training is requested or not.
    grouped = {g for g in (r.group_id for r in rows)}
    counts = {l: sum(r.language == l for r in rows) for l in {r.language for r in rows}}
    split_groups(rows)
    if any(c < 30 for c in counts.values()):
        parser.error(f"Need >=30 examples per tested language; currently {counts}")
    print(json.dumps({"readiness": "data-schema-valid", "examples": len(rows),
                      "independent_groups": len(grouped), "languages": counts,
                      "trained": False}, indent=2))
    if not args.train:
        return
    try:
        report = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"Training blocked: run bridgecaption-preflight first ({exc})")
    if not report.get("offline_code_checks_pass"):
        parser.error("Training blocked: preflight code compilation or test suite failed")
    if not args.confirm_live_verified:
        parser.error("Training blocked: independently verify live audio/API/privacy/latency first; "
                     "then add --confirm-live-verified")
    outcome = run_experiment(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(outcome, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Offline predictor trained; evaluation and weights written to {args.out}")


if __name__ == "__main__":
    main()
