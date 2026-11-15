"""Opt-in multilingual slang-sense classifier with safe JSON artifacts.

A supervised character n-gram model; not an LLM/JEPA. Meaning labels require
independent fluent-speaker annotation. With no vetted model, the app abstains.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .research_training import split_groups


@dataclass(frozen=True)
class SenseExample:
    case_id: str
    group_id: str
    language: str
    expression: str
    utterance: str
    previous: str
    situation: str
    meaning: str
    provenance: str

    @classmethod
    def from_dict(cls, data):
        fields = set(cls.__dataclass_fields__)
        if any(k not in data for k in fields):
            raise ValueError("Missing required sense-example field")
        record = {k: data[k] for k in fields}
        if any(not isinstance(v, str) or not v.strip() or len(v) > 2000
               for v in record.values()):
            raise ValueError("Invalid or empty sense example")
        if len(record["expression"]) > 60 or len(record["meaning"]) > 120:
            raise ValueError("Sense label is too long")
        return cls(**record)


def read_examples(path: Path) -> list[SenseExample]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = SenseExample.from_dict(json.loads(line))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid example line {line_number}: {exc}") from exc
        if row.case_id in seen:
            raise ValueError("Duplicate case identifier")
        seen.add(row.case_id)
        rows.append(row)
    if not rows:
        raise ValueError("No sense examples")
    return rows


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def contains_expression(text: str, expression: str) -> bool:
    txt, needle = normalize(text), normalize(expression)
    if not needle:
        return False
    if re.search(r"[a-z]", needle):
        return bool(re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", txt))
    return needle in txt


def grams(text: str, expression: str):
    txt = normalize(text)
    needle = normalize(expression)
    if needle:
        txt = txt.replace(needle, " ")
    txt = " ".join(txt.split())[:1800]
    return Counter(txt[i:i + n] for n in (2, 3, 4)
                   for i in range(max(0, len(txt) - n + 1))
                   if txt[i:i + n].strip())


def view_text(utterance: str, previous: str, situation: str, two_view: bool):
    return " ".join((utterance, previous, situation if two_view else ""))


def fit(rows: list[SenseExample], two_view: bool = True) -> dict:
    groups = defaultdict(lambda: defaultdict(Counter))
    documents = defaultdict(Counter)
    for row in rows:
        key = f"{row.language}\t{normalize(row.expression)}"
        documents[key][row.meaning] += 1
        groups[key][row.meaning].update(grams(
            view_text(row.utterance, row.previous, row.situation, two_view), row.expression))
    result = {}
    for key, labels in groups.items():
        if len(labels) < 2:
            continue
        result[key] = {
            "docs": dict(documents[key]),
            "ngrams": {label: dict(count) for label, count in labels.items()},
        }
    if not result:
        raise ValueError("At least one expression needs two sense labels in training groups")
    return {"format": "bridgecaption_senses_v1", "two_view": two_view,
            "expressions": result}


def predict(model: dict, utterance: str, previous: str, situation: str,
            language: str, min_probability: float = .85) -> dict[str, str]:
    if model.get("format") != "bridgecaption_senses_v1":
        raise ValueError("Unrecognized sense artifact")
    if not 0 < min_probability < 1:
        raise ValueError("Invalid confidence threshold")
    hints = {}
    for key, group in model.get("expressions", {}).items():
        lang, expr = key.split("\t", 1)
        if lang != language or not contains_expression(utterance, expr):
            continue
        features = grams(view_text(utterance, previous, situation, model["two_view"]), expr)
        vocabulary = set().union(*(set(c) for c in group["ngrams"].values()))
        counts = group["docs"]
        total_docs = sum(counts.values())
        scores = {}
        for label, n in counts.items():
            occurrences = group["ngrams"][label]
            total_tokens = sum(occurrences.values())
            smooth = max(len(vocabulary), 1)
            evidence = sum(freq * math.log((occurrences.get(token, 0) + 1) /
                              (total_tokens + smooth)) for token, freq in features.items())
            scores[label] = math.log((n + 1) / (total_docs + len(counts))) + \
                            evidence / max(sum(features.values()), 1)
        best = max(scores, key=scores.get)
        high = scores[best]
        probability = 1 / sum(math.exp(value - high) for value in scores.values())
        if probability >= min_probability:
            hints[expr] = best
    return hints


def evaluate(rows: list[SenseExample], model: dict, threshold: float):
    scored = 0
    correct = 0
    abstained = 0
    by_language = defaultdict(lambda: {"n": 0, "correct": 0, "abstained": 0})
    for row in rows:
        hints = predict(model, row.utterance, row.previous, row.situation,
                        row.language, threshold)
        expected = normalize(row.expression)
        prediction = hints.get(expected)
        stats = by_language[row.language]
        stats["n"] += 1
        if prediction is None:
            abstained += 1
            stats["abstained"] += 1
        else:
            scored += 1
            correct += prediction == row.meaning
            stats["correct"] += prediction == row.meaning
    return {"count": len(rows), "coverage": scored / len(rows) if rows else 0.0,
            "accuracy_on_covered": correct / scored if scored else None,
            "abstentions": abstained, "by_language": dict(by_language)}


def train(rows: list[SenseExample]) -> dict:
    train_rows, dev, test = split_groups(rows)
    model = fit(train_rows)
    candidates = (.5, .6, .7, .8, .85, .9, .95)
    def utility(t):
        ev = evaluate(dev, model, t)
        n = ev["count"] - ev["abstentions"]
        correct = int(round((ev["accuracy_on_covered"] or 0) * n))
        return correct - 2 * (n - correct)
    threshold = max(candidates, key=lambda t: (utility(t), t))
    return {"model": model, "threshold": threshold, "test": evaluate(test, model, threshold),
            "train_n": len(train_rows), "dev_n": len(dev), "test_n": len(test)}


def load_predictor(path: str | Path | None = None):
    path = path if path is not None else os.getenv("BRIDGE_SENSE_MODEL", "")
    if not path:
        return None
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("model", {}).get("format") != "bridgecaption_senses_v1":
        raise ValueError("Unsupported sense artifact")
    return obj


def inferred_senses(artifact, text: str, previous: list[str], topic: str, language: str):
    if not artifact or not previous:
        return {}
    return predict(artifact["model"], text, " ".join(previous[-4:]), topic, language,
                   artifact["threshold"])


def main():
    parser = argparse.ArgumentParser(description="Validate/train grounded slang sense classifier")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--confirm-live-verified", action="store_true")
    parser.add_argument("--preflight-report", type=Path,
                        default=Path("results/foundation_preflight.json"))
    parser.add_argument("--out", type=Path, default=Path("results/sense_model.json"))
    args = parser.parse_args()
    rows = read_examples(args.data)
    split_groups(rows)
    print(f"Validated {len(rows)} sense examples across {len({r.group_id for r in rows})} groups")
    if not args.train:
        return
    if not args.confirm_live_verified:
        parser.error("Training blocked until actual audio, provider, privacy and latency are verified")
    try:
        preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        parser.error("Run bridgecaption-preflight successfully before training")
    if not preflight.get("offline_code_checks_pass"):
        parser.error("Offline foundation checks must pass before training")
    result = train(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote opt-in sense model and holdout metrics to {args.out}")


if __name__ == "__main__":
    main()
