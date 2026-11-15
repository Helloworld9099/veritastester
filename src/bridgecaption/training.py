"""Reproducible supervised topic classifier; trained on small synthetic data.

This trains a topic HINT model, NOT the main Whisper or Gemini models.
"""
import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

def features(text):
    words = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
    return words + [a + "__" + b for a, b in zip(words, words[1:])]

def fit(rows):
    if not rows:
        raise ValueError("No training rows")
    labels = Counter()
    counts = defaultdict(Counter)
    for text, topic in rows:
        labels[topic] += 1
        counts[topic].update(features(text))
    vocab = set(word for count in counts.values() for word in count)
    return {"labels": dict(labels), "counts": {k: dict(v) for k, v in counts.items()},
            "vocab": len(vocab)}

def classify(model, text):
    labels, counts = model["labels"], model["counts"]
    if not labels:
        return "general", 0
    total = sum(labels.values())
    seen = set(word for c in counts.values() for word in c)
    scores = {}
    for name, documents in labels.items():
        current = counts[name]
        denominator = sum(current.values()) + model["vocab"]
        score = math.log((documents + 1) / (total + len(labels)))
        for word, repeats in Counter(features(text)).items():
            if word in seen:
                score += repeats * math.log((current.get(word, 0) + 1) / denominator)
        scores[name] = score
    best = max(scores, key=scores.get)
    peak = scores[best]
    weights = [math.exp(value - peak) for value in scores.values()]
    return best, 1 / sum(weights)

def holdout(rows, fraction=.2, seed=2026):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[1]].append(row)
    training, testing = [], []
    for label, group in sorted(grouped.items()):
        items = group[:]
        rng.shuffle(items)
        split = max(1, round(len(items) * fraction))
        training.extend(items[split:])
        testing.extend(items[:split])
    return training, testing

def train(path: Path, output: Path, metrics_path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [(r["text"].strip(), r["topic"].strip()) for r in csv.DictReader(stream)]
    rows = [(text, topic) for text, topic in rows if text and topic]
    if len({text.lower() for text, _ in rows}) != len(rows):
        raise ValueError("Duplicate examples")
    training, testing = holdout(rows)
    model = fit(training)
    correct = sum(classify(model, text)[0] == topic for text, topic in testing)
    per_topic = {}
    for topic in sorted({y for _, y in testing}):
        group = [(text, target) for text, target in testing if target == topic]
        per_topic[topic] = {"count": len(group), "correct": sum(classify(model, t)[0] == topic for t, _ in group)}
    metrics = {"train_n": len(training), "test_n": len(testing), "correct": correct,
               "accuracy": correct / len(testing) if testing else 0, "per_topic": per_topic,
               "warning": "Synthetic topic tests are not multilingual speech-translation scores."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics

def infer_topic(text: str, model_path: Path | None = None) -> str:
    if model_path is None:
        model_path = Path(__file__).resolve().parents[2] / "data" / "topic_model.json"
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        topic, confidence = classify(model, text)
        return topic if confidence >= .47 else "general"
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
        return "general"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/topic_train.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/topic_model.json"))
    parser.add_argument("--metrics", type=Path, default=Path("data/topic_metrics.json"))
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out, args.metrics), indent=2))

if __name__ == "__main__":
    main()
