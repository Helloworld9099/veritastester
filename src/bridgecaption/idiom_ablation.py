"""Opt-in *auxiliary idiomaticity* ablation on MAGPIE and SemEval labels.

This does NOT train a translation model or exact slang meanings. The two-view
condition adds independently available conversational context (when present);
it is not JEPA, RL, or an independently learned semantic description.
All training runs require explicit consent, rights and offline preflight gates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from .corpus_curriculum import load_normalized, leakage_report

LABELS = {"literal": 0, "idiomatic": 1}


def eligible(records):
    return [r for r in records if r["role"] == "idiomaticity" and
            r["source_label"] in LABELS and r["dataset"] in {"magpie", "semeval"}]


def validate_partitions(rows):
    groups = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["split"])
    if any(len(partitions) > 1 for partitions in groups.values()):
        raise ValueError("Same idiom type occurs in more than one split")
    by_split = {key: [r for r in rows if r["split"] == key]
                for key in ("train", "dev", "test")}
    if not by_split["train"] or not by_split["dev"] or not by_split["test"]:
        raise ValueError("Need non-empty source-provided train, dev, and test groups")
    for split, examples in by_split.items():
        if len({r["group_id"] for r in examples}) < (3 if split == "train" else 1):
            raise ValueError(f"Too few independent idiom types in {split}")
        if {r["source_label"] for r in examples} != set(LABELS):
            raise ValueError(f"{split} needs both idiomatic and literal examples")
    return by_split


def render(row, two_view: bool):
    # Avoid leaking the gold label/translation into the predictor's input.
    current = f"expression: {row['expression']} current: {row['utterance']}"
    if two_view:
        return current + f" previous: {row['previous']} scene: {row['situation']}"
    return current


def score_binary(real, predicted):
    """Return balanced accuracy and macro F1 with no hidden data leakage."""
    totals = Counter(real)
    by_class = {}
    f1s = []
    recalls = []
    for label in (0, 1):
        tp = sum(a == label and b == label for a, b in zip(real, predicted))
        fp = sum(a != label and b == label for a, b in zip(real, predicted))
        fn = sum(a == label and b != label for a, b in zip(real, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_class["idiomatic" if label else "literal"] = {
            "support": totals[label], "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}
        f1s.append(f1)
        recalls.append(recall)
    return {"macro_f1": round(sum(f1s) / 2, 4),
            "balanced_accuracy": round(sum(recalls) / 2, 4),
            "confusion_matrix_labels": ["literal", "idiomatic"],
            "confusion_matrix": [[sum(a == i and b == j for a, b in zip(real, predicted))
                                  for j in (0, 1)] for i in (0, 1)],
            "by_class": by_class}


def run_model(train, dev, test, two_view: bool):
    # Lazy import: simply verifying corpus splits never installs/trains a model.
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    model_hash_dim = 2**15
    vectorizer = HashingVectorizer(n_features=model_hash_dim, analyzer="char",
                                  ngram_range=(3, 4), lowercase=True,
                                  alternate_sign=False, norm="l2")
    classifier = SGDClassifier(loss="log_loss", alpha=.0001, max_iter=30,
                               random_state=2026, class_weight="balanced",
                               tol=.0001)
    begin = time.perf_counter()
    x_train = vectorizer.transform([render(row, two_view) for row in train])
    y_train = [LABELS[row["source_label"]] for row in train]
    classifier.fit(x_train, y_train)
    training_seconds = time.perf_counter() - begin

    def confidences(rows):
        x = vectorizer.transform([render(row, two_view) for row in rows])
        return classifier.predict_proba(x)[:, 1].tolist()

    dev_probs = confidences(dev)
    y_dev = [LABELS[row["source_label"]] for row in dev]
    thresholds = (.25, .35, .45, .5, .55, .65, .75)
    # Threshold chosen only on development set; never optimize against test.
    threshold = max(thresholds, key=lambda t: (
        score_binary(y_dev, [int(p >= t) for p in dev_probs])["macro_f1"],
        -abs(t - .5)))
    y_test = [LABELS[row["source_label"]] for row in test]
    test_probs = confidences(test)
    test_metrics = score_binary(y_test, [int(p >= threshold) for p in test_probs])
    by_language = {}
    for language in sorted({r["language"] for r in test}):
        indices = [i for i,r in enumerate(test) if r["language"] == language]
        by_language[language] = score_binary(
            [y_test[i] for i in indices],
            [int(test_probs[i] >= threshold) for i in indices])
        by_language[language]["n"] = len(indices)

    # JSON-only artifact, no pickle/joblib executable deserialization.
    latency = []
    for row in test[:200]:
        begin = time.perf_counter()
        classifier.predict_proba(vectorizer.transform([render(row, two_view)]))
        latency.append((time.perf_counter() - begin) * 1000)
    latency.sort()
    return {
        "two_view": two_view,
        "threshold": threshold,
        "hash_features": {"dimension": model_hash_dim, "analyzer": "char",
                          "ngram_range": [3,4], "alternate_sign": False},
        "weights": [round(float(w), 8) for w in classifier.coef_[0]],
        "intercept": round(float(classifier.intercept_[0]), 8),
        "train_n": len(train), "dev_n": len(dev), "test_n": len(test),
        "test_metrics": test_metrics, "by_language": by_language,
        "training_seconds": round(training_seconds, 6),
        "inference_ms_p50": round(latency[len(latency)//2], 4),
        "inference_ms_p95": round(latency[min(len(latency)-1, int(.95*len(latency)))], 4),
        "latency_sample_n": len(latency),
        "test_predictions": [{"case_id": row["case_id"], "group_id": row["group_id"],
                              "gold": y, "probability_idiomatic": round(p, 8)}
                             for row, y, p in zip(test, y_test, test_probs)],
        "partitions": {name: {"rows": len(rows),
                               "groups": len({r["group_id"] for r in rows}),
                               "class_balance": dict(Counter(r["source_label"] for r in rows)),
                               "by_language": dict(Counter(r["language"] for r in rows))}
                       for name, rows in [("train", train), ("dev", dev), ("test", test)]},
    }


def main():
    p = argparse.ArgumentParser(description="Auxiliary single/two-view idiomaticity research")
    p.add_argument("--inputs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("results/idiomaticity_models.json"))
    p.add_argument("--train", action="store_true")
    p.add_argument("--confirm-dataset-rights", action="store_true")
    p.add_argument("--preflight-report", type=Path,
                   default=Path("results/foundation_preflight.json"))
    args = p.parse_args()
    try:
        records = load_normalized(args.inputs)
        r = leakage_report(records)
        if r["cross_split_group_conflicts"]:
            raise ValueError("Data leakage: idiom type in multiple splits")
        examples = eligible(records)
        parts = validate_partitions(examples)
    except (ValueError, OSError, json.JSONDecodeError) as err:
        p.error(str(err))
    print(json.dumps({"eligible_idiomaticity": len(examples),
                      "splits": {k: len(v) for k,v in parts.items()},
                      "trained": False}, indent=2))
    if not args.train:
        return
    if not args.confirm_dataset_rights:
        p.error("Confirm dataset rights and conditions before training")
    try:
        report = json.loads(args.preflight_report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        p.error(f"Run bridgecaption-preflight first: {exc}")
    if not report.get("offline_code_checks_pass"):
        p.error("Offline code preflight failed")
    if any(Counter(r["source_label"] for r in parts["train"])[key] < 50 for key in LABELS):
        p.error("Need at least 50 examples of each binary class to train")
    try:
        results = {name: run_model(parts["train"], parts["dev"], parts["test"], two_view)
                   for name, two_view in [("single_view", False), ("two_view", True)]}
    except ImportError as exc:
        p.error(f"Install the optional research dependency: pip install -e '.[research]' ({exc})")
    artifact = {
        "format": "bridgecaption_idiom_ablation_v1",
        "label_task": "idiomatic_vs_literal_not_translation",
        "models": results,
        "warning": "Human idiomaticity labels do not provide exact slang meanings",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        p.error("Refusing to overwrite prior training artifact")
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False)
    print(f"Saved {args.out}; no cloud requests made")


if __name__ == "__main__":
    main()
