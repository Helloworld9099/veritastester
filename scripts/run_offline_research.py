"""Prepare pinned corpora and run a CPU-only auxiliary experiment. No live activation.

Run acquire_datasets.py --download first. Both --prepare and --train are explicit.
Never uses the live-verification attestation or supplies human meaning labels.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import resource
import subprocess
import sys
import time
import zipfile

from bridgecaption.corpus_prep import prepare, private_write, audit
from bridgecaption.corpus_curriculum import human_memory_candidates, human_sense_candidates
from bridgecaption.review_export import export_candidates
from bridgecaption.idiom_ablation import score_binary

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data/private"
RAW = PRIVATE / "raw"
RESULTS = ROOT / "results"
SEED = 2026
PRIORITY = {"train": 0, "dev": 1, "test": 2}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    path.chmod(0o600)


def decontaminate(rows):
    """Keep original holdouts; remove lower-priority rows, never move a split.

    Screens BNC documents, normalized text and >=.9 word-5-gram Jaccard.
    Near-duplicate candidates require a shared rare shingle (<=50 references).
    This is a lexical screen, not a guarantee against all semantic duplicates.
    """
    owners = {}
    def keys(r):
        text = " ".join(re.findall(r"\w+", r["utterance"].casefold()))
        doc = r.get("extra", {}).get("document_id")
        return (["doc:" + doc] if doc else []) + (["text:" + text] if len(text) >= 32 else [])
    for r in rows:
        for key in keys(r):
            owners[key] = max(owners.get(key, -1), PRIORITY[r["split"]])
    clean, removed = [], Counter()
    for r in rows:
        conflict = [k.split(":", 1)[0] for k in keys(r) if owners[k] > PRIORITY[r["split"]]]
        if conflict:
            removed[r["split"] + ":" + "+".join(sorted(set(conflict)))] += 1
        else:
            clean.append(r)
    def shingles(r):
        w = re.findall(r"\w+", r["utterance"].casefold())
        return set(zip(*(w[i:] for i in range(5)))) if len(w) >= 8 else set()
    held = [(r, shingles(r)) for r in clean if r["split"] != "train"]
    index = defaultdict(list)
    for i, (_, ss) in enumerate(held):
        for sh in ss:
            index[sh].append(i)
    final = []
    for r in clean:
        ss = shingles(r)
        hits = Counter(i for sh in ss if len(index.get(sh, [])) <= 50
                       for i in index.get(sh, [])
                       if PRIORITY[held[i][0]["split"]] > PRIORITY[r["split"]])
        duplicate = any(len(ss & held[i][1]) / max(1, len(ss | held[i][1])) >= .9 for i in hits)
        if duplicate:
            removed[r["split"] + ":near_duplicate"] += 1
        else:
            final.append(r)
    return final, {"removed": dict(removed), "retained": len(final),
                   "method": "original splits; test>dev>train document/text priority; rare-shingle .9 Jaccard screen"}


def prepare_magpie():
    stats = {}
    rows = prepare("magpie", RAW / "magpie-typebased.jsonl", language="en", limit=200000,
                   skip_invalid=True, rights="cc-by-4.0-reviewed", stats=stats)
    save_json(RESULTS / "magpie_import.json", {"import": stats, "audit": audit(rows)})
    private_write(PRIVATE / "magpie-official.jsonl", rows)
    clean, cleaning = decontaminate(rows)
    save_json(RESULTS / "magpie_cleaning.json", {**cleaning, "audit": audit(clean)})
    private_write(PRIVATE / "magpie-clean.jsonl", clean)
    return clean


def prepare_semeval():
    train = prepare("semeval", RAW / "semeval-train.csv", source_split="train",
                    rights="repository-gpl-3.0-research-private", limit=200000)
    with (RAW / "semeval-dev-gold.csv").open() as f:
        labels = {r["ID"]: r for r in csv.DictReader(f)}
    with (RAW / "semeval-dev.csv").open() as f:
        dev = list(csv.DictReader(f))
    assert len(labels) == len(dev) and all(r["ID"] in labels for r in dev)
    joined = []
    for r in dev:
        gold = labels[r["ID"]]
        assert r["Language"] == gold["Language"]
        joined.append({**r, "DataID": gold["DataID"], "Label": gold["Label"]})
    join_path = RAW / "semeval-dev-joined.jsonl"
    private_write(join_path, joined)
    dev_rows = prepare("semeval", join_path, source_split="dev", rights="repository-gpl-3.0-research-private")
    # The public test file lacks gold labels: preserve as unlabelled test only.
    with (RAW / "semeval-test.csv").open() as f:
        tests = [{**r, "DataID": "official-test:"+r["ID"]} for r in csv.DictReader(f)]
    test_path = RAW / "semeval-test-with-ids.jsonl"
    private_write(test_path, tests)
    test_rows = prepare("semeval", test_path, source_split="test", rights="repository-gpl-3.0-research-private")
    rows = train + dev_rows + test_rows
    private_write(PRIVATE / "semeval.jsonl", rows)
    save_json(RESULTS / "semeval_import.json", {"audit": audit(rows),
              "official_test_gold_available": False, "one_shot_imported": False,
              "experiment": "not trained: official test has no independent gold in pinned repository"})


def prepare_multi3woz():
    all_rows = []
    receipt = {}
    with zipfile.ZipFile(RAW / "multi3woz.zip") as z:
        parallel_ids = None
        for name, language in [("English", "en"), ("French", "fr"), ("Arabic", "ar"), ("Turkish", "tr")]:
            prefix = "data/" + name + "/"
            raw = z.read(prefix + "data.json")
            member_sha256 = hashlib.sha256(raw).hexdigest()
            data = json.loads(raw)
            dev_ids = set(z.read(prefix+"valListFile.txt").decode().splitlines())
            test_ids = set(z.read(prefix+"testListFile.txt").decode().splitlines())
            assert not dev_ids & test_ids
            missing_split_ids = {"dev": len(dev_ids - data.keys()), "test": len(test_ids - data.keys())}
            # The released split lists include MultiWOZ IDs omitted from this corpus.
            # Intersect with actual dialogues; never invent or reassign missing ones.
            dev_ids &= data.keys()
            test_ids &= data.keys()
            train_ids = set(data) - dev_ids - test_ids
            selected = {}
            for split, ids, n in [("train", train_ids, 120), ("dev", dev_ids, 30), ("test", test_ids, 30)]:
                # Same IDs and official split assignments across all parallel languages.
                choices = sorted(ids)
                random.Random(SEED).shuffle(choices)
                selected[split] = sorted(choices[:n])
                subset = {k: data[k] for k in selected[split]}
                path = RAW / f"multi3woz-{language}-{split}.json"
                save_json(path, subset)
                rows = prepare("multi3woz", path, language=language, source_split=split,
                               rights="repository-mit-research-private")
                for row in rows:
                    row["extra"]["source_member"] = prefix+"data.json"
                    row["extra"]["source_member_sha256"] = member_sha256
                all_rows.extend(rows)
            if parallel_ids is None:
                parallel_ids = selected
            assert selected == parallel_ids, "Parallel languages have different grouping/splits"
            receipt[language] = {"official_dialogues": len(data), "split_list_ids_absent_from_release": missing_split_ids,
                                 "selected_dialogues": {k: len(v) for k,v in selected.items()},
                                 "selected_ids_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()}
            del data, raw
    private_write(PRIVATE / "multi3woz.jsonl", all_rows)
    save_json(RESULTS / "multi3woz_import.json", {"languages": receipt, "audit": audit(all_rows)})
    return all_rows


def export_queues(magpie, dialogues):
    memory = []
    for language in ("en", "fr", "ar", "tr"):
        subset = [r for r in dialogues if r["language"] == language and r["split"] == "train"]
        grouped = defaultdict(list)
        for candidate in human_memory_candidates(subset, max_items=10000, seed=SEED):
            grouped[candidate["group_id"]].append(candidate)
        spread = [grouped[g][level] for level in range(100) for g in sorted(grouped)
                  if len(grouped[g]) > level]
        memory.extend(spread[:400])
    # Spread sense cases across idiom families before taking the bounded queue.
    buckets = defaultdict(list)
    for r in magpie:
        if r["split"] == "train":
            buckets[r["group_id"]].append(r)
    # Put families with both source idiomaticity labels first so a reviewer can
    # work on actual ambiguity. These remain unverified exact meanings.
    ambiguous = {g for g, values in buckets.items()
                 if len(Counter(r["source_label"] for r in values)) == 2
                 and min(Counter(r["source_label"] for r in values).values()) >= 10}
    sense_rows = [r for g in sorted(ambiguous) for r in buckets[g]]
    for level in range(10):
        for group in sorted(buckets):
            if group not in ambiguous and len(buckets[group]) > level:
                sense_rows.append(buckets[group][level])
    sense = human_sense_candidates(sense_rows, max_items=1000)
    reports = []
    for task, rows in [("memory", memory), ("sense", sense)]:
        private_write(PRIVATE / f"{task}_candidates.jsonl", rows)
        reports.append(export_candidates(rows, task, PRIVATE))
    save_json(RESULTS / "review_queue_counts.json", reports)


def train_and_measure():
    subprocess.run([sys.executable, "-m", "bridgecaption.preflight"], cwd=ROOT, check=True)
    start = time.perf_counter()
    output = RESULTS / "idiomaticity_models.json"
    subprocess.run([sys.executable, "-m", "bridgecaption.idiom_ablation", "--inputs",
                    str(PRIVATE / "magpie-clean.jsonl"), "--train", "--confirm-dataset-rights",
                    "--out", str(output)], cwd=ROOT, check=True)
    artifact = json.loads(output.read_text())
    summaries = {}
    for name, m in artifact["models"].items():
        summaries[name] = {k: v for k,v in m.items() if k not in {"weights", "test_predictions", "intercept", "hash_features"}}
        summaries[name]["weight_json_bytes"] = len(json.dumps({"weights": m["weights"], "intercept": m["intercept"]}).encode())
    a, b = [artifact["models"][k] for k in ("single_view", "two_view")]
    ga, gb = defaultdict(list), defaultdict(list)
    for model, target in [(a, ga), (b, gb)]:
        for row in model["test_predictions"]:
            target[row["group_id"]].append((row["gold"], int(row["probability_idiomatic"] >= model["threshold"])))
    assert set(ga) == set(gb)
    groups = sorted(ga)
    rng = random.Random(SEED)
    deltas = []
    for _ in range(500):
        sample = rng.choices(groups, k=len(groups))
        scores = []
        for data in (ga, gb):
            pairs = [pair for g in sample for pair in data[g]]
            scores.append(score_binary([p[0] for p in pairs], [p[1] for p in pairs])["macro_f1"])
        deltas.append(scores[1] - scores[0])
    deltas.sort()
    y = [p[0] for values in ga.values() for p in values]
    majority = int(a["partitions"]["train"]["class_balance"].get("idiomatic", 0) >= a["partitions"]["train"]["class_balance"].get("literal", 0))
    save_json(RESULTS / "experiment_summary.json", {
        "task": "auxiliary_idiomaticity_not_translation_or_memory_usefulness", "seed": SEED,
        "models": summaries, "majority_baseline": score_binary(y, [majority]*len(y)),
        "two_minus_single_macro_f1": round(b["test_metrics"]["macro_f1"] - a["test_metrics"]["macro_f1"], 4),
        "paired_idiom_group_bootstrap_95_interval": [round(deltas[12],4), round(deltas[487],4)],
        "bootstrap_replicates": 500, "process_total_seconds": round(time.perf_counter()-start, 3),
        "artifact_bytes": output.stat().st_size, "runner_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "CPU text classifier only; no Whisper/LLM fine-tuning or live model activation"})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    if not (args.prepare or args.train):
        print("DRY RUN: --prepare builds private datasets/queues; --train runs preflight and auxiliary idiomaticity only")
        return
    if args.prepare:
        magpie = prepare_magpie()
        prepare_semeval()
        dialogues = prepare_multi3woz()
        export_queues(magpie, dialogues)
    if args.train:
        train_and_measure()


if __name__ == "__main__":
    main()
