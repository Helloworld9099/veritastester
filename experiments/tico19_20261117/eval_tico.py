#!/usr/bin/env python3
"""One-time CC0 TICO-19 cross-domain test of frozen custom Marian LoRA variants.

No tuning or training on this benchmark. Scores are NOT translation accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/local_adapter_review_20261117"))
from translate import translate  # noqa: E402

SOURCE_REV = "55d70dc0b1d1d0b2151c5e22815d823fedac3f2f"
SOURCE_REPO = "gmnlp/tico19"
URL_ROOT = f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{SOURCE_REV}/test"
DATA_FILES = {"french": ("test.en-fr.tsv", "fr"),
              "spanish": ("test.en-es-LA.tsv", "es-LA")}
SEED = 202611177
COUNT = 256
VARIANTS = ("base", "massive", "massive2", "massive3")


def normalize(s: str):
    return " ".join(s.strip().casefold().split())


def select_rows(payload: bytes, language: str, count=COUNT, seed=SEED):
    if language not in DATA_FILES:
        raise ValueError("Unsupported TICO language")
    target = DATA_FILES[language][1]
    identifiers, src_seen, en_seen = set(), set(), set()
    candidates = []
    for line in payload.decode("utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) != 8:
            raise ValueError("Unexpected TICO parallel TSV schema")
        expected_targets = {target}
        if language == "spanish":
            expected_targets.update(("es", "es-419"))
        if fields[0].strip() != "en" or fields[1].strip() not in expected_targets:
            raise ValueError(
                f"TICO source or target language mismatch: got {fields[0].strip()!r} -> "
                f"{fields[1].strip()!r}; expected en -> {sorted(expected_targets)!r}")
        english, foreign, sample_id = fields[2].strip(), fields[3].strip(), fields[4].strip()
        if not sample_id or sample_id in identifiers:
            raise ValueError("Missing or duplicate TICO sample identifier")
        identifiers.add(sample_id)
        if not (3 <= len(english.split()) <= 70 and 3 <= len(foreign.split()) <= 70):
            continue
        if normalize(english) in en_seen or normalize(foreign) in src_seen:
            continue
        src_seen.add(normalize(foreign))
        en_seen.add(normalize(english))
        rank = hashlib.sha256(f"{seed}/{sample_id}".encode()).hexdigest()
        candidates.append((rank, sample_id, foreign, english, fields[5]))
    candidates.sort(key=lambda x: x[0])
    chosen = candidates[:count]
    if len(chosen) != count:
        raise ValueError("Insufficient unique out-of-domain TICO cases")
    digest = hashlib.sha256("\n".join(row[1] for row in chosen).encode()).hexdigest()
    rows = [(row[2], row[3]) for row in chosen]
    return rows, {"selected_ids_sha256": digest, "n": len(rows),
                  "source_group_count": len({r[4] for r in chosen})}


def benchmark(language: str, out: Path):
    if out.exists():
        raise ValueError("Refusing to overwrite a frozen research evaluation")
    if language not in DATA_FILES:
        raise ValueError("Unsupported language")
    name, _ = DATA_FILES[language]
    url = URL_ROOT + "/" + name
    with urlopen(Request(url, headers={"User-Agent": "BridgeCaption-research/1.0"}),
                 timeout=60) as response:
        payload = response.read(5 * 1024 * 1024 + 1)
    if len(payload) > 5 * 1024 * 1024:
        raise RuntimeError("Unexpectedly large TICO dataset")
    pairs, selection = select_rows(payload, language)
    source_digest = hashlib.sha256(payload).hexdigest()
    del payload

    from sacrebleu.metrics import CHRF, BLEU
    originals = [source for source, _ in pairs]
    references = [english for _, english in pairs]
    scores, adapter_hashes = {}, {}
    for variant in VARIANTS:
        # Real pinned model offline inference, one load per variant, 8-row batches.
        results = translate(language, variant, originals,
                            out.parent / "research-base-cache", REPO)
        generated = results["translations"]
        if len(generated) != len(references):
            raise RuntimeError("Missing model predictions")
        scores[variant] = {
            "chrf": CHRF().corpus_score(generated, [references]).score,
            "bleu": BLEU().corpus_score(generated, [references]).score,
            "reference_exact_match_fraction_not_adequacy": (
                sum(normalize(a) == normalize(b)
                    for a, b in zip(generated, references)) / len(references)),
            "n": len(references),
        }
        adapter_hashes[variant] = results["adapter_sha256"]
        print("TICO_EXTERNAL", language, variant, scores[variant], flush=True)

    out.mkdir(parents=True, exist_ok=False)
    report = {
        "language": language, "source": SOURCE_REPO, "revision": SOURCE_REV,
        "source_url": url, "source_archive_sha256": source_digest,
        "license": "TICO-19 project CC0, cite Anastasopoulos et al.",
        "frozen_test_selection": selection, "model_scores": scores,
        "adapter_sha256": adapter_hashes,
        "model_selection": "No model training or selection using TICO-19 evaluation",
        "limitations": (
            "Out-of-domain medical/crisis text, not dialogue or live speech. "
            "Possible original pretraining exposure. BLEU/chrF and exact-match "
            "are not percentages of correctly translated language. No real "
            "independent bilingual human adequacy ratings. This TICO-19 "
            "test selection is now consumed and must not be reused to tune."),
    }
    (out / "external_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("BENCHMARK_COMPLETE", language, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=tuple(DATA_FILES), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    benchmark(args.language, args.output_dir.resolve())
