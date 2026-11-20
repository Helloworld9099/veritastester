#!/usr/bin/env python3
"""One-time, independent OUT-OF-DOMAIN Spanish→English generalization diagnostic.

Use the public FLORES-200 *devtest* benchmark, not any video subtitle cohort
used in optimization/development. Compare PRESELECTED Spanish sweep champion
to original stage-one context checkpoint. No model fitting, model selection,
or reuse of these test examples is permitted. FLORES is published news,
not conversational slang, and chrF/BLEU are NOT 95% semantic accuracy.
Original FLORES-200 is CC BY-SA 4.0; DGME is a distribution mirror.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments" / p) for p in
                ("cpu_continuation_20261117", "opus_books_20261117")]
import train as core
from train_books import evaluate_frozen_test

SOURCE_REPO = "DGME/FLORES-200"
SOURCE_REV = "3cb4c84"
SOURCE_BASE = f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{SOURCE_REV}/devtest/"
FILES = {"spanish": "spa_Latn.devtest", "english": "eng_Latn.devtest"}
SAMPLE_N = 256
SEED = 2026112051
SAVED = ROOT / "experiments/spanish_fifteen_rounds_20261120/saved"


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def download_lines(name: str) -> tuple[list[str], str]:
    url = SOURCE_BASE + FILES[name]
    req = Request(url, headers={"User-Agent": "BridgeCaption-independent-research-eval/1.0"})
    with urlopen(req, timeout=60) as res:
        data = res.read(400_000)
    if not data or len(data) >= 400_000:
        raise RuntimeError("FLORES source download failed length bound")
    rows = data.decode("utf-8-sig").splitlines()
    if len(rows) != 1012 or any(not row.strip() for row in rows):
        raise RuntimeError("Unexpected official FLORES-200 devtest shape")
    return rows, sha256(data)


def frozen_test():
    es, es_hash = download_lines("spanish")
    en, en_hash = download_lines("english")
    ids = list(range(len(es)))
    random.Random(SEED).shuffle(ids)
    chosen = sorted(ids[:SAMPLE_N])
    data = [(es[i], en[i]) for i in chosen]
    if len(set(s for s, _ in data)) != SAMPLE_N:
        raise RuntimeError("Duplicated benchmark source")
    if len(set(t for _, t in data)) != SAMPLE_N:
        raise RuntimeError("Duplicated benchmark reference")
    manifest = {
        "publisher": "FLORES-200 (Meta); DGME public mirror",
        "publisher_license": "CC BY-SA 4.0",
        "mirror_repo": SOURCE_REPO, "mirror_revision": SOURCE_REV,
        "official_split": "devtest", "language_direction": "spa_Latn -> eng_Latn",
        "source_sha256": es_hash, "reference_sha256": en_hash,
        "selection_seed": SEED, "rows_evaluated": len(data),
        "chosen_row_ids_sha256": sha256(",".join(map(str, chosen)).encode()),
        "pair_sha256": sha256("\n".join(s + "\0" + e for s, e in data).encode()),
        "no_training_or_dev_model_selection_using_this_flores_split": True,
        "not_slang_or_dialogue": True,
        "original_marian_upstream_pretraining_overlap_not_known": True,
    }
    return data, manifest


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError("Never overwrite or repeat a consumed external benchmark")
    summary = json.loads((SAVED / "summary.json").read_text())
    if summary["number_of_completed_full_rounds"] != 15:
        raise RuntimeError("Preselected champion has not completed all 15 full rounds")
    champion = SAVED / "champion_adapter.safetensors"
    baseline = SAVED / "round-15/adapter_selected.safetensors"
    expected = summary["winner"]["adapter_sha256"]
    if core.sha256(champion) != expected:
        raise RuntimeError("FROZEN champion identity mismatch")
    if core.sha256(baseline) != summary["winner"]["warm_sha256"]:
        raise RuntimeError("FROZEN baseline identity mismatch")
    if summary["human_95_percent_semantic_accuracy_verified"] is not False:
        raise RuntimeError("An unearned 95% accuracy claim appeared")
    test, dataset = frozen_test()
    # The winner was frozen BEFORE test download, no conditions allow altering
    # champion selection in response to any FLORES result.
    print("FLORES_FRESH_FROZEN_TEST_MANIFEST", json.dumps(dataset), flush=True)
    with tempfile.TemporaryDirectory(prefix="BridgeCaption-flores-first-and-only-") as temporary:
        pinned = Path(temporary) / "pinned_base"
        spec = core.LANGUAGES["spanish"]
        core.download_base(pinned, ROOT / "research" / spec["manifest_rel"], spec)
        working = Path(temporary)
        prior = evaluate_frozen_test("spanish", working, test, "previous",
                                     previous_adapter=baseline)
        champ = evaluate_frozen_test("spanish", working, test, "previous",
                                     previous_adapter=champion)
    if prior["base"] != champ["base"]:
        raise RuntimeError("Control translation scores changed between frozen evaluations")
    report = {
        "test_used_once_only": True, "test_used_for_training": False,
        "test_used_for_checkpoint_selection": False,
        "frozen_winner_selected_on_dev_only": summary["winner"]["round"],
        "model_base_identity": spec["model"],
        "base_sha256": None,
        "starting_spanish_adapter_sha256": core.sha256(baseline),
        "champion_spanish_adapter_sha256": core.sha256(champion),
        "dataset": dataset,
        "scores": {
            "unadapted_marian_base": prior["base"],
            "first_spanish_context_checkpoint": prior["previous"],
            "winner_after_15_rounds": champ["previous"],
        },
        "human_95_percent_semantic_accuracy_verified": False,
        "commercial_use_allowed": False,
        "limitations": "Published FLORES devtest is an external broad-domain diagnostic "
                       "not natural spoken dialogue, context or slang; source model may "
                       "have seen it upstream. chrF and BLEU are not human meaning accuracy."
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "flores_external_evaluation.json").write_text(json.dumps(report, indent=2))
    print("FROZEN_FLORES_ONCE_ONLY_RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, required=True)
    run(cli.parse_args().output_dir.resolve())
