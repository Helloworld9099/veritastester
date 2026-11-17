#!/usr/bin/env python3
"""Bounded, research-only OPUS Books adaptation with a frozen document-level test.

The original selected model always remains intact; the new adapter is never
promoted to the live BridgeCaption app.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/cpu_continuation_20261117"))
import train as existing_train  # noqa: E402
from books import BOOKS, SEED, split_books, read_parallel  # noqa: E402


def experiment(language: str, output_dir: Path):
    if language not in BOOKS:
        raise ValueError("Unsupported language")
    if output_dir.exists():
        raise ValueError("Refusing to overwrite an existing experiment output folder")

    spec = BOOKS[language]
    original = existing_train.LANGUAGES[language].copy()
    existing_train.LANGUAGES[language] = {
        **original, "data_url": spec["url"],
        "archive_name": "Books." + spec["pair"],
    }
    existing_train.SEED = SEED
    existing_train.TOTAL_UPDATES = 192
    existing_train.TRAIN_COUNT = 192 * 16
    existing_train.CHECKPOINTS = (48, 96, 144, 192)
    # Exclude source book groups from all fit/development/test parts.
    split_holder = {}

    def parse(data: bytes, *, archive_name: str):
        if archive_name != "Books." + spec["pair"]:
            raise ValueError("Unexpected OPUS Books corpus selected")
        grouped = read_parallel(data, language)
        split = split_books(grouped, seed=SEED,
                            train_count=existing_train.TRAIN_COUNT,
                            dev_count=128, test_count=256)
        split_holder.update(split)
        return split["train"], split["dev"], sum(map(len, grouped.values()))

    existing_train.parse_pairs = parse
    report = existing_train.train_experiment(REPO, output_dir, language)
    report["attribution"] = ("OPUS Books v1; aligned bilingual public-domain-era "
                             "books by Andras Farkas, distributed by OPUS / Jorg Tiedemann; "
                             "research and educational use only; mass redistribution restricted.")
    report["warning"] = (
        "Document-disjoint OPUS Books development, but possible base-pretraining overlap. "
        "No independent bilingual human-quality evidence; never use heldout test "
        "for checkpoint selection.")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not split_holder:
        raise RuntimeError("Experiment split was not recorded")

    # The development gate and selected adapter were frozen inside
    # train_experiment before the test group was first passed to a model.
    test_results = evaluate_frozen_test(language, output_dir, split_holder["test"],
                                        report["selected"])
    final = {
        "language": language,
        "research_only": True,
        "training_data": "OPUS Books v1 (Andras Farkas bilingual books)",
        "domain": "literary, not live conversational speech",
        "books_groups": split_holder["book_groups_total"],
        "heldout_book_groups_hashes": split_holder["doc_group_sha256"],
        "selection_frozen_before_test": True,
        "previous_adapter_preserved": True,
        "train_steps": report["completed_updates"],
        "dev": report["scores"],
        "selected_on_dev": report["selected"],
        "independent_book_groups_test": test_results,
        "evaluation_limitations": (
            "Source document separation protects against within-Books document reuse in "
            "this experiment, but pretrained OPUS models may have seen these literary "
            "sentences. Literary chrF/BLEU or reference-exact match is NOT 90% accuracy. "
            "This cohort is consumed after the frozen test. No human adequacy ratings."),
    }
    (output_dir / "books_evaluation.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8")
    print("FROZEN BOOK-HELDOUT RESULTS", json.dumps(final, indent=2), flush=True)


def evaluate_frozen_test(language: str, output_dir: Path, test_rows, selected: str):
    import torch
    from torch import nn
    from sacrebleu.metrics import BLEU, CHRF
    from safetensors.torch import load_file
    from transformers import MarianMTModel, MarianTokenizer

    model_dir = output_dir / "pinned_base"
    info = existing_train.LANGUAGES[language]
    tok = MarianTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = MarianMTModel.from_pretrained(
        str(model_dir), local_files_only=True, weights_only=True,
        use_safetensors=info["use_safetensors"])
    for par in model.parameters():
        par.requires_grad_(False)

    class LoRA(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.lora_A = nn.Parameter(torch.zeros(8, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, 8))
        def forward(self, x):
            return self.base(x) + ((x @ self.lora_A.T) @ self.lora_B.T) * 2.0

    protocol = json.loads((REPO / "research" / info["protocol_rel"]).read_text())
    if protocol["model"]["model"] != info["model"]:
        raise ValueError("Selected LoRA protocol does not match base")
    for name in protocol["target_modules"]:
        child = model.get_submodule(name)
        if not isinstance(child, nn.Linear):
            raise ValueError("Unexpected LoRA target: " + name)
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, LoRA(child))

    state = model.state_dict()
    expected = {key for key in state if ".lora_" in key}
    original_adapter = REPO / "research" / info["adapter_rel"]
    candidates = {"base": None, "previous": original_adapter}
    if selected != "previous":
        candidates["selected"] = output_dir / "candidate_selected.safetensors"

    def use_adapter(path: Path | None):
        weights = load_file(str(path)) if path else {}
        if path and set(weights) != expected:
            raise ValueError("Adapter key mismatch during frozen test")
        with torch.no_grad():
            for key in expected:
                target = state[key]
                value = weights.get(key)
                if value is None:
                    target.zero_()
                else:
                    if value.shape != target.shape or not torch.isfinite(value).all():
                        raise ValueError("Nonfinite or incompatible LoRA adapter")
                    target.copy_(value)

    references = [reference for _, reference in test_rows]
    scores = {}
    for arm, path in candidates.items():
        use_adapter(path)
        model.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(test_rows), 8):
                block = test_rows[i:i + 8]
                input_tensors = tok([source for source, _ in block],
                                    return_tensors="pt", padding=True,
                                    truncation=True, max_length=128)
                translated = model.generate(**input_tensors, num_beams=4,
                                            max_new_tokens=96)
                outputs.extend(tok.batch_decode(translated, skip_special_tokens=True))
        metrics = {
            "n": len(references),
            "chrf": CHRF().corpus_score(outputs, [references]).score,
            "bleu": BLEU().corpus_score(outputs, [references]).score,
            "reference_exact_match_fraction_not_adequacy": (
                sum(existing_train.canonical(out) == existing_train.canonical(ref)
                    for out, ref in zip(outputs, references)) / len(references)),
        }
        scores[arm] = metrics
        print("HELDOUT BOOKS", language, arm, metrics, flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(BOOKS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    opts = parser.parse_args()
    experiment(opts.language, opts.output_dir.resolve())


if __name__ == "__main__":
    main()
