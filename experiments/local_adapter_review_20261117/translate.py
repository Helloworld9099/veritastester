#!/usr/bin/env python3
"""Research-only, local BridgeCaption LoRA inference for bilingual human review.

Supports saved OPUS Books, QED, GlobalVoices and MASSIVE candidates, and
unmodified pretrained base. Never touches the live caption provider or keys.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/cpu_continuation_20261117"))
from train import LANGUAGES, download_base, sha256  # noqa: E402

MODELS = ("base", "historical", "books", "massive", "massive2", "massive3",
          "qed", "globalvoices")

RESEARCH_VARIANTS = {
    "massive": {
        "directory": "massive_20261117",
        "evaluation": "massive_evaluation.json",
        "selected_field": "selected_on_dev",
        "frozen_field": "dev_selection_frozen_before_test",
    },
    "massive2": {
        "directory": "massive_stage2_20261117",
        "evaluation": "stage2_evaluation.json",
        "selected_field": "selected_based_on_dev_alone",
        "frozen_field": None,
    },
    "massive3": {
        "directory": "massive_stage3_20261117",
        "evaluation": "stage3_evaluation.json",
        "selected_field": "selected_using_dev_only",
        "frozen_field": None,
    },
    "qed": {
        "directory": "qed_subtitles_20261117",
        "evaluation": "qed_evaluation.json",
        "selected_field": "selected_on_dev_only",
        "frozen_field": "selection_frozen_before_test",
    },
    "globalvoices": {
        "directory": "globalvoices_20261117",
        "evaluation": "globalvoices_evaluation.json",
        "selected_field": "selected_on_dev_only",
        "frozen_field": "selection_frozen_before_test",
    },
}


def expected_saved_sha(language: str, variant: str, repo_root: Path = REPO) -> str:
    """Accept only selected, evidence-backed, checksum-verified research adapters."""
    if variant not in RESEARCH_VARIANTS or language not in LANGUAGES:
        raise ValueError("Unsupported saved adapter variant/language")
    info = RESEARCH_VARIANTS[variant]
    root = repo_root / "experiments" / info["directory"] / "saved" / language
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    evaluation = json.loads((root / info["evaluation"]).read_text(encoding="utf-8"))
    if (report["language"] != language or evaluation["language"] != language
            or report["selected"] == "previous"
            or report["selected"] != evaluation[info["selected_field"]]
            or (info["frozen_field"] is not None
                and evaluation.get(info["frozen_field"]) is not True)):
        raise ValueError("Model has no accepted research checkpoint for this language")
    if variant == "massive" and (evaluation.get("aligned_by_original_slurp_id") is not True
                                  or evaluation.get("revision") !=
                                  "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"):
        raise ValueError("MASSIVE SLURP original-cohort provenance mismatch")
    if variant == "massive2":
        previous = json.loads((repo_root /
            "experiments/massive_20261117/saved" / language /
            "massive_evaluation.json").read_text(encoding="utf-8"))
        if (report.get("stage") != 2 or evaluation.get("stage") != 2
                or report.get("original_stage1_ids_sha256") != previous["split_ids_sha256"]
                or evaluation.get("prior_stage_one_ids_sha256") != previous["split_ids_sha256"]
                or evaluation.get("new_cohorts_exclude_prior_train_dev_test_exact_text") is not True
                or report.get("fresh_cohort_ids_sha256") != evaluation.get("second_stage_ids_sha256")):
            raise ValueError("MASSIVE stage-two checkpoint or cohort provenance mismatch")
    if variant == "massive3":
        previous = json.loads((repo_root /
            "experiments/massive_stage2_20261117/saved" / language /
            "stage2_evaluation.json").read_text(encoding="utf-8"))
        if (report.get("stage") != 3 or evaluation.get("stage") != 3
                or report.get("no_previous_train_dev_or_test_text_reused") is not True
                or evaluation.get("no_previous_train_dev_or_test_text_reused") is not True
                or report.get("old_cohort_sha256", {}).get("old_stage2_ids_sha256")
                    != previous["second_stage_ids_sha256"]
                or report.get("new_cohort_ids_sha256") != evaluation.get("new_ids_sha256")
                or evaluation.get("stage2_warm_sha256") != report.get("previous_stage2_checkpoint_sha256")):
            raise ValueError("MASSIVE stage-three checkpoint or cohort provenance mismatch")
    row = next((line for line in (root / "SHA256SUMS.txt").read_text().splitlines()
                if line.endswith("  adapter_selected.safetensors")), None)
    if row is None:
        raise ValueError("No saved accepted adapter for this language")
    digest = row.split(" ", 1)[0]
    if digest != report["selected_candidate_sha256"]:
        raise ValueError("Research report and adapter SHA-256 manifest disagree")
    return digest


def expected_book_sha(language: str, repo_root: Path = REPO) -> str:
    if language not in LANGUAGES:
        raise ValueError("Unsupported language")
    root = repo_root / "experiments/opus_books_20261117/saved" / language
    record = json.loads((root / "report.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "books_evaluation.json").read_text(encoding="utf-8"))
    if (record["language"] != language or frozen["language"] != language
            or frozen["selected_on_dev"] != record["selected"]
            or not frozen["selection_frozen_before_test"]
            or record["selected"] == "previous"):
        raise ValueError("Saved Books research selection provenance mismatch")
    entry = next((line for line in (root / "SHA256SUMS.txt").read_text().splitlines()
                  if line.endswith("  adapter_selected.safetensors")), None)
    if entry is None:
        raise ValueError("No saved Book adapter hash")
    from_summary = entry.split(" ", 1)[0]
    if from_summary != record["selected_candidate_sha256"]:
        raise ValueError("Book research record and checksum disagree")
    return from_summary


def resolve_weights(repo_root: Path, language: str, variant: str):
    if variant not in MODELS or language not in LANGUAGES:
        raise ValueError("Unsupported model/language")
    if variant == "base":
        return None
    if variant == "books":
        root = repo_root / "experiments/opus_books_20261117/saved" / language
        path = root / "adapter_selected.safetensors"
        expected = expected_book_sha(language, repo_root)
    elif variant in RESEARCH_VARIANTS:
        info = RESEARCH_VARIANTS[variant]
        root = repo_root / "experiments" / info["directory"] / "saved" / language
        path = root / "adapter_selected.safetensors"
        expected = expected_saved_sha(language, variant, repo_root)
    else:
        relative = LANGUAGES[language]["adapter_rel"]
        path = repo_root / "research" / relative
        checks = json.loads((repo_root / "research/CHECKSUMS.json").read_text())["files"]
        expected = checks[relative]["sha256"]
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise ValueError("Saved research adapter checksum mismatch")
    return path


def validated_sources(text):
    """Return bounded source texts; batch mode loads the weights only once."""
    batch = not isinstance(text, str)
    items = [text] if not batch else text
    if (not isinstance(items, list) or not 0 < len(items) <= 256
            or any(not isinstance(x, str) or not 0 < len(x.strip()) <= 5000
                   for x in items)):
        raise ValueError("Provide 1-256 nonempty texts of at most 5000 characters each")
    return items, batch


def validated_rows(path: Path):
    """Read only the minimum permitted frozen evaluation fields."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    seen = set()
    if not 0 < len(rows) <= 256:
        raise ValueError("Batch must contain 1-256 source rows")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str):
            raise ValueError("Every input needs an explicit sample_id")
        if row["sample_id"] in seen or not row["sample_id"]:
            raise ValueError("Input sample IDs must be unique and nonempty")
        seen.add(row["sample_id"])
        if row.get("category") not in ("everyday", "idiom", "slang", "context", "technical"):
            raise ValueError("Missing independent human review category")
        digest = row.get("source_group_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("Missing source-group SHA-256")
    validated_sources([row.get("source_text") for row in rows])
    return rows


def translate(language: str, variant: str, text, cache_dir: Path, repo_root: Path):
    if language not in LANGUAGES or variant not in MODELS:
        raise ValueError("Unsupported language/variant")
    sources, batch_mode = validated_sources(text)
    spec = LANGUAGES[language]
    checkpoint = resolve_weights(repo_root, language, variant)
    model_dir = cache_dir / (language + "-" + spec["revision"][:12])
    if model_dir.is_symlink():
        raise ValueError("Symlinked model directory not permitted")
    # Downloader progress belongs on stderr, leaving stdout as a single JSON record.
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        download_base(model_dir, repo_root / "research" / spec["manifest_rel"], spec)

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    import torch
    from torch import nn
    from safetensors.torch import load_file
    from transformers import MarianMTModel, MarianTokenizer
    torch.set_num_threads(2)
    tok = MarianTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = MarianMTModel.from_pretrained(
        str(model_dir), local_files_only=True, weights_only=True,
        use_safetensors=spec["use_safetensors"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    class LoRA(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.lora_A = nn.Parameter(torch.zeros(8, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, 8))
        def forward(self, value):
            return self.base(value) + ((value @ self.lora_A.T) @ self.lora_B.T) * 2

    if checkpoint is not None:
        protocol = json.loads((repo_root / "research" / spec["protocol_rel"]).read_text())
        targets = protocol["target_modules"]
        if len(targets) != 36 or len(set(targets)) != 36:
            raise ValueError("Unexpected LoRA architecture")
        if (protocol["model"]["model"] != spec["model"]
                or protocol["model"]["revision"] != spec["revision"]):
            raise ValueError("LoRA model provenance mismatch")
        for name in targets:
            module = model.get_submodule(name)
            if not isinstance(module, nn.Linear):
                raise ValueError("Unexpected LoRA target")
            parent, key = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), key, LoRA(module))
        weights = load_file(str(checkpoint))
        state = model.state_dict()
        expected = {key for key in state if ".lora_" in key}
        if set(weights) != expected:
            raise ValueError("Adapter keys not compatible with model")
        with torch.no_grad():
            for key, weight in weights.items():
                if weight.shape != state[key].shape or not torch.isfinite(weight).all():
                    raise ValueError("Invalid research adapter weights")
                state[key].copy_(weight)
    model.eval()
    translations = []
    with torch.no_grad():
        for offset in range(0, len(sources), 8):
            chunk = sources[offset:offset + 8]
            inputs = tok(chunk, return_tensors="pt", padding=True,
                         truncation=True, max_length=128)
            generated = model.generate(**inputs, num_beams=4, max_new_tokens=96)
            translations.extend(tok.batch_decode(generated, skip_special_tokens=True))
    if len(translations) != len(sources) or not all(isinstance(t, str) and t.strip()
                                                    for t in translations):
        raise RuntimeError("Translation batch output size or quality mismatch")
    result = {
        "language": language,
        "variant": variant,
        "adapter_sha256": sha256(checkpoint) if checkpoint else None,
        "research_only": True,
        "limitations": "Text-only model, not independently verified 90% translation adequacy.",
    }
    if batch_mode:
        result["translations"] = translations
    else:
        result["translation"] = translations[0]
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--language", required=True, choices=tuple(LANGUAGES))
    cli.add_argument("--variant", choices=MODELS, required=True)
    text_input = cli.add_mutually_exclusive_group(required=True)
    text_input.add_argument("--text")
    text_input.add_argument("--input-jsonl", type=Path,
                            help="Private sample IDs/categories/sources, no reference translations")
    cli.add_argument("--output-jsonl", type=Path,
                     help="Private prediction file for blind human review; never upload")
    cli.add_argument("--cache-dir", type=Path,
                     default=Path.home() / ".cache/bridgecaption/research-models")
    opts = cli.parse_args()
    if bool(opts.input_jsonl) != bool(opts.output_jsonl):
        cli.error("--input-jsonl requires --output-jsonl and vice versa")
    if opts.output_jsonl and opts.output_jsonl.exists():
        cli.error("Refusing to overwrite an existing frozen review output")
    try:
        if opts.input_jsonl:
            rows = validated_rows(opts.input_jsonl)
            result = translate(opts.language, opts.variant,
                               [row["source_text"] for row in rows],
                               opts.cache_dir, REPO)
            if result["adapter_sha256"] is None:
                cli.error("A frozen custom LoRA adapter is required for blinded review")
            output = []
            for row, translation in zip(rows, result["translations"]):
                output.append({
                    "sample_id": row["sample_id"],
                    "language": opts.language,
                    "category": row["category"],
                    "source_group_sha256": row["source_group_sha256"],
                    "source_text": row["source_text"],
                    "candidate_translation": translation,
                })
            with opts.output_jsonl.open("x", encoding="utf-8") as writer:
                for row in output:
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("Frozen adapter SHA-256: " + result["adapter_sha256"], file=sys.stderr)
            print("Wrote " + str(len(output)) + " private predictions to " + str(opts.output_jsonl),
                  file=sys.stderr)
            return
        result = translate(opts.language, opts.variant, opts.text, opts.cache_dir, REPO)
    except (ValueError, KeyError, OSError) as exc:
        cli.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
