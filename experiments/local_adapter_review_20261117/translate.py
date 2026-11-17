#!/usr/bin/env python3
"""Research-only, local BridgeCaption LoRA inference for bilingual human review.

Supports the original trained checkpoint, saved OPUS Books candidate, and
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

MODELS = ("base", "historical", "books")


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
    else:
        relative = LANGUAGES[language]["adapter_rel"]
        path = repo_root / "research" / relative
        checks = json.loads((repo_root / "research/CHECKSUMS.json").read_text())["files"]
        expected = checks[relative]["sha256"]
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise ValueError("Saved research adapter checksum mismatch")
    return path


def translate(language: str, variant: str, text: str, cache_dir: Path, repo_root: Path):
    if language not in LANGUAGES or variant not in MODELS:
        raise ValueError("Unsupported language/variant")
    if not isinstance(text, str) or not 0 < len(text.strip()) <= 5000:
        raise ValueError("Enter 1-5000 characters of non-empty source text")
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
    with torch.no_grad():
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
        outputs = model.generate(**inputs, num_beams=4, max_new_tokens=96)
        result = tok.decode(outputs[0], skip_special_tokens=True)
    return {
        "language": language,
        "variant": variant,
        "translation": result,
        "adapter_sha256": sha256(checkpoint) if checkpoint else None,
        "research_only": True,
        "limitations": "Context-free text inference; not independent quality validation or live audio.",
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--language", required=True, choices=tuple(LANGUAGES))
    cli.add_argument("--variant", choices=MODELS, required=True)
    cli.add_argument("--text", required=True)
    cli.add_argument("--cache-dir", type=Path,
                     default=Path.home() / ".cache/bridgecaption/research-models")
    opts = cli.parse_args()
    try:
        result = translate(opts.language, opts.variant, opts.text,
                           opts.cache_dir, REPO)
    except (ValueError, KeyError, OSError) as exc:
        cli.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
