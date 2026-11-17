#!/usr/bin/env python3
"""Explicit, research-only local French/Spanish translation using saved LoRA adapters.

Does not change the GUI, default providers, credentials, or production settings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from train import LANGUAGES, download_base


def build_command(repo_root: Path, language: str, model_dir: Path, text: str,
                  baseline: bool = False) -> list[str]:
    if language not in LANGUAGES or not text.strip():
        raise ValueError("Supported language and non-empty text are required")
    spec = LANGUAGES[language]
    research = repo_root / "research"
    args = [sys.executable, str(research / "replay_adapter.py"),
            "--experiment-dir", str(research / f"neural_continuation_20261116/{language}"),
            "--base-model", str(model_dir),
            "--base-manifest", str(research / spec["manifest_rel"]),
            "--text", text]
    if baseline:
        args.append("--baseline")
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--baseline", action="store_true", help="Compare original base model")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path.home() / ".cache/bridgecaption/research-models")
    args = parser.parse_args()
    if not args.text.strip() or len(args.text) > 5000:
        parser.error("Text must contain 1-5000 characters")
    repo = Path(__file__).resolve().parents[2]
    spec = LANGUAGES[args.language]
    model_dir = args.cache_dir / (args.language + "-" + spec["revision"][:12])
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_dir.is_symlink():
        parser.error("Model directory must not be a symlink")
    download_base(model_dir, repo / "research" / spec["manifest_rel"], spec)
    command = build_command(repo, args.language, model_dir, args.text, args.baseline)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=150,
                               check=False)
    if completed.returncode:
        parser.error("Pinned research replay failed: " + completed.stderr[-1500:])
    result = json.loads(completed.stdout)
    print(json.dumps({"language": args.language, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
