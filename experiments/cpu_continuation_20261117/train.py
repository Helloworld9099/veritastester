#!/usr/bin/env python3
"""Reproducible research-only French/Spanish LoRA continuations on a free CPU runner.

Never modifies the BridgeCaption application or promotes a candidate to production.
The Tatoeba/ManyThings corpus may overlap old research or pretrained data;
its development score is an internal screening signal, NOT independent evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import time
from urllib.request import Request, urlopen
import zipfile

LANGUAGES = {
    "french": {
        "model": "Helsinki-NLP/opus-mt-fr-en",
        "revision": "c4aed37b318c763fd177aa449b44e3b783cc6c02",
        "data_url": "https://www.manythings.org/anki/fra-eng.zip",
        "archive_name": "fra.txt", "use_safetensors": True,
        "adapter_rel": "neural_continuation_20261116/french/adapter_selected.safetensors",
        "manifest_rel": "translation_fr_training_20261116/model/download_manifest.json",
        "protocol_rel": "neural_continuation_20261116/french/protocol.json",
    },
    "spanish": {
        "model": "Helsinki-NLP/opus-mt-es-en",
        "revision": "c96e2c5399ebfae4fc43d9669556b9afa74bb69d",
        "data_url": "https://www.manythings.org/anki/spa-eng.zip",
        "archive_name": "spa.txt", "use_safetensors": False,
        "adapter_rel": "neural_continuation_20261116/spanish/adapter_selected.safetensors",
        "manifest_rel": "translation_training_20261116/model/download_manifest.json",
        "protocol_rel": "neural_continuation_20261116/spanish/protocol.json",
    },
}
SEED = 20261117
TOTAL_UPDATES = 96
CHECKPOINTS = (48, 96)
TRAIN_COUNT = TOTAL_UPDATES * 16
DEV_COUNT = 128
MIN_CHRF_GAIN = 0.25
# Optional, backwards-compatible research performance knobs. The defaults
# reproduce every previously published training experiment.
LEARNING_RATE = 2.5e-5
LENGTH_BUCKET_SIZE = 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def parse_pairs(data: bytes, *, seed: int = SEED,
                training_count: int = TRAIN_COUNT,
                dev_count: int = DEV_COUNT, archive_name: str = "fra.txt"):
    """Return source/target-disjoint source->English train/dev from public Tatoeba pairs."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [n for n in archive.namelist() if Path(n).name == archive_name]
        if len(names) != 1:
            raise ValueError("Expected exactly one " + archive_name + " in ManyThings archive")
        payload = archive.read(names[0]).decode("utf-8-sig")
    unique = []
    seen_src, seen_en = set(), set()
    for line in payload.splitlines():
        columns = line.split("\t")
        if len(columns) < 2:
            continue
        en, source = columns[0].strip(), columns[1].strip()
        if not (3 <= len(source.split()) <= 32 and 3 <= len(en.split()) <= 36):
            continue
        src_key, en_key = canonical(source), canonical(en)
        if not src_key or not en_key or src_key in seen_src or en_key in seen_en:
            continue
        seen_src.add(src_key)
        seen_en.add(en_key)
        unique.append((source, en))
    if len(unique) < training_count + dev_count:
        raise ValueError("Not enough unique source-English sentence pairs")
    random.Random(seed).shuffle(unique)
    # The selection order and count are locked by the committed experiment code.
    dev = unique[:dev_count]
    train = unique[dev_count:dev_count + training_count]
    assert not ({canonical(s) for s, _ in train} & {canonical(s) for s, _ in dev})
    assert not ({canonical(t) for _, t in train} & {canonical(t) for _, t in dev})
    return train, dev, len(unique)


def accepted_candidate(candidate: dict, baseline: dict, minimum: float = MIN_CHRF_GAIN) -> bool:
    """Conservative dev-only gate registered before the run."""
    return (candidate["chrf"] >= baseline["chrf"] + minimum
            and candidate["bleu"] >= baseline["bleu"])


def verified_warm_adapter(repo_root: Path, research: Path, spec: dict) -> Path:
    """Verify historical or versioned research checkpoint without rewriting either."""
    original = spec["adapter_rel"]
    override = spec.get("warm_adapter_override")
    if override is None:
        path = research / original
        checks = json.loads((research / "CHECKSUMS.json").read_text())["files"]
        expected = checks[original]
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError("Historical selected adapter does not match research checksums")
        return path
    if not isinstance(override, dict) or set(override) != {"path", "sha256"}:
        raise ValueError("Warm adapter override requires an exact path and SHA-256")
    relative = Path(override["path"])
    digest = override["sha256"]
    if (relative.is_absolute() or ".." in relative.parts
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("Unsafe checkpoint path or invalid SHA-256 pin")
    root = repo_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Checkpoint path is outside the repository or missing")
    if sha256(path) != digest:
        raise RuntimeError("Versioned warm adapter checkpoint SHA-256 mismatch")
    return path


def download_base(model_dir: Path, manifest: Path, spec: dict):
    """Download only the original pinned model files; reject hash mismatches."""
    from huggingface_hub import hf_hub_download
    model_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("Missing pinned base-model manifest")
    for row in records:
        name = row["file"]
        if Path(name).name != name:
            raise ValueError("Invalid pinned filename")
        print("DOWNLOAD BASE", name, flush=True)
        cache_path = Path(hf_hub_download(spec["model"], filename=name, revision=spec["revision"]))
        if cache_path.stat().st_size != row["size_bytes"] or sha256(cache_path) != row["sha256"]:
            raise RuntimeError("Downloaded model file checksum mismatch: " + name)
        target = model_dir / name
        if target.is_symlink():
            raise RuntimeError("Symlinked local model file is not allowed: " + name)
        if not target.is_file() or target.stat().st_size != row["size_bytes"] or sha256(target) != row["sha256"]:
            shutil.copyfile(cache_path, target)
        if target.stat().st_size != row["size_bytes"] or sha256(target) != row["sha256"]:
            raise RuntimeError("Pinned local model file checksum mismatch: " + name)
    print("VERIFIED PINNED BASE", len(records), "files", flush=True)


def train_experiment(repo_root: Path, output_dir: Path, language: str = "french") -> dict:
    import torch
    from torch import nn
    from safetensors.torch import load_file, save_file
    from sacrebleu.metrics import BLEU, CHRF
    from transformers import MarianMTModel, MarianTokenizer

    start_time = time.monotonic()
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    # PyTorch permits setting interop threads only before the first use.
    # v2 serially runs multiple full training continuations in one process.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if language not in LANGUAGES:
        raise ValueError("Unsupported language")
    spec = LANGUAGES[language]
    research = repo_root / "research"
    adapter_path = verified_warm_adapter(repo_root, research, spec)
    manifest = research / spec["manifest_rel"]

    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = output_dir / "pinned_base"
    download_base(model_dir, manifest, spec)

    print("DOWNLOADING SOURCE-ONLY TRAINING DATA", flush=True)
    max_data_bytes = int(spec.get("max_dataset_bytes", 30 * 1024 * 1024))
    if not 0 < max_data_bytes <= 150 * 1024 * 1024:
        raise ValueError("Invalid source-only corpus byte cap")
    with urlopen(Request(spec["data_url"], headers={"User-Agent": "BridgeCaption-research/1.0"}),
                 timeout=90) as stream:
        data = stream.read(max_data_bytes + 1)
    if len(data) > max_data_bytes:
        raise RuntimeError("Unexpectedly large corpus archive")
    corpus_digest = hashlib.sha256(data).hexdigest()
    train, dev, unique_count = parse_pairs(data, archive_name=spec["archive_name"])
    del data  # Don't distribute or save unreviewed third-party sentence text.
    print("PAIRS", len(train), "train", len(dev), "dev", unique_count, "unique", flush=True)

    tok = MarianTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = MarianMTModel.from_pretrained(str(model_dir), local_files_only=True,
                                         use_safetensors=spec["use_safetensors"], weights_only=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.scale = 2.0
            self.lora_A = nn.Parameter(torch.zeros(8, base.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, 8))

        def forward(self, x):
            return self.base(x) + ((x @ self.lora_A.T) @ self.lora_B.T) * self.scale

    protocol = json.loads((research / spec["protocol_rel"]).read_text())
    if (protocol["model"]["model"] != spec["model"]
            or protocol["model"]["revision"] != spec["revision"]):
        raise RuntimeError("Prior adapter model identity differs from pinned base")
    targets = protocol["target_modules"]
    if len(targets) != 36 or len(set(targets)) != 36:
        raise RuntimeError("Unexpected prior LoRA target list")
    for name in targets:
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear) or not name.endswith(("q_proj", "v_proj")):
            raise RuntimeError("Unexpected LoRA projection: " + name)
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, LoRALinear(module))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(p.numel() for p in trainable) != 294912:
        raise RuntimeError("Trainable parameter count differs from existing adapter")

    old_weights = load_file(str(adapter_path))
    model_state = model.state_dict()
    lora_keys = {key for key in model_state if ".lora_" in key}
    if set(old_weights) != lora_keys:
        raise RuntimeError("Adapter keys are incompatible")
    with torch.no_grad():
        for key, tensor in old_weights.items():
            if tensor.shape != model_state[key].shape or not torch.isfinite(tensor).all():
                raise RuntimeError("Invalid adapter tensor: " + key)
            model_state[key].copy_(tensor)

    def batch(rows):
        enc = tok([src for src, _ in rows], text_target=[tgt for _, tgt in rows],
                  padding=True, truncation=True, max_length=96, return_tensors="pt")
        enc["labels"][enc["labels"] == tok.pad_token_id] = -100
        return enc

    def score(rows):
        model.eval()
        preds = []
        with torch.no_grad():
            for offset in range(0, len(rows), 8):
                inputs = tok([src for src, _ in rows[offset:offset + 8]],
                             padding=True, truncation=True, max_length=128,
                             return_tensors="pt")
                seqs = model.generate(**inputs, num_beams=4, max_new_tokens=96,
                                      use_cache=True)
                preds.extend(tok.batch_decode(seqs, skip_special_tokens=True))
        refs = [tgt for _, tgt in rows]
        return {"chrf": CHRF().corpus_score(preds, [refs]).score,
                "bleu": BLEU().corpus_score(preds, [refs]).score,
                "n": len(rows)}

    report = {
        "experiment": "research-only CPU " + language + "->English LoRA continuation",
        "language": language, "seed": SEED, "base_model": spec["model"], "base_revision": spec["revision"],
        "warm_adapter_sha256": sha256(adapter_path),
        "corpus_source": spec["data_url"], "corpus_sha256": corpus_digest,
        "attribution": "Tatoeba " + language + "-English pairs via ManyThings.org; verify license and attribution before redistribution",
        "train_rows": len(train), "dev_rows": len(dev),
        "unique_pairs_after_filtering": unique_count,
        "warning": "The newly downloaded corpus may overlap earlier research and base pretraining. Development-only results do not establish generalization, production readiness, or translation improvement.",
        "planned_updates": TOTAL_UPDATES, "learning_rate": LEARNING_RATE,
        "length_bucket_size": LENGTH_BUCKET_SIZE,
        "selection": "Accept new checkpoint only if dev chrF improves >=0.25 AND dev BLEU does not fall, compared with unchanged prior adapter.",
        "scores": {}, "selected": "previous", "completed_updates": 0,
    }
    print("SCORING EXISTING ADAPTER", flush=True)
    report["scores"]["previous"] = score(dev)
    print("DEV previous", report["scores"]["previous"], flush=True)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    selected = "previous"
    best = report["scores"]["previous"]
    indices = list(range(len(train)))
    random.Random(SEED).shuffle(indices)
    # Group similarly sized examples within pre-shuffled local windows. This
    # reduces padded tokens without dropping, shortening, or duplicating any
    # training cases; opt-in only, all legacy results retain original order.
    if LENGTH_BUCKET_SIZE:
        if (LENGTH_BUCKET_SIZE < 16 or LENGTH_BUCKET_SIZE % 16
                or len(indices) % 16):
            raise ValueError("Length bucket must contain complete 16-case updates")
        lengths = [max(len(tok.encode(s, truncation=True, max_length=96)),
                       len(tok.encode(t, truncation=True, max_length=96)))
                   for s, t in train]
        indices = [item for p in range(0, len(indices), LENGTH_BUCKET_SIZE)
                   for item in sorted(indices[p:p + LENGTH_BUCKET_SIZE],
                                      key=lambda i: lengths[i])]
        print("LENGTH_BUCKET_TRAINING", LENGTH_BUCKET_SIZE,
              "sample_count", len(indices), flush=True)
    losses = []
    for step in range(1, TOTAL_UPDATES + 1):
        model.train()
        for micro in range(4):
            subset = [train[indices[(step - 1) * 16 + micro * 4 + j]] for j in range(4)]
            loss = model(**batch(subset)).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            (loss / 4).backward()
        grad_norm = float(nn.utils.clip_grad_norm_(trainable, 1.0))
        if not (0 <= grad_norm < float("inf")):
            raise RuntimeError("Invalid gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        report["completed_updates"] = step
        if step % 16 == 0:
            print("TRAIN", step, "/", TOTAL_UPDATES,
                  "last_loss", round(losses[-1], 5), flush=True)
        if step in CHECKPOINTS:
            path = output_dir / f"candidate_step{step}.safetensors"
            weights = {key: value.detach().cpu().contiguous().clone()
                       for key, value in model.state_dict().items() if key in lora_keys}
            save_file(weights, str(path))
            if not all(torch.isfinite(tensor).all() for tensor in weights.values()):
                raise RuntimeError("Nonfinite LoRA checkpoint")
            metrics = score(dev)
            report["scores"][f"step{step}"] = metrics
            print("DEV step", step, metrics, flush=True)
            if (accepted_candidate(metrics, report["scores"]["previous"])
                    and metrics["chrf"] > best["chrf"]):
                selected = f"step{step}"
                best = metrics
            model.config.use_cache = False

    report["selected"] = selected
    if selected != "previous":
        src = output_dir / f"candidate_{selected}.safetensors"
        shutil.copyfile(src, output_dir / "candidate_selected.safetensors")
        report["selected_candidate_sha256"] = sha256(output_dir / "candidate_selected.safetensors")
    report["last_step_loss"] = losses[-1]
    report["elapsed_seconds"] = time.monotonic() - start_time
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COMPLETE", json.dumps({"selected": selected,
                                  "completed_updates": report["completed_updates"],
                                  "scores": report["scores"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must not already exist (protect prior experiments)")
    train_experiment(args.repo_root.resolve(), args.output_dir.resolve(), args.language)


if __name__ == "__main__":
    main()
