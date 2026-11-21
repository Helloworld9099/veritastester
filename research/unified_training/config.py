"""Immutable, offline training metadata. No backend imports or execution."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import os
import stat
from types import MappingProxyType


class ConfigError(ValueError):
    """Invalid, incomplete, or unsafe training metadata."""


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
DIR = re.compile(r"([a-z]{2,3})->([a-z]{2,3})\Z")
OPAQUE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,79}\Z")
MODULE = re.compile(r"[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*\Z")
SEMVER = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:[-+][a-zA-Z0-9.-]+)?\Z")
REPO = re.compile(r"[a-zA-Z0-9_-][a-zA-Z0-9_.-]*/[a-zA-Z0-9_-][a-zA-Z0-9_.-]*\Z")
PATH_SEGMENT = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,119}\Z")
RIGHTS = frozenset({"commercial_candidate", "private_noncommercial"})
OPEN_LICENSES = frozenset({"CC0-1.0", "CC-BY-2.0-FR", "CC-BY-3.0", "CC-BY-4.0",
                          "CC-BY-SA-4.0", "Apache-2.0", "MIT"})
RESEARCH_LICENSES = frozenset({"CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"})
RESUME_COMPONENTS = frozenset({"optimizer", "scheduler", "rng", "sampler"})
CONFIG_KEYS = frozenset({"schema_version", "direction", "architecture", "backend", "base",
                         "device", "strategy", "optimizer", "schedule", "seed", "data",
                         "rights", "source", "output_id"})


def _require(condition, message):
    if not condition:
        raise ConfigError(message)


def _exact(obj, keys, label):
    _require(isinstance(obj, Mapping) and set(obj) == set(keys), f"Invalid {label} field set")
    return obj


def _text(value, pattern, label):
    _require(isinstance(value, str) and pattern.fullmatch(value), f"Invalid {label}")
    return value


def _one_of(value, choices, label):
    _require(isinstance(value, str) and value in choices, f"Invalid {label}")
    return value


def _integer(value, minimum, maximum, label):
    _require(type(value) is int and minimum <= value <= maximum, f"Invalid {label}")
    return value


def _number(value, lo, hi, label, *, lo_exclusive=False, hi_exclusive=False):
    _require(type(value) in (int,float) and lo <= value <= hi and math.isfinite(value)
             and (not lo_exclusive or value > lo) and (not hi_exclusive or value < hi),
             f"Invalid {label}")
    # Canonicalize exact same numeric values before fingerprinting.
    return float(value) or 0.0


def _sha(value, label):
    return _text(value, SHA256, label)


def _opaque(value, label):
    return _text(value, OPAQUE, label)


def _path(value, label):
    _require(isinstance(value, str) and len(value) <= 500
             and all(PATH_SEGMENT.fullmatch(part) for part in value.split("/")),
             f"Invalid relative {label}")
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key:_freeze(item) for key,item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key:_plain(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def canonical_sha256(mapping) -> str:
    """Fingerprint canonical UTF-8 JSON for binding receipts to exact metadata."""
    return hashlib.sha256(json.dumps(_plain(mapping), sort_keys=True, separators=(",",":"),
                                     allow_nan=False, ensure_ascii=False).encode("utf-8")).hexdigest()


def _base(raw, architecture):
    obj = _exact(raw, {"model_id", "revision", "files"}, "base")
    _text(obj["model_id"], REPO, "base model name")
    _text(obj["revision"], COMMIT, "immutable base revision")
    files = obj["files"]
    _require(isinstance(files, Mapping) and 1 <= len(files) <= 256, "Invalid base file manifest")
    for name, digest in files.items():
        _path(name, "base file name")
        _sha(digest, "base file digest")
    _require("config.json" in files and any(name.endswith((".safetensors",".bin")) for name in files),
             "Base configuration and weight pins required")
    tokenizer_pinned = ("tokenizer.json" in files or "tokenizer.model" in files or
                        {"source.spm","target.spm","vocab.json"} <= files.keys())
    if architecture == "marian":
        tokenizer_pinned = {"source.spm","target.spm","vocab.json"} <= files.keys()
    _require(tokenizer_pinned, "Pinned tokenizer identity required")
    return _plain(obj)


def _device(raw):
    obj = _exact(raw, {"kind", "gpu_opt_in", "precision", "threads"}, "device")
    kind = _one_of(obj["kind"], {"cpu", "gpu"}, "device kind")
    _require(type(obj["gpu_opt_in"]) is bool and obj["gpu_opt_in"] == (kind == "gpu"),
             "GPU requires explicit opt-in; CPU does not opt into GPU")
    _one_of(obj["precision"], {"fp32"} if kind == "cpu" else {"fp32","fp16","bf16"}, "device precision")
    _integer(obj["threads"], 1, 256, "device thread count")
    return _plain(obj)


def _strategy(raw, device):
    _require(isinstance(raw, Mapping), "Invalid training strategy")
    kind = _one_of(raw.get("kind"), {"lora", "qlora"}, "training strategy")
    keys = {"kind","rank","alpha","target_modules","protocol_sha256"}
    obj = _exact(raw, keys | ({"load_bits"} if kind == "qlora" else set()), "strategy")
    if kind == "qlora":
        _require(device["kind"] == "gpu" and device["gpu_opt_in"], "QLoRA requires explicit GPU opt-in")
        _require(type(obj["load_bits"]) is int and obj["load_bits"] == 4, "QLoRA must declare 4-bit load")
    rank = _integer(obj["rank"], 1, 1024, "LoRA rank")
    alpha = _number(obj["alpha"], 0, 4096, "LoRA alpha", lo_exclusive=True)
    targets = obj["target_modules"]
    _require(isinstance(targets,(list,tuple)) and 1 <= len(targets) <= 1024,
             "Pinned LoRA target module list required")
    for target in targets:
        _text(target, MODULE, "LoRA target module")
        _require(len(target) <= 200 and not any(part.startswith("_") for part in target.split(".")),
                 "Private/invalid module name in LoRA targets")
    _require(len(targets) == len(set(targets)), "Duplicate LoRA targets")
    _sha(obj["protocol_sha256"], "LoRA protocol digest")
    return {**_plain(obj), "rank":rank, "alpha":alpha}


def _optimizer(raw):
    obj = _exact(raw, {"name","learning_rate","weight_decay","beta1","beta2","epsilon"}, "optimizer")
    _one_of(obj["name"], {"adamw"}, "optimizer name")
    return {
        "name":obj["name"],
        "learning_rate":_number(obj["learning_rate"],0,1,"learning rate",lo_exclusive=True),
        "weight_decay":_number(obj["weight_decay"],0,1,"weight decay"),
        "beta1":_number(obj["beta1"],0,1,"beta1",hi_exclusive=True),
        "beta2":_number(obj["beta2"],0,1,"beta2",hi_exclusive=True),
        "epsilon":_number(obj["epsilon"],0,1,"optimizer epsilon",lo_exclusive=True),
    }


def _schedule(raw):
    bounds = {"total_updates":(1,100_000_000),"warmup_updates":(0,100_000_000),
              "micro_batch_size":(1,1_024),"gradient_accumulation_steps":(1,1_024),
              "max_source_tokens":(1,131_072),"max_target_tokens":(1,131_072)}
    obj = _exact(raw, {"kind"} | bounds.keys(), "schedule")
    _one_of(obj["kind"], {"constant","linear","cosine"}, "schedule kind")
    for key, (lo,hi) in bounds.items():
        _integer(obj[key],lo,hi,key)
    _require(obj["warmup_updates"] < obj["total_updates"], "Warmup must end before training schedule")
    return _plain(obj)


def _data(raw, rights):
    _require(isinstance(raw,(list,tuple)) and 1 <= len(raw) <= 1024, "Open data manifests required")
    data, identities, digest_set, paths = [], set(), set(), set()
    for row in raw:
        obj = _exact(row, {"dataset_id","version","manifest_sha256","split","visibility","license",
                           "local_path"}, "dataset reference")
        name = _opaque(obj["dataset_id"], "dataset ID")
        version = _opaque(obj["version"], "dataset version")
        split = _one_of(obj["split"], {"train","dev"}, "open dataset split; final/test/holdout forbidden")
        visibility = _one_of(obj["visibility"], {"public","private"}, "data visibility; sealed inputs forbidden")
        license_id = _one_of(obj["license"], OPEN_LICENSES | RESEARCH_LICENSES, "reviewable data license")
        if rights == "commercial_candidate":
            _require(license_id not in RESEARCH_LICENSES and visibility == "public",
                     "Restricted/private data cannot enter a commercial-candidate lineage")
        digest = _sha(obj["manifest_sha256"], "dataset manifest digest")
        path = _path(obj["local_path"], "data manifest path")
        _require(path.endswith(".json"), "Data pin must name a JSON manifest, not records")
        key = (name,version,split)
        _require(key not in identities and digest not in digest_set and path not in paths,
                 "Duplicate or reused data-manifest identity across train/dev")
        identities.add(key); digest_set.add(digest); paths.add(path)
        data.append(_plain(obj))
    _require(any(row["split"] == "train" for row in data), "At least one open train split required")
    return data


def _source(raw, rights, output_id):
    _require(isinstance(raw, Mapping), "Invalid checkpoint source")
    mode = _one_of(raw.get("mode"), {"base","warm_start","exact_resume"}, "checkpoint start mode")
    if mode == "base":
        return _plain(_exact(raw, {"mode"}, "base start"))
    keys = {"mode","artifact_id","manifest_sha256","weights_sha256","compatibility_sha256",
            "ancestry_sha256","rights"}
    if mode == "exact_resume":
        keys |= {"resume_fingerprint", "component_sha256"}
    obj = _exact(raw, keys, "checkpoint source")
    _opaque(obj["artifact_id"], "source checkpoint ID")
    _require(obj["artifact_id"] != output_id, "Output ID must not override the source")
    for key in ("manifest_sha256","weights_sha256","compatibility_sha256","ancestry_sha256"):
        _sha(obj[key], key)
    source_rights = _one_of(obj["rights"], RIGHTS, "source ancestry rights")
    _require(rights != "commercial_candidate" or source_rights == rights,
             "Noncommercial ancestry cannot warm-start a commercial candidate")
    if mode == "exact_resume":
        _require(source_rights == rights, "Exact resume cannot change lineage rights")
        _sha(obj["resume_fingerprint"], "resume identity")
        states = _exact(obj["component_sha256"], RESUME_COMPONENTS, "resume state digests")
        for name,digest in states.items():
            _sha(digest, name + " state digest")
    return _plain(obj)


@dataclass(frozen=True, init=False)
class TrainingConfig:
    """Checked structural metadata; construct via from_mapping or load_config."""

    schema_version: int
    direction: str
    architecture: str
    backend: Mapping
    base: Mapping
    device: Mapping
    strategy: Mapping
    optimizer: Mapping
    schedule: Mapping
    seed: int
    data: tuple = field(repr=False)
    rights: str
    source: Mapping = field(repr=False)
    output_id: str = field(repr=False)

    def __init__(self, raw: Mapping):
        raw = _exact(raw, CONFIG_KEYS, "training configuration")
        _require(type(raw["schema_version"]) is int and raw["schema_version"] == 1,
                 "Unsupported training config schema")
        direction = _text(raw["direction"], DIR, "explicit language direction")
        match = DIR.fullmatch(direction)
        _require(match[1] != match[2], "Source and target languages must differ")
        arch = _one_of(raw["architecture"], {"marian","causal_lm","seq2seq_lm"}, "architecture")
        backend = _exact(raw["backend"], {"id","version"}, "backend selector")
        _text(backend["id"], re.compile(r"[a-z][a-z0-9_]{0,47}\Z"), "backend identity")
        _text(backend["version"], SEMVER, "concrete backend version")
        device = _device(raw["device"])
        rights = _one_of(raw["rights"], RIGHTS, "training rights lane")
        output_id = _opaque(raw["output_id"], "new output artifact ID")
        validated = {
            "schema_version":1, "direction":direction, "architecture":arch,
            "backend":_plain(backend), "base":_base(raw["base"],arch), "device":device,
            "strategy":_strategy(raw["strategy"],device), "optimizer":_optimizer(raw["optimizer"]),
            "schedule":_schedule(raw["schedule"]), "seed":_integer(raw["seed"],0,2**32-1,"seed"),
            "data":_data(raw["data"],rights), "rights":rights,
            "source":_source(raw["source"],rights,output_id), "output_id":output_id,
        }
        for key,value in validated.items():
            object.__setattr__(self,key,_freeze(value))
        if self.source["mode"] != "base":
            _require(self.source["compatibility_sha256"] == self.compatibility_sha256,
                     "Source checkpoint is incompatible with direction/backend/base/LoRA")
        if self.source["mode"] == "exact_resume":
            _require(self.source["resume_fingerprint"] == self.resume_fingerprint,
                     "Exact-resume recipe, execution environment, data, or rights changed")

    @classmethod
    def from_mapping(cls, raw: Mapping) -> TrainingConfig:
        return cls(raw)

    def to_mapping(self) -> dict:
        return {key:_plain(getattr(self,key)) for key in sorted(CONFIG_KEYS)}

    def artifact_compatibility(self) -> dict:
        """Byte/format identity for warm-start checkpoint compatibility."""
        return {key:_plain(getattr(self,key)) for key in (
            "direction","architecture","backend","base","strategy")}

    @property
    def compatibility_sha256(self) -> str:
        return canonical_sha256(self.artifact_compatibility())

    @property
    def resume_fingerprint(self) -> str:
        """Exact training recipe, not a promise of backend determinism.

        Output/source locators and local data paths are excluded; immutable
        dataset digests, order, splits and rights are included.
        """
        rows = [{key:_plain(value) for key,value in row.items() if key != "local_path"}
                for row in self.data]
        recipe = {key:_plain(getattr(self,key)) for key in (
            "schema_version","device","optimizer","schedule","seed","rights")}
        return canonical_sha256({"compatibility":self.artifact_compatibility(), "data":rows,
                                 "recipe":recipe})


def _unique_keys(pairs):
    result = {}
    for key,value in pairs:
        _require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def load_config(path: str | Path) -> TrainingConfig:
    """Read a bounded JSON config only; do not open dataset/base/checkpoint paths."""
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), "Config must be a regular local file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        _require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Config must be a regular local file")
        raw = stream.read(4 * 1024 * 1024 + 1)
    _require(len(raw) <= 4 * 1024 * 1024, "Training config exceeds metadata size limit")
    try:
        return TrainingConfig.from_mapping(json.loads(raw.decode("utf-8"),object_pairs_hook=_unique_keys))
    except (UnicodeError,json.JSONDecodeError,RecursionError) as error:
        raise ConfigError("Malformed JSON training metadata") from error
