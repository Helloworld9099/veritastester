"""Offline audit utility for historical checkpoints; never used by the model loader."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


INVENTORY_PATH = Path(__file__).with_name("checkpoint_inventory.json")
ROUTES = frozenset(("es-en", "fr-en", "en-es"))
ADAPTER_ROUTES = ROUTES - {"en-es"}
CLASSES = frozenset((
    "catalog-research-only",
    "pinned-registry-reference-active-adapter",
    "pinned-registry-reference-rollback-adapter",
    "research-archive-noncommercial",
))
ALIAS_ROLES = frozenset((
    "candidate-step",
    "initial-untrained-adapter",
    "last-optimizer-adapter",
    "research-champion-filename-not-release",
    "research-selection-filename-not-release",
    "retained-warm-start-not-new-output",
    "warm-start",
))


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _safe_file_path(value):
    if not isinstance(value, str) or any(c in value for c in ("\\", "\x00", ":")):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.as_posix() == value
        and len(path.parts) >= 2
        and path.parts[0] in ("experiments", "research")
        and all(name not in (".", "..") for name in path.parts)
    )


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        _check(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_inventory(path=INVENTORY_PATH):
    """Read and validate the catalog; this neither searches for nor opens weights."""
    catalog = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
    validate_inventory(catalog)
    return catalog


def _check_receipt(receipt, refs):
    _check(
        isinstance(receipt, dict)
        and set(receipt) == {"ref", "path", "git_blob_sha1"}
        and receipt["ref"] in refs
        and _safe_file_path(receipt["path"])
        and _hex(receipt["git_blob_sha1"], 40),
        "invalid receipt",
    )


def validate_inventory(catalog):
    """Check integrity, graph consistency and research-only classifications offline."""
    _check(catalog.get("schema_version") == 1, "unknown checkpoint inventory schema")
    _check(catalog.get("repository") == "Helloworld9099/veritastester", "unexpected checkpoint inventory repository")
    arch = catalog["architecture"]
    _check(
        arch.get("id") == "marian-attention-lora-r8"
        and arch.get("format") == "safetensors"
        and arch.get("tensor_count") == 72
        and arch.get("dtype") == "F32"
        and arch.get("rank") == 8
        and arch.get("bytes_each") == 1_187_800
        and _hex(arch.get("tensor_schema_sha256"), 64),
        "unsupported architecture profile",
    )
    bases = catalog["bases"]
    _check(set(bases) == ROUTES, "expected explicit known route descriptions")
    for route, base in bases.items():
        _check(
            base["model"] == f"Helsinki-NLP/opus-mt-{route}"
            and _hex(base["revision"], 40)
            and base["adapter_payloads_in_catalog"] is (route in ADAPTER_ROUTES)
            and _safe_file_path(base["external_metadata"]),
            f"invalid base provenance for {route}",
        )
    refs, refsets, runs = (catalog[k] for k in ("refs", "refsets", "runs"))
    _check("main" in refs and all(_hex(v, 40) for v in refs.values()), "invalid immutable references")
    for rset in refsets.values():
        _check(rset and all(n in refs for n in rset) and len(set(rset)) == len(rset), "invalid refset")
    _check(catalog["artifact_columns"] == ["sha256", "route", "catalog_class", "aliases"], "invalid artifact columns")
    _check(catalog["alias_columns"] == ["path", "refset", "filename_role", "run_report_path"], "invalid alias columns")
    artifact_by_hash = {}
    object_by_path_and_ref = {}
    used_runs = set()
    used_sets = set()
    edges = {}
    for oid, (digest, route, classification, aliases) in catalog["artifacts"].items():
        _check(_hex(oid, 40) and _hex(digest, 64) and digest not in artifact_by_hash, "invalid or duplicate checkpoint identity")
        _check(route in ADAPTER_ROUTES and classification in CLASSES and aliases, "invalid checkpoint class/route or no source")
        artifact_by_hash[digest] = route
        edges[digest] = set()
        seen_paths = set()
        marks_nc = False
        for path, rset, role, run_path in aliases:
            _check(
                _safe_file_path(path) and path.endswith(".safetensors")
                and path not in seen_paths and rset in refsets and role in ALIAS_ROLES,
                f"invalid path/source for {oid}",
            )
            seen_paths.add(path)
            used_sets.add(rset)
            _check(run_path in runs and runs[run_path]["route"] == route, f"unknown or cross-route run for {oid}")
            used_runs.add(run_path)
            run = runs[run_path]
            for ref in refsets[rset]:
                k = (path, ref)
                _check(k not in object_by_path_and_ref, f"multiple blobs at {path} in {ref}")
                object_by_path_and_ref[k] = oid
            marks_nc |= run["noncommercial_archive"] is True
            parent = run["parent_sha256"]
            if role in ("warm-start", "retained-warm-start-not-new-output"):
                _check(parent == digest, f"warm-start alias has different identity for {oid}")
            else:
                _check(parent != digest, f"output claims itself as ancestry for {oid}")
                if parent is not None:
                    edges[digest].add(parent)
        _check(marks_nc is (classification == "research-archive-noncommercial"), f"noncommercial archive misclassified for {oid}")
    _check(used_runs == set(runs) and used_sets == set(refsets), "unused metadata records")
    for path, run in runs.items():
        _check(isinstance(run["source_family"], str) and bool(run["source_family"]), "missing source family")
        _check(run["receipt"]["path"] == path, "receipt path does not match run key")
        _check_receipt(run["receipt"], refs)
        if run["further_data_receipt"] is not None:
            _check_receipt(run["further_data_receipt"], refs)
        data = run["data_sha256"]
        _check(data is None or _hex(data, 64), "bad data SHA-256")
        parent, state = run["parent_sha256"], run["parent_state"]
        comparator = run.get("comparator_sha256_not_ancestor")
        if comparator is not None:
            _check(
                _hex(comparator, 64) and artifact_by_hash.get(comparator) == run["route"]
                and parent is None and state == "base-start-comparator-not-ancestor",
                "comparator incorrectly claimed a training ancestor",
            )
        if parent is None:
            _check(state in ("base-start-no-parent-reported", "parent-not-reported")
                   or (state == "base-start-comparator-not-ancestor" and comparator is not None),
                   "unreported parent status mismatch")
        else:
            _check(_hex(parent, 64), "bad parent SHA-256")
            if parent in artifact_by_hash:
                _check(state == "indexed" and artifact_by_hash[parent] == run["route"], "cross-route or mislabeled parent")
            else:
                _check(state == "reported-parent-not-in-scope", "missing parent incorrectly claimed indexed")
    visiting, visited = set(), set()

    def visit(digest):
        _check(digest not in visiting, "cyclic checkpoint ancestry")
        if digest in visited or digest not in edges:
            return
        visiting.add(digest)
        for parent in edges[digest]:
            visit(parent)
        visiting.remove(digest)
        visited.add(digest)

    for digest in edges:
        visit(digest)
    for receipt in catalog["archive_receipts_not_in_git"]:
        _check(
            receipt["route"] in ADAPTER_ROUTES and receipt["ref"] in refs
            and _hex(receipt["archive_sha256"], 64) and receipt["payload_verified"] is False,
            "unverified off-tree archive must remain metadata-only",
        )


def verify_checkout(catalog, directory, ref):
    """Hash already-present files for one pinned ref; never download or load a model."""
    _check(ref in catalog["refs"], "unknown checkpoint inventory ref")
    root = Path(directory).resolve(strict=True)
    found, missing, mismatched = 0, [], []
    expected_size = catalog["architecture"]["bytes_each"]
    for oid, (digest, _route, _classification, aliases) in catalog["artifacts"].items():
        for relative, rset, _role, _run in aliases:
            if ref not in catalog["refsets"][rset]:
                continue
            path = root / relative
            if not path.is_file():
                missing.append(relative)
                continue
            if not path.resolve().is_relative_to(root) or path.stat().st_size != expected_size:
                mismatched.append(relative)
                continue
            sha256 = hashlib.sha256()
            git_sha1 = hashlib.sha1(f"blob {expected_size}\0".encode(), usedforsecurity=False)
            size = 0
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    size += len(block)
                    sha256.update(block)
                    git_sha1.update(block)
            if size == expected_size and sha256.hexdigest() == digest and git_sha1.hexdigest() == oid:
                found += 1
            else:
                mismatched.append(relative)
    return {"ref": ref, "verified_paths": found, "missing_paths": sorted(missing), "mismatched_paths": sorted(mismatched)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-dir", type=Path, help="optional existing checkout; no fetching")
    parser.add_argument("--ref", help="catalog ref label for --verify-dir")
    args = parser.parse_args(argv)
    if (args.verify_dir is None) != (args.ref is None):
        parser.error("--verify-dir and --ref must be supplied together")
    catalog = read_inventory()
    print(json.dumps({"objects": len(catalog["artifacts"]), "classes": dict(Counter(a[2] for a in catalog["artifacts"].values())), "refs": catalog["refs"]}, indent=2))
    if args.verify_dir is not None:
        result = verify_checkout(catalog, args.verify_dir, args.ref)
        print(json.dumps(result, indent=2))
        return 1 if result["missing_paths"] or result["mismatched_paths"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
