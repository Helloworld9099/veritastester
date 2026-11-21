"""Offline, research-only conversion and split audit for BridgeCaption datasets.

Explicitly map source columns and supply rights/provenance. This deliberately
does not download, repair, normalize, rebalance, publish, or train on corpora.
Only the *comparison keys* used by the split audit normalize text. Keep bundles
with private material (including test/holdout text) out of version control.
Never give this converter, or the training process, raw sealed final holdout
data. An unsealed development/test bundle can be checked for split overlap;
the final custodian must run the separate sealed split check in their boundary.

Example (all paths here are local; pass a separately trusted manifest digest
for each version rather than treating a matching digest as authentication):

    spec = DatasetSpec("demo", "r1", "example provenance", "CC-BY-4.0",
                       rights="private_research", rights_evidence="review note",
                       source_language="zh", target_language="en")
    columns = {"original_id": "id", "source": "text",
               "target": "translation", "split": "original_role"}
    manifest = convert_dataset("example.jsonl", "./demo-r1", spec, columns)
    bundle = load_bundle("./demo-r1", expected_manifest_sha256=trusted_digest)
    rows = training_rows([bundle], purpose="private_research",
                         expected_manifest_sha256s=[trusted_digest])

The manifest records the exact input and converted file digests, conversion
parameters, aggregate counts, and reviewed source, license, and rights claims.
File digests are integrity evidence, not legal verification or signatures.
This v1 tool materializes the corpus in memory; for large inputs use small
versioned shards and audit all applicable unsealed bundles together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
RECORDS_NAME = "records.jsonl"
MANIFEST_NAME = "manifest.json"
SPLITS = frozenset(("train", "dev", "test", "holdout"))
RIGHTS = frozenset(("unknown", "prohibited", "private_research", "commercial"))
PRIVACY = frozenset(("public", "restricted", "private"))
PURPOSES = frozenset(("private_research", "commercial"))
_COLUMN_FIELDS = frozenset((
    "original_id", "source", "target", "source_language", "target_language",
    "split", "context", "group_id", "slang", "context_required", "holdout", "tags",
    "speaker_id", "dialect", "idiom", "slang_category", "noisy_transcription",
    "source_basis", "verification_status",
))
_RECORD = _COLUMN_FIELDS | frozenset(("schema_version", "dataset_id", "dataset_version"))


class DatasetError(ValueError):
    """Invalid data, split leakage or unsupported rights; never includes raw text."""


def _text(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value.strip()
            or "\x00" in value or any(0xD800 <= ord(c) <= 0xDFFF for c in value)):
        raise DatasetError(f"{label} must be nonempty Unicode text")
    return value


def _identifier(value: Any, label: str) -> str | int:
    if type(value) is int:
        return value
    return _text(value, label)


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise DatasetError(f"{label} must be a stable portable identifier")
    return value


def _language(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", value):
        raise DatasetError(f"{label} must be a BCP 47-style language tag")
    return value


@dataclass(frozen=True)
class DatasetSpec:
    """Caller-reviewed provenance and permissions, not a license interpreter."""

    dataset_id: str
    version: str
    source_uri: str
    license: str
    rights: str = "unknown"
    privacy: str = "private"
    rights_evidence: str | None = None
    source_language: str | None = None
    target_language: str | None = None

    def __post_init__(self) -> None:
        _slug(self.dataset_id, "dataset_id")
        _slug(self.version, "version")
        _text(self.source_uri, "source_uri")
        _text(self.license, "license")
        if not isinstance(self.rights, str) or self.rights not in RIGHTS:
            raise DatasetError("unknown rights policy")
        if not isinstance(self.privacy, str) or self.privacy not in PRIVACY:
            raise DatasetError("unknown privacy policy")
        if self.rights_evidence is not None:
            _text(self.rights_evidence, "rights_evidence")
        if self.rights in PURPOSES and (
            not self.rights_evidence
            or self.license.strip().casefold() in {"unknown", "unverified", "n/a", "unlicensed", "none", "no license"}
        ):
            raise DatasetError("explicit training permission requires a known license and rights evidence")
        rights_label = self.license.casefold()
        if self.rights == "commercial" and (
            re.search(r"(?:^|[^a-z])(?:nc|nd)(?:$|[^a-z])", rights_label)
            or re.search(r"non[ -]?commercial|no[ -]?derivat", rights_label)
        ):
            raise DatasetError("commercial training conflicts with the documented restriction")
        for field in ("source_language", "target_language"):
            value = getattr(self, field)
            if value is not None:
                _language(value, field)


def _json(raw: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DatasetError("duplicate JSON field")
            result[key] = value
        return result

    def bad_number(_: str) -> Any:
        raise DatasetError("non-finite JSON number")

    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=bad_number)


def _canon_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_text(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _source_rows(raw: bytes, fmt: str) -> Iterable[dict[str, Any]]:
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DatasetError("source file must be UTF-8") from None
    if fmt == "jsonl":
        for number, line in enumerate(io.StringIO(decoded), 1):
            try:
                value = _json(line)
            except (ValueError, RecursionError):
                raise DatasetError(f"source line {number}: invalid JSON object") from None
            if not isinstance(value, dict):
                raise DatasetError(f"source line {number}: invalid JSON object")
            yield value
    elif fmt in {"csv", "tsv"}:
        try:
            reader = csv.DictReader(io.StringIO(decoded, newline=""), delimiter="\t" if fmt == "tsv" else ",", strict=True)
            fields = reader.fieldnames
            if not fields or len(set(fields)) != len(fields) or any(not x for x in fields):
                raise DatasetError("invalid or duplicate source column names")
            for n, row in enumerate(reader, 1):
                if None in row or any(value is None for value in row.values()):
                    raise DatasetError(f"source record {n}: column count mismatch")
                yield row
        except csv.Error:
            raise DatasetError("malformed delimited source") from None
    else:
        raise DatasetError("input format must be jsonl, csv, or tsv")


def _validate_columns(columns: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(columns, Mapping):
        raise DatasetError("column mapping must be an object")
    if not {"original_id", "source", "split"} <= set(columns) or set(columns) - _COLUMN_FIELDS:
        raise DatasetError("map original_id, source, and split explicitly; unknown canonical column")
    if not all(isinstance(x, str) and x.strip() for x in columns.values()):
        raise DatasetError("mapped source column names must be nonempty strings")
    if len(set(columns.values())) != len(columns):
        raise DatasetError("one source column cannot supply multiple canonical fields")
    return dict(columns)


def _flag(value: Any, label: str, *, from_delimited: bool) -> bool:
    if type(value) is bool:
        return value
    if from_delimited and value in ("true", "false"):
        return value == "true"
    raise DatasetError(f"{label} must be a boolean (true/false in delimited input)")


def _validate_record(record: Mapping[str, Any], spec: DatasetSpec) -> None:
    if set(record) != _RECORD or type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
        raise DatasetError("unknown or malformed dataset-record schema")
    if record["dataset_id"] != spec.dataset_id or record["dataset_version"] != spec.version:
        raise DatasetError("record dataset identity does not match the manifest")
    _identifier(record["original_id"], "original_id")
    _text(record["source"], "source")
    if record["target"] is not None:
        _text(record["target"], "target")
    _language(record["source_language"], "source_language")
    _language(record["target_language"], "target_language")
    for field in ("source_language", "target_language"):
        if getattr(spec, field) is not None and record[field] != getattr(spec, field):
            raise DatasetError(f"{field} conflicts with the release declaration")
    role = record["split"]
    if not isinstance(role, str) or role not in SPLITS:
        raise DatasetError("split must be explicit: train, dev, test, or holdout")
    for field in ("holdout", "context_required"):
        _flag(record[field], field, from_delimited=False)
    for field in ("slang", "idiom", "noisy_transcription"):
        if record[field] is not None:
            _flag(record[field], field, from_delimited=False)
    if record["holdout"] != (role == "holdout"):
        raise DatasetError("holdout flag must match the holdout split")
    if role == "train" and record["target"] is None:
        raise DatasetError("training records must have a target")
    for field in ("group_id", "speaker_id"):
        if record[field] is not None:
            _identifier(record[field], field)
    for field in ("dialect", "slang_category"):
        if record[field] is not None:
            _text(record[field], field)
    for field, labels in (
        ("source_basis", ("authentic", "synthetic", "mixed")),
        ("verification_status", ("unreviewed", "single_review", "independent_review", "human_verified")),
    ):
        if record[field] is not None and record[field] not in labels:
            raise DatasetError(f"unknown {field}")
    context = record["context"]
    if not isinstance(context, list):
        raise DatasetError("context must be a list of ordered source-language turns")
    for turn in context:
        _text(turn, "context turn")
    if context and record["group_id"] is None:
        raise DatasetError("context records require a stable group_id")
    if record["context_required"] and not context:
        raise DatasetError("context-required records must include prior turns")
    if not isinstance(record["tags"], list):
        raise DatasetError("tags must be a list")
    for tag in record["tags"]:
        _text(tag, "tag")
    if len(set(record["tags"])) != len(record["tags"]):
        raise DatasetError("duplicate tags")


def _convert_rows(raw: bytes, fmt: str, spec: DatasetSpec, columns: Mapping[str, str]) -> list[dict[str, Any]]:
    columns = _validate_columns(columns)
    records = []
    for number, item in enumerate(_source_rows(raw, fmt), 1):
        try:
            if set(columns.values()) - set(item):
                raise DatasetError("mapped source value missing")
            values = {field: item[source] for field, source in columns.items()}
            if fmt != "jsonl":
                for field in ("context", "tags"):
                    if field in values:
                        # For delimited input use a JSON list when multiple
                        # turns/tags are needed; string context remains one turn.
                        cell = values[field]
                        if cell.startswith("["):
                            values[field] = _json(cell)
                for field in ("target", "context", "group_id", "tags", "speaker_id", "dialect",
                              "slang_category", "idiom", "noisy_transcription", "source_basis",
                              "verification_status", "slang"):
                    if values.get(field) == "":
                        values[field] = None
            for field in ("context_required", "holdout"):
                if field in values:
                    values[field] = _flag(values[field], field, from_delimited=fmt != "jsonl")
            for field in ("slang", "idiom", "noisy_transcription"):
                if values.get(field) is not None:
                    values[field] = _flag(values[field], field, from_delimited=fmt != "jsonl")
            context = values.get("context")
            if isinstance(context, str):
                context = [context]
            record = {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": spec.dataset_id,
                "dataset_version": spec.version,
                "original_id": values["original_id"],
                "source": values["source"],
                "target": values.get("target"),
                "source_language": values.get("source_language", spec.source_language),
                "target_language": values.get("target_language", spec.target_language),
                "split": values["split"],
                "group_id": values.get("group_id"),
                "context": [] if context is None else context,
                "context_required": values.get("context_required", False),
                "holdout": values.get("holdout", values["split"] == "holdout"),
                "slang": values.get("slang"),
                "tags": [] if values.get("tags") is None else values["tags"],
                **{field: values.get(field) for field in (
                    "speaker_id", "dialect", "idiom", "slang_category", "noisy_transcription",
                    "source_basis", "verification_status",
                )},
            }
            _validate_record(record, spec)
        except (DatasetError, ValueError, TypeError, RecursionError) as exc:
            reason = str(exc) if isinstance(exc, DatasetError) else "invalid field encoding"
            raise DatasetError(f"source record {number}: {reason}") from None
        records.append(record)
    if not records:
        raise DatasetError("dataset has no source records")
    audit_split_integrity(records)
    return records


def _fingerprint(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    key = "".join(c for c in normalized if unicodedata.category(c)[0] in "LMNS")
    return _sha((key or normalized.strip()).encode("utf-8"))


def _primary_language(tag: str) -> str:
    return tag.split("-", 1)[0].casefold()


def audit_split_integrity(records: Iterable[Mapping[str, Any]]) -> None:
    """Compare exact normalized source/context, identity, group and pairs across splits.

    Different benchmark corpora can leak by text even when their IDs differ.
    Train source, prior, and target are all compared to evaluative source,
    prior, and reference in the same language, including when direction
    reverses. Identical short/generic phrases across roles can be false
    positives; this conservative gate has no override or semantic matching.
    Same-split contextual examples may intentionally repeat a current turn.
    """
    ids: dict[tuple[Any, ...], str] = {}
    release_ids: set[tuple[Any, ...]] = set()
    groups: dict[tuple[Any, ...], str] = {}
    prompts: dict[tuple[str, str], str] = {}
    loaned_text: dict[tuple[str, str], bool] = {}
    pair_ids: dict[tuple[tuple[str, str], ...], str] = {}
    for record in records:
        split = record["split"]
        release = record["dataset_version"]
        identity = (record["dataset_id"].casefold(), record["original_id"])
        if ids.setdefault(identity, split) != split:
            raise DatasetError("split contamination: original ID appears in different splits")
        release_key = (*identity, release)
        if release_key in release_ids:
            raise DatasetError("duplicate original ID within a dataset version")
        release_ids.add(release_key)
        for field, label in (("group_id", "conversation/group"), ("speaker_id", "speaker")):
            if record[field] is not None:
                group = (record["dataset_id"].casefold(), field, record[field])
                if groups.setdefault(group, split) != split:
                    raise DatasetError(f"split contamination: {label} appears in different splits")
        language = _primary_language(record["source_language"])
        source_key = (language, _fingerprint(record["source"]))
        text_keys = []
        for key in (source_key, *((language, _fingerprint(turn)) for turn in record["context"])):
            if prompts.setdefault(key, split) != split:
                raise DatasetError("split contamination: source or context overlaps another split")
            text_keys.append(key)
        if record["target"] is not None:
            target_key = (_primary_language(record["target_language"]), _fingerprint(record["target"]))
            text_keys.append(target_key)
            pair = tuple(sorted((source_key, target_key)))
            if pair_ids.setdefault(pair, split) != split:
                raise DatasetError("split contamination: translation pair overlaps another split")
        is_train = split == "train"
        for key in text_keys:
            if loaned_text.setdefault(key, is_train) != is_train:
                raise DatasetError("split contamination: training text overlaps an evaluation source, context, or reference")


def _counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "by_split": dict(Counter(r["split"] for r in records)),
        "by_language_pair": dict(Counter(f'{r["source_language"]}->{r["target_language"]}' for r in records)),
        "with_context": sum(bool(r["context"]) for r in records),
        "context_required": sum(r["context_required"] for r in records),
        "slang": sum(r["slang"] is True for r in records),
    }


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def convert_dataset(
    source: str | Path,
    output: str | Path,
    spec: DatasetSpec,
    columns: Mapping[str, str],
    *,
    input_format: str | None = None,
) -> dict[str, Any]:
    """Create a new non-overwriting bundle; write the readiness manifest last."""
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    fmt = input_format or source.suffix.lstrip(".").lower()
    records = _convert_rows(raw, fmt, spec, columns)
    data = b"".join(_canon_bytes(r) for r in records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "spec": asdict(spec),
        "conversion": {"input_format": fmt, "source_sha256": _sha(raw), "columns": dict(columns)},
        "records_sha256": _sha(data),
        "counts": _counts(records),
    }
    receipt = _canon_bytes(manifest)
    try:
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        raise DatasetError("output bundle already exists; use a new version directory") from None
    # Exclusive create and commit-marker rename prevent an incomplete bundle
    # looking ready. A failed write leaves the new directory for diagnosis; an
    # existing bundle is never deleted or overwritten.
    _write_new(output / RECORDS_NAME, data)
    temporary = output / ".manifest.partial"
    _write_new(temporary, receipt)
    os.rename(temporary, output / MANIFEST_NAME)
    return manifest


@dataclass(frozen=True)
class DatasetBundle:
    spec: DatasetSpec
    records: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    manifest_sha256: str


def load_bundle(
    directory: str | Path,
    *,
    original_source: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> DatasetBundle:
    """Verify canonical files; optionally pin the manifest or replay raw conversion."""
    directory = Path(directory)
    paths = (directory, directory / MANIFEST_NAME, directory / RECORDS_NAME)
    if any(p.is_symlink() for p in paths) or not all(p.is_file() for p in paths[1:]):
        raise DatasetError("incomplete bundle or symlinked artifact")
    receipt, data = (p.read_bytes() for p in paths[1:])
    manifest_sha = _sha(receipt)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise DatasetError("manifest checksum mismatch")
    try:
        manifest = _json(receipt.decode("utf-8"))
        if set(manifest) != {"schema_version", "spec", "conversion", "records_sha256", "counts"}:
            raise DatasetError("malformed bundle manifest")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
            raise DatasetError("unsupported bundle schema version")
        if _canon_bytes(manifest) != receipt:
            raise DatasetError("noncanonical manifest")
        conversion = manifest["conversion"]
        if set(conversion) != {"input_format", "source_sha256", "columns"}:
            raise DatasetError("malformed conversion receipt")
        if not _sha_text(conversion["source_sha256"]) or conversion["input_format"] not in ("jsonl", "csv", "tsv"):
            raise DatasetError("malformed source provenance")
        _validate_columns(conversion["columns"])
        spec = DatasetSpec(**manifest["spec"])
        if _sha(data) != manifest["records_sha256"]:
            raise DatasetError("record checksum mismatch")
        records = list(_source_rows(data, "jsonl"))
        if not records:
            raise DatasetError("empty converted bundle")
        for record in records:
            _validate_record(record, spec)
        if b"".join(_canon_bytes(record) for record in records) != data or _counts(records) != manifest["counts"]:
            raise DatasetError("canonical record or count mismatch")
        audit_split_integrity(records)
        if original_source is not None:
            raw = Path(original_source).read_bytes()
            if _sha(raw) != conversion["source_sha256"]:
                raise DatasetError("original source checksum mismatch")
            replayed = _convert_rows(raw, conversion["input_format"], spec, conversion["columns"])
            if b"".join(_canon_bytes(record) for record in replayed) != data:
                raise DatasetError("conversion does not reproduce from the original source")
    except (UnicodeError, ValueError, TypeError, KeyError, RecursionError) as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError("malformed bundle") from None
    return DatasetBundle(spec, tuple(records), manifest, manifest_sha)


def training_rows(
    bundles: Iterable[DatasetBundle],
    *,
    purpose: str,
    expected_manifest_sha256s: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    """Return train only; fail closed on denied rights or non-public commercial inputs.

    One independently trusted manifest digest is required per verified local
    bundle, in matching order. Compare applicable *unsealed* test/dev bundles
    before constructing a train-only view. Do not supply a sealed holdout.
    """
    if not isinstance(purpose, str) or purpose not in PURPOSES:
        raise DatasetError("training purpose must be explicit")
    bundles = tuple(bundles)
    pins = tuple(expected_manifest_sha256s)
    if len(pins) != len(bundles) or not all(_sha_text(pin) for pin in pins):
        raise DatasetError("provide one trusted manifest SHA-256 per bundle")
    checked = []
    for bundle, pin in zip(bundles, pins):
        if not isinstance(bundle, DatasetBundle):
            raise DatasetError("invalid training bundle")
        try:
            receipt = _canon_bytes(bundle.manifest)
            if _sha(receipt) != pin or bundle.manifest_sha256 != pin:
                raise DatasetError("trusted training manifest checksum mismatch")
            manifest = _json(receipt.decode("utf-8"))
            spec = DatasetSpec(**manifest["spec"])
            if spec != bundle.spec or manifest["schema_version"] != SCHEMA_VERSION:
                raise DatasetError("training bundle manifest identity mismatch")
            data = b"".join(_canon_bytes(row) for row in bundle.records)
            if _sha(data) != manifest["records_sha256"]:
                raise DatasetError("training records checksum mismatch")
            # Take an independent snapshot; do not reuse caller-owned mutable rows.
            records = list(_source_rows(data, "jsonl"))
            for row in records:
                _validate_record(row, spec)
            if not records or _counts(records) != manifest["counts"]:
                raise DatasetError("invalid training bundle counts")
        except (UnicodeError, ValueError, TypeError, KeyError, RecursionError) as exc:
            if isinstance(exc, DatasetError):
                raise
            raise DatasetError("malformed training bundle") from None
        checked.append((spec, records))
    audit_split_integrity(row for _, records in checked for row in records)
    out = []
    for spec, records in checked:
        selected = [row for row in records if row["split"] == "train"]
        if not selected:
            continue
        if spec.rights not in PURPOSES:
            raise DatasetError("training blocked by unknown or prohibited data rights")
        if purpose == "commercial" and (spec.rights != purpose or spec.privacy != "public"):
            raise DatasetError("commercial training requires affirmative rights and public data")
        # Do not let caller edits mutate the verified view's list fields.
        out.extend({**row, "context": list(row["context"]), "tags": list(row["tags"])} for row in selected)
    if not out:
        raise DatasetError("no eligible train rows")
    return tuple(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="map a pinned source into a new checksummed bundle")
    convert.add_argument("source", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--spec", type=Path, required=True, help="JSON DatasetSpec")
    convert.add_argument("--columns", type=Path, required=True, help="JSON mapping of canonical to source fields")
    convert.add_argument("--format", choices=("jsonl", "csv", "tsv"))
    verify = commands.add_parser("verify", help="verify checksums and compare splits across the given bundles")
    verify.add_argument("bundles", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "convert":
            spec = DatasetSpec(**_json(args.spec.read_text(encoding="utf-8")))
            mapping = _json(args.columns.read_text(encoding="utf-8"))
            receipt = convert_dataset(args.source, args.output, spec, mapping, input_format=args.format)
            print(json.dumps({k: receipt[k] for k in ("schema_version", "records_sha256", "counts")}, sort_keys=True))
        else:
            bundles = [load_bundle(path) for path in args.bundles]
            audit_split_integrity(r for b in bundles for r in b.records)
            print(json.dumps({"verified_bundles": len(bundles), "manifest_sha256": [b.manifest_sha256 for b in bundles]}, sort_keys=True))
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        # Raw paths, source text and spec details can be private. Only sanitized
        # DatasetError reasons are shown through the supported CLI.
        reason = str(exc) if isinstance(exc, DatasetError) else "could not read or validate input"
        parser.exit(2, f"{reason}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
