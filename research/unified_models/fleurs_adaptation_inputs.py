"""Prepare exact dual-label FLEURS train/DEV metadata; no audio or test inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import unicodedata

from research.workflow_contract import ContractError, _read_relative

REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
ORIGIN = f"https://huggingface.co/datasets/google/fleurs/tree/{REVISION}"
LOCALES = {"es_419": "es->en", "fr_fr": "fr->en"}
SPLITS = ("train", "dev")
_PINS = {
    ("en_us", "train"): "3ccfc83672cc03a835143e325abb38b4163e3a21725bc1a7d1165bc309b95852",
    ("en_us", "dev"): "9d57ee7e91e9d4c92edb39f6bbea668ef8dc2a3ff96eb510d5580b2ad05d17ec",
    ("es_419", "train"): "7755cc1bad2759d43a4299a24df08a9702986a0ea8e203412a7c7dd9c3955638",
    ("es_419", "dev"): "352b2a5f03f02fe752254fab737f76519963650f4630d2fb63957177aeec927f",
    ("fr_fr", "train"): "3df668c3b9b4101cc3e3d8f3024b311ccfd2e9a8c8f910b3efedc27fd4219a0f",
    ("fr_fr", "dev"): "3e0b792358c1cb4a426fe1c18fc1571d5406b390d738a2ed1bfd3c8b9d28de44",
}
_AUDIO = re.compile(r"[0-9]{1,24}\.wav\Z")
_ID = re.compile(r"[1-9][0-9]{0,6}\Z")


class InputError(ValueError):
    """Invalid or contaminated preparation input, with no source values."""


def _check(value, code):
    if not value:
        raise InputError(code)


def _normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def _parse(raw):
    rows = []
    transcript = {}
    members = set()
    try:
        lines = raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeError:
        raise InputError("invalid_tsv_utf8") from None
    _check(0 < len(lines) <= 10_000, "invalid_tsv_row_count")
    for line in lines:
        fields = line.split("\t")
        _check(len(fields) == 7, "invalid_fleurs_fields")
        identity, member, source = fields[:3]
        _check(_ID.fullmatch(identity) and _AUDIO.fullmatch(member) and "\0" not in source
               and len(source) <= 4096 and _normalized(source), "invalid_fleurs_row")
        identity = int(identity)
        _check(member not in members, "duplicate_audio_member")
        _check(transcript.setdefault(identity, source) == source, "ambiguous_sentence_reference")
        rows.append((identity, member, source))
        members.add(member)
    return rows, transcript


def _public_tsv(root):
    tables = {}
    for (locale, split), expected in _PINS.items():
        raw = _read_relative(Path(root), (f"{locale}-{split}.tsv",), size_limit=4 << 20)
        _check(sha256(raw).hexdigest() == expected, "publisher_tsv_pin_mismatch")
        tables[locale, split] = raw
    return tables


def _prepare(tables):
    _check(set(tables) == set(_PINS), "only_six_train_dev_tables_allowed")
    parsed = {key: _parse(raw) for key, raw in tables.items()}
    # Check all source rows, including those later skipped for missing English.
    ids = {split: set().union(*(set(parsed[loc, split][1]) for loc in (*LOCALES, "en_us")))
           for split in SPLITS}
    _check(not (ids["train"] & ids["dev"]), "cross_split_parallel_sentence_id")
    members = {split: {row[1] for key, (rows, _) in parsed.items()
                       if key[1] == split for row in rows} for split in SPLITS}
    _check(not (members["train"] & members["dev"]), "cross_split_audio_member_name")
    # Global across all locales, including the common English reference text.
    texts = {split: {_normalized(s) for key, (_, refs) in parsed.items()
                     if key[1] == split for s in refs.values()} for split in SPLITS}
    _check(not (texts["train"] & texts["dev"]), "cross_split_exact_normalized_text")

    bundles = defaultdict(list)
    coverage = {}
    for locale, direction in LOCALES.items():
        coverage[direction] = {}
        for split in SPLITS:
            references = parsed["en_us", split][1]
            rows = parsed[locale, split][0]
            missing = 0
            for identity, member, transcript in sorted(rows):
                if identity not in references:
                    missing += 1
                    continue
                bundles[split].append({
                    "direction": direction, "source_locale": locale, "split": split,
                    "sentence_id": identity, "audio_member": member,
                    "source_transcript": transcript, "english_reference": references[identity],
                })
            paired = len(rows) - missing
            _check(paired > 0, "missing_split_direction")
            coverage[direction][split] = {
                "audio_rows": len(rows), "paired_rows": paired, "missing_english_rows": missing,
            }
    return bundles, coverage


def _jsonl(rows):
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def run(root, *, export_dir=None):
    bundles, coverage = _prepare(_public_tsv(root))
    raw = {split: _jsonl(bundles[split]) for split in SPLITS}
    summary = {
        "kind": "bridgecaption_fleurs_adaptation_input_prep",
        "dataset": {"revision": REVISION, "origin": ORIGIN, "license": "CC-BY-4.0"},
        "scope": "pinned_train_dev_metadata_only",
        "directions": coverage,
        "manifest_sha256": {split: sha256(raw[split]).hexdigest() for split in SPLITS},
        "exact_cross_split_checks_passed": [
            "parallel_sentence_id", "audio_member_filename", "normalized_text",
        ],
        "audio_bytes_checked": False,
        "speaker_audit_verified": False,
        "foundation_pretraining_disjoint_verified": False,
        "holdout_opened": False,
    }
    if export_dir is not None:
        output = Path(export_dir).resolve()
        repo = Path(__file__).resolve().parents[2]
        if output.is_relative_to(repo):
            raise InputError("use_new_local_directory_outside_checkout")
        output.mkdir(mode=0o700)
        for split in SPLITS:
            (output / f"{split}.jsonl").write_bytes(raw[split])
        (output / "ATTRIBUTION.json").write_text(json.dumps(summary["dataset"], indent=2) + "\n",
                                                 encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Six exact publisher TSVs; no other files opened")
    parser.add_argument("--export-dir", help="New directory outside the checkout for local JSONL")
    args = parser.parse_args(argv)
    try:
        result = run(args.root, export_dir=args.export_dir)
    except (InputError, ContractError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except OSError:
        print("could_not_write_new_local_export", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
