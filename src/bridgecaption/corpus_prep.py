"""Offline, provenance-preserving preparation for BridgeCaption research corpora.

This tool never downloads data, contacts a model, or fabricates training labels.
The outputs are private normalized records; downstream models must only learn
from reviewed labels that match their actual target. Official held-out records
are never reassigned to training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SOURCES = {
    "magpie": ("idiomaticity", "source-provided idiomatic/literal labels"),
    "semeval": ("idiomaticity", "source-provided idiomatic/literal labels"),
    "slanginterp": ("sense_review", "definitions are candidate evidence, not reviewed sense labels"),
    "multi3woz": ("conversation_review", "multi-turn dialogue; no memory-utility labels"),
    "bangor": ("conversation_review", "requires normalized TalkBank/CHAT export"),
    "diabla": ("evaluation_only", "human reference translation; official test-only set"),
    "naijasenti": ("informal_review", "sentiment is NOT a slang-sense label"),
    "okwu": ("parallel_text", "aligned translation; NO dialogue or memory labels"),
}
LANG = {"english": "en", "en": "en", "french": "fr", "fr": "fr",
        "arabic": "ar", "ar": "ar", "turkish": "tr", "tr": "tr",
        "spanish": "es", "es": "es", "portuguese": "pt", "pt": "pt",
        "galician": "gl", "gl": "gl", "yoruba": "yo", "yor": "yo", "yo": "yo",
        "swahili": "sw", "swa": "sw", "sw": "sw", "nigerian pidgin": "pcm",
        "pcm": "pcm", "hausa": "ha", "ha": "ha", "igbo": "ig", "ig": "ig"}


def tidy(value: object, cap: int = 1500) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, int, float)):
        raise ValueError("Expected textual field")
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())[:cap]


def lang(value: object) -> str:
    text = tidy(value, 30).casefold()
    if not text:
        raise ValueError("Explicit language required")
    return LANG.get(text, text)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def identity(source: str, text: str) -> str:
    return f"{source}:{digest(text)}"


def group_split(group: str) -> str:
    """Stable split across files and languages, independent of row order."""
    position = int(digest("bridgecaption-2026|" + group), 16) % 100
    return "train" if position < 60 else "dev" if position < 80 else "test"


def normalize_split(value: str, default: str):
    value = tidy(value).casefold()
    aliases = {"val": "dev", "validation": "dev", "eval": "test", "development": "dev"}
    if value:
        value = aliases.get(value, value)
        if value not in {"train", "dev", "test"}:
            raise ValueError("Unexpected original split: " + value)
        return value
    return default


def make_record(source: str, *, case_id: str, group_id: str,
                language: str, utterance: str, previous: str = "",
                situation: str = "", expression: str = "", label: str = "",
                label_kind: str = "", reference_translation: str = "",
                candidate_definition: str = "", original_split: str = "",
                provenance: str = "", rights: str = "",
                extra: dict | None = None):
    if source not in SOURCES:
        raise ValueError("Unrecognized dataset")
    utterance = tidy(utterance)
    if not utterance:
        raise ValueError("Empty utterance")
    group_id = tidy(group_id, 200)
    if not group_id:
        raise ValueError("Missing dialogue, speaker family, or idiom grouping")
    role = SOURCES[source][0]
    split = ("test" if source == "diabla" else
             normalize_split(original_split, group_split(group_id)))
    return {
        "schema": "bridgecaption_corpus_v1",
        "dataset": source,
        "role": role,
        "case_id": tidy(case_id, 200),
        "group_id": group_id,
        "split": split,
        "original_split": tidy(original_split, 20),
        "language": lang(language),
        "utterance": utterance,
        "previous": tidy(previous),
        "situation": tidy(situation),
        "expression": tidy(expression, 100),
        "label_kind": label_kind,
        "source_label": tidy(label, 160),
        "reference_translation": tidy(reference_translation),
        "candidate_definition": tidy(candidate_definition),
        "provenance": tidy(provenance, 350),
        "rights_status": tidy(rights, 100),
        "extra": extra or {},
    }


def load_rows(path: Path, *, limit: int):
    if path.suffix.lower() in {".csv", ".tsv"}:
        with path.open(encoding="utf-8-sig", newline="") as file:
            yield from capped(csv.DictReader(file, delimiter="\t" if path.suffix == ".tsv" else ","), limit)
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as file:
            for i, line in enumerate(file):
                if not line.strip():
                    continue
                if i >= limit:
                    return
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL rows must be objects")
                yield value
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from capped(data, limit)
        elif isinstance(data, dict):
            # MultiWOZ-style datasets frequently index dialogues by ID.
            for dialogue_id, value in capped(data.items(), limit):
                if not isinstance(value, dict):
                    raise ValueError("Expected dialogue objects")
                yield {"dialogue_id": dialogue_id, **value}
        else:
            raise ValueError("JSON must contain an array or ID-to-dialogue mapping")
    else:
        raise ValueError("Provide CSV, TSV, JSONL or JSON (extract zip archives first)")


def capped(items, limit: int):
    for i, value in enumerate(items):
        if i >= limit:
            break
        yield value


def column(row: dict, name: str, mapping: dict, *, required: bool = False):
    actual = mapping.get(name, name)
    value = row.get(actual)
    if required and not tidy(value):
        raise ValueError(f"Required field {name!r} (source column {actual!r}) missing")
    return value


def sequence(row: dict, mapping: dict):
    for name in ("turns", "log", "dialogue", "messages"):
        value = row.get(mapping.get(name, name))
        if isinstance(value, list):
            return value
    raise ValueError("Missing dialogue turns list (turns/log/dialogue/messages)")


def one_record(source: str, row: dict, number: int, *, language: str,
               mapping: dict, supplied_split: str, rights: str):
    """Transform sentence-level sources. Dialogue sources have separate adapter."""
    provenance = f"{source}:row-{number}"
    supplied_lang = column(row, "language", mapping) or language
    if source == "magpie":
        context = row.get(mapping.get("context", "context"))
        expression = tidy(row.get(mapping.get("idiom", "idiom")), 100)
        if not isinstance(context, list) or not expression:
            raise ValueError("MAGPIE requires context list and idiom")
        matches = [i for i, sentence in enumerate(context)
                   if expression.casefold() in tidy(sentence).casefold()]
        if len(matches) != 1:
            raise ValueError("MAGPIE target idiom is not uniquely located in context")
        idx = matches[0]
        raw_label = tidy(column(row, "label", mapping)).casefold()
        if raw_label not in {"i", "l"}:
            raise ValueError("MAGPIE example lacks a reliable binary sense label")
        return make_record(source,
            case_id=identity(source, str(row.get("id", number))),
            group_id=f"idiom:{lang(supplied_lang or 'en')}:{expression.casefold()}",
            language=supplied_lang or "en", utterance=context[idx],
            previous=" ".join(tidy(x) for x in context[max(idx-2, 0):idx]),
            expression=expression, label={"i": "idiomatic", "l": "literal"}[raw_label],
            label_kind="idiomaticity_not_exact_sense",
            original_split=row.get("split") or supplied_split,
            provenance=provenance, rights=rights,
            extra={"document_id": tidy(row.get("document_id"), 100)})
    if source == "semeval":
        raw_label = tidy(column(row, "label", mapping) or row.get("Label")).casefold()
        if raw_label not in {"0", "1"} and supplied_split != "test":
            raise ValueError("SemEval needs binary Label (0 idiomatic, 1 non-idiomatic)")
        expression = tidy(column(row, "expression", mapping) or row.get("MWE"))
        if not expression:
            raise ValueError("SemEval needs MWE expression")
        source_lang = supplied_lang or row.get("Language")
        return make_record(source,
            case_id=identity(source, str(row.get("DataID", number))),
            group_id=f"idiom:{lang(source_lang)}:{expression.casefold()}",
            language=source_lang, utterance=column(row, "utterance", mapping) or row.get("Target"),
            previous=column(row, "previous", mapping) or row.get("Previous"),
            expression=expression,
            label={"0": "idiomatic", "1": "literal"}.get(raw_label, ""),
            label_kind="idiomaticity_not_exact_sense" if raw_label in {"0","1"} else "",
            original_split=supplied_split,
            provenance=provenance, rights=rights)
    if source == "slanginterp":
        expression = tidy(column(row, "expression", mapping) or row.get("Slang"), 100)
        usage = tidy(column(row, "utterance", mapping) or row.get("Usage Context"))
        meaning = tidy(column(row, "definition", mapping) or row.get("Definition"), 200)
        if not expression or not usage or not meaning:
            raise ValueError("SlangInterp requires Slang, Definition and Usage Context")
        if "[*SLANGAAAP*]" in usage:
            usage = usage.replace("[*SLANGAAAP*]", expression)
        elif expression.casefold() not in usage.casefold():
            raise ValueError("SlangInterp usage lacks token or expression")
        return make_record(source,
            case_id=identity(source, f"{expression}|{usage}|{meaning}"),
            group_id=f"slang:{expression.casefold()}", language=supplied_lang or "en",
            utterance=usage, expression=expression, candidate_definition=meaning,
            provenance=provenance, rights=rights, original_split=supplied_split,
            label_kind="unverified_dictionary_definition")
    if source == "naijasenti":
        text = column(row, "utterance", mapping, required=True)
        source_lang = supplied_lang or column(row, "lang", mapping)
        case_id = str(column(row, "id", mapping) or number)
        family = column(row, "author_id", mapping) or case_id
        return make_record(source, case_id=identity(source, f"{lang(source_lang)}:{case_id}"),
            group_id=identity("author", f"{lang(source_lang)}:{family}"), language=source_lang,
            utterance=text, provenance=provenance, rights=rights,
            label_kind="sentiment_is_not_sense",
            extra={"has_sentiment_label": bool(row.get(mapping.get("sentiment", "label")))})
    if source == "okwu":
        utterance = column(row, "utterance", mapping) or row.get("source_text")
        reference = column(row, "reference_translation", mapping) or row.get("english_translation")
        source_lang = supplied_lang or row.get("source_lang")
        if not utterance or not reference or not source_lang:
            raise ValueError("Okwu requires source_lang, source_text and english_translation")
        key = f"{lang(source_lang)}:{tidy(utterance)}:{tidy(reference)}"
        return make_record(source,
            case_id=identity(source, str(row.get("record_id") or key)),
            group_id=identity("pair", f"{lang(source_lang)}:{tidy(utterance)}"),
            language=source_lang, utterance=utterance,
            reference_translation=reference, original_split=supplied_split,
            provenance=provenance, rights=rights,
            label_kind="parallel_text_not_dialogue",
            extra={"source_verified_claim": row.get("verified") is True,
                   "original_source": tidy(row.get("source"), 80)})
    if source == "diabla":
        group = column(row, "dialogue_id", mapping)
        if not group:
            raise ValueError(
                "DiaBLa is TEST-ONLY. Supply a verified conversation ID as dialogue_id "
                "before import; do not infer a group from utterance id")
        meta = row.get("utterance_meta") or {}
        source_lang = supplied_lang or meta.get("lang")
        return make_record(source,
            case_id=identity(source, str(row.get("id") or number)),
            group_id=identity("dialogue", str(group)), language=source_lang,
            utterance=column(row, "utterance", mapping) or row.get("orig"),
            reference_translation=column(row, "reference_translation", mapping) or row.get("ref"),
            previous=column(row, "previous", mapping),
            provenance=provenance, rights=rights, original_split="test",
            label_kind="human_reference_test_only")
    raise ValueError(f"{source} uses a dialogue adapter")


def dialogue_records(source: str, rows, *, mapping: dict, language: str,
                     rights: str, supplied_split: str):
    """Multi3WOZ and normalized Bangor exports. No automatic utility labels."""
    history: dict[tuple[str, str], list[str]] = defaultdict(list)
    for number, row in enumerate(rows, 1):
        if source == "bangor":
            group = column(row, "conversation_id", mapping, required=True)
            raw_lang = column(row, "language", mapping) or language
            utterance = column(row, "utterance", mapping, required=True)
            turns = [utterance]
        else:
            group = column(row, "dialogue_id", mapping, required=True)
            raw_lang = column(row, "language", mapping) or language
            turns = [turn.get(mapping.get("utterance", "utterance")) or
                     turn.get(mapping.get("text", "text")) or
                     turn.get(mapping.get("content", "content"))
                     for turn in sequence(row, mapping)]
            if not all(isinstance(turn, str) and turn.strip() for turn in turns):
                raise ValueError("Some dialogue turns have no text; specify --columns mapping")
        if not raw_lang:
            raise ValueError("Dialogue language must be explicitly supplied or present in record")
        group_id = identity("dialogue", f"{source}:{tidy(group)}")
        source_lang = lang(raw_lang)
        key = (group_id, source_lang)
        for turn_number, turn in enumerate(turns, 1):
            text = tidy(turn)
            if not text:
                continue
            yield make_record(source,
                case_id=identity(source, f"{group}:{source_lang}:{number}:{turn_number}"),
                group_id=group_id, language=source_lang, utterance=text,
                previous=" ".join(history[key][-3:]), original_split=supplied_split,
                provenance=f"{source}:dialogue-{tidy(group,80)}:turn-{turn_number}",
                rights=rights)
            history[key].append(text)
            if len(history[key]) > 4:
                history[key].pop(0)


def prepare(source: str, path: Path, *, language: str = "", columns: dict | None = None,
            source_split: str = "", rights: str = "review-required",
            limit: int = 50_000):
    if source not in SOURCES:
        raise ValueError(f"Unknown dataset: {source}")
    if not 1 <= limit <= 200_000:
        raise ValueError("Limit must be 1-200000")
    mapping = columns or {}
    if any(not isinstance(k, str) or not isinstance(v, str) for k,v in mapping.items()):
        raise ValueError("Column mappings must be strings")
    if source_split:
        normalize_split(source_split, "train")
    rows = load_rows(path, limit=limit)
    if source in {"multi3woz", "bangor"}:
        records = list(dialogue_records(source, rows, mapping=mapping,
                                        language=language, rights=rights,
                                        supplied_split=source_split))
    else:
        records = [one_record(source, row, n, language=language, mapping=mapping,
                              supplied_split=source_split, rights=rights)
                   for n, row in enumerate(rows, 1)]
    if not records:
        raise ValueError("No valid records: verify the source schema")
    ids = [r["case_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate normalized case IDs; retain unique source identifiers")
    return records


def audit(records):
    duplicates = defaultdict(set)
    cross_split_groups = defaultdict(set)
    counts = Counter()
    split_counts = Counter()
    for record in records:
        counts[(record["dataset"], record["role"])] += 1
        split_counts[record["split"]] += 1
        text_hash = digest(record["utterance"].casefold())
        if len(record["utterance"]) >= 32:
            duplicates[text_hash].add(record["split"])
        cross_split_groups[record["group_id"]].add(record["split"])
    return {
        "count": len(records),
        "by_dataset_role": {f"{s}/{r}": n for (s,r),n in sorted(counts.items())},
        "by_split": dict(sorted(split_counts.items())),
        "cross_split_group_conflicts": sum(len(v) > 1 for v in cross_split_groups.values()),
        "cross_split_exact_text_warnings": sum(len(v) > 1 for v in duplicates.values()),
        "label_types": dict(Counter(r["label_kind"] or "none" for r in records)),
        "training_ready": False,
        "reason": "Source examples are not independently reviewed memory/sense utility labels",
    }


def private_write(path: Path, records):
    """No overwrite or accidental copy into repo's ordinary tracked source tree."""
    safe_parts = set(path.parts)
    if "private" not in safe_parts and "results" not in safe_parts:
        raise ValueError("Output must be located under a private/ or results/ folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    p = argparse.ArgumentParser(description="Offline dataset preparation; never trains or downloads")
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--language", default="", help="Required if source does not identify language")
    p.add_argument("--columns", default="{}", help='JSON: {"utterance":"Text","definition":"Definition"}')
    p.add_argument("--source-split", choices=("train","dev","test"), default="")
    p.add_argument("--rights", default="review-required",
                   help="Dataset's reviewed license/access status, stored as provenance")
    p.add_argument("--limit", type=int, default=50000)
    p.add_argument("--out", type=Path)
    p.add_argument("--write", action="store_true", help="Save private records locally; never uploads")
    args = p.parse_args()
    try:
        mapping = json.loads(args.columns)
        if not isinstance(mapping, dict):
            raise ValueError("--columns must be a JSON object")
        records = prepare(args.source, args.input, language=args.language, columns=mapping,
                          source_split=args.source_split, rights=args.rights, limit=args.limit)
        report = audit(records)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["cross_split_group_conflicts"]:
            p.error("Group leakage found. Correct grouping/source splits before writing")
        if report["cross_split_exact_text_warnings"]:
            print("WARNING: Repeated long text across splits. Review duplicates before training.")
        if args.write:
            if not args.out:
                p.error("--out is required with --write")
            private_write(args.out, records)
            print(f"Wrote {len(records)} privately stored examples to {args.out}")
        else:
            print("DRY RUN ONLY. No data written or training started.")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
