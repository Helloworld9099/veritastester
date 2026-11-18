#!/usr/bin/env python3
"""Prepare CC0 DRAL Spanish-English real-conversation transcript pairs safely.

The transcript files are author-published tuning data, not the separate 2026
challenge evaluation release. All outputs are aggregate hashes; no raw lines
or conversation text are committed or uploaded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import re
from urllib.request import Request, urlopen

SOURCE = {
    "spanish": "https://www.cs.utep.edu/topi/ES_transcripts.txt",
    "english": "https://www.cs.utep.edu/topi/EN_transcripts.txt",
}
# Filled from a preliminary integrity-only, author-URL fetch before training.
PINNED_SHA256 = {"spanish": None, "english": None}
SEED = "bridgecaption-dral-20261118-v1"
FORMAT = "dral-es-source-context-v1-four-previous-six-words"
COUNTS = {"train": 768, "dev": 128, "test": 256}
PREFIX = {"spanish": "ES", "english": "EN"}
KEY_RE = re.compile(r"^([A-Z]{2})_([0-9]+)_([0-9.]+)\\.wav$")
DIGITS = re.compile(r"\\d+(?:[.,]\\d+)*")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def download_sources(*, require_pins: bool = True) -> tuple[dict[str, dict], dict]:
    parsed, manifest = {}, {}
    for lang, url in SOURCE.items():
        with urlopen(Request(url, headers={"User-Agent": "BridgeCaptionResearch/1.0"}),
                     timeout=40) as response:
            raw = response.read(750_001)
        if len(raw) > 750_000:
            raise ValueError("DRAL transcript exceeds bounded source size")
        sha = hashlib.sha256(raw).hexdigest()
        expected = PINNED_SHA256[lang]
        if require_pins and (not expected or sha != expected):
            raise ValueError(f"Missing or changed pinned {lang} transcript SHA-256")
        if expected and sha != expected:
            raise ValueError(f"Source changed: {lang}")
        try:
            rows = raw.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError("DRAL source requires verified UTF-8") from exc
        parsed[lang] = parse_transcripts(rows, lang)
        manifest[lang] = {"url": url, "sha256": sha, "bytes": len(raw),
                          "rows": len(parsed[lang])}
    if set(parsed["spanish"]) != set(parsed["english"]):
        raise ValueError("Spanish/English source IDs differ; never pair by line index")
    manifest["matched_ids"] = len(parsed["spanish"])
    manifest["conversation_groups"] = len(
        {row["group"] for row in parsed["spanish"].values()})
    return parsed, manifest


def parse_transcripts(text: str, language: str) -> dict:
    if language not in PREFIX:
        raise ValueError("Unsupported language")
    result = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        fields = raw.split("|", 2)
        if len(fields) != 3:
            raise ValueError(f"Malformed DRAL row {lineno}")
        file_id, speaker, utterance = fields
        match = KEY_RE.fullmatch(file_id.strip())
        if not match or match.group(1) != PREFIX[language]:
            raise ValueError(f"Unexpected DRAL transcript identifier at row {lineno}")
        if not speaker.strip() or len(speaker) > 40:
            raise ValueError(f"Unexpected DRAL speaker label at row {lineno}")
        # Use the author's exact paired conversation/segment identifiers.
        group, segment = match.group(2), match.group(3)
        key = f"{group}_{segment}"
        if key in result:
            raise ValueError("Duplicate paired identifier")
        result[key] = {"group": group, "utterance": utterance.strip(),
                       "source_order": lineno}
    if len(result) < 1100:
        raise ValueError("Too few transcript records; source likely incomplete")
    return result


def eligible(source: str, target: str) -> bool:
    if not source or not target:
        return False
    source_words, target_words = source.split(), target.split()
    if not (1 <= len(source_words) <= 22 and 1 <= len(target_words) <= 30):
        return False
    if len(source) > 220 or len(target) > 250:
        return False
    if normalized(source) == normalized(target):
        return False
    # Mismatched literal digits are an obvious error, not a semantic judge.
    if sorted(DIGITS.findall(source)) != sorted(DIGITS.findall(target)):
        return False
    return True


def prompt(source: str, previous: list[str], *, contextual: bool) -> str:
    """The exact same versioned formatter must be used at training/inference."""
    if not contextual:
        return source
    if not previous or len(previous) > 4:
        raise ValueError("Contextual sample requires 1-4 actual prior utterances")
    if any(not s or len(s) > 220 for s in previous):
        raise ValueError("Invalid natural preceding turn")
    # Bounded to preserve current utterance under Marian's 96-token train limit.
    clipped = [" ".join(turn.split()[:6]) for turn in previous]
    return "Contexto previo: " + " / ".join(clipped) + " . Mensaje actual: " + source


def choose_cohorts(parsed: dict, *, counts: dict | None = None) -> tuple[dict, dict]:
    """Never share a conversation, utterance or exact reference across splits."""
    counts = COUNTS if counts is None else counts
    es, en = parsed["spanish"], parsed["english"]
    if set(es) != set(en):
        raise ValueError("English and Spanish alignment IDs differ")
    groups = defaultdict(list)
    for key, item in es.items():
        if item["group"] != en[key]["group"]:
            raise ValueError("Conversation group differs across source/target")
        groups[item["group"]].append(key)
    if len(groups) < 75:
        raise ValueError("Not enough independent DRAL dialogue groups")
    ordered = sorted(groups, key=lambda group: digest(f"{SEED}|{group}"))
    if len(ordered) < 70:
        raise ValueError("Insufficient author dialogue groups")
    # This assignment is frozen before examining development or test metrics.
    membership = {
        "test": set(ordered[:28]),
        "dev": set(ordered[28:48]),
        "train": set(ordered[48:]),
    }
    outputs, evidence = {}, {}
    used_source, used_target = set(), set()
    for split in ("test", "dev", "train"):
        selected, selected_groups = [], set()
        group_order = sorted(membership[split],
                             key=lambda group: digest(f"{SEED}|pick|{group}"))
        for group in group_order:
            # Conversation chronology comes from author-transcript line ordering.
            keys = sorted(groups[group], key=lambda key: es[key]["source_order"])
            for index, key in enumerate(keys):
                source = es[key]["utterance"]
                target = en[key]["utterance"]
                if not eligible(source, target):
                    continue
                past = [es[k]["utterance"] for k in keys[:index] if es[k]["utterance"]]
                previous = past[-4:]
                if not previous:
                    # Exclude first turn equally from both matched experiment arms.
                    continue
                a, b = normalized(source), normalized(target)
                if a in used_source or b in used_target:
                    continue
                selected.append({"key": key, "group": group, "source": source,
                                 "target": target, "previous": previous})
                used_source.add(a)
                used_target.add(b)
                selected_groups.add(group)
                if len(selected) == counts[split]:
                    break
            if len(selected) == counts[split]:
                break
        if len(selected) != counts[split]:
            raise ValueError(
                f"DRAL {split} has only {len(selected)} safe group-isolated rows; "
                f"needs {counts[split]}. Do not move heldout records into training.")
        outputs[split] = selected
        ids = sorted(row["key"] for row in selected)
        evidence[split] = {
            "rows": len(selected), "groups": len(selected_groups),
            "group_ids_sha256": digest("\\n".join(sorted(selected_groups))),
            "sample_ids_sha256": digest("\\n".join(ids)),
            "source_and_reference_pairs_sha256": digest(
                "\\n".join(digest(row["source"] + "\\0" + row["target"])
                          for row in selected)),
        }
    assert not (membership["train"] & membership["dev"] or
                membership["train"] & membership["test"] or
                membership["dev"] & membership["test"])
    evidence["split_definition"] = {
        "seed": SEED, "first_28_test_groups": True,
        "next_20_development_groups": True,
        "remaining_training_groups": True,
        "group_disjoint": True, "exact_text_disjoint": True,
        "all_pairs_authentic_author_transcripts": True,
        "semantic_equivalence_human_verified": False,
    }
    return outputs, evidence


def inspect():
    data, provenance = download_sources(require_pins=False)
    output = {"provenance": provenance}
    try:
        _, splits = choose_cohorts(data)
        output["cohorts"] = splits
        output["eligible_for_pinned_experiment"] = True
    except ValueError as exc:
        output["eligible_for_pinned_experiment"] = False
        output["blocker"] = str(exc)
    print("DRAL_SOURCE_PREFLIGHT", json.dumps(output, sort_keys=True))
    if not output["eligible_for_pinned_experiment"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--inspect", action="store_true")
    args = cli.parse_args()
    if not args.inspect:
        cli.error("Only provenance/split inspection is implemented here")
    inspect()
