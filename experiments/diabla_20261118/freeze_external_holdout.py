#!/usr/bin/env python3
"""Freeze never-trained-on DiaBLa conversation groups BEFORE model evaluation.

Research-only external test: public human conversations with bilingual
reference translations; no model inference or training occurs here.
Source test language is determined by utterance_meta.lang. Existing human
labels are for OTHER baseline MT systems, not BridgeCaption predictions.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

REPO = "rbawden/DiaBLa"
LANGUAGE = "french"
SPLIT = "test"
LICENSE = "cc-by-sa-4.0"


def group_from_id(row):
    """Keep whole dialogues together using the published record identifier."""
    identifier = row.get("id")
    if not isinstance(identifier, str) or not re.search(r"_\d+$", identifier):
        raise ValueError("Unrecognized DiaBLa utterance ID")
    return identifier.rsplit("_", 1)[0]


def validate_row(row):
    if not isinstance(row, dict):
        raise ValueError("Missing dialogue source record")
    group = group_from_id(row)
    meta = row.get("utterance_meta")
    if not isinstance(meta, dict) or meta.get("lang") not in ("french", "english"):
        raise ValueError("Unexpected speaker-language metadata")
    for field in ("orig", "ref", "mt"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError("Missing original/reference/old-system text")
    if not isinstance(row.get("dialogue_history"), list):
        raise ValueError("Missing real preceding-turn context")
    return group, meta["lang"]


def freeze(rows, revision: str):
    if not (isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)):
        raise ValueError("Require immutable 40-character upstream Git commit")
    seen = set()
    group_lang, counts = {}, Counter()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate bilingual utterance ID")
        seen.add(row["id"])
        key, lang = validate_row(row)
        group_lang.setdefault(key, set()).add(lang)
        counts[lang] += 1
    if len(seen) < 4000 or len(group_lang) < 100 or counts["french"] < 1000:
        raise ValueError("DiaBLa dataset too small or publisher schema changed")
    groups = sorted(group_lang)
    # All dialogues remain held out, no split is allowed in BridgeCaption training.
    group_digests = [hashlib.sha256(g.encode()).hexdigest() for g in groups]
    return {
        "dataset": REPO, "revision": revision, "publisher_split": SPLIT,
        "license": LICENSE,
        "all_utterances_test_only_no_train_or_dev": True,
        "n_utterances": len(seen),
        "n_dialogue_groups": len(groups),
        "source_language_counts": dict(counts),
        "french_original_source_utterances": counts["french"],
        "frozen_group_ids_sha256": hashlib.sha256(
            "\n".join(groups).encode()).hexdigest(),
        "dialogue_group_digests_sha256": hashlib.sha256(
            "\n".join(group_digests).encode()).hexdigest(),
        "human_judgments_apply_only_to_older_translation_models": True,
        "not_sufficient_for_90_percent_new_model_semantic_adequacy": True,
        "test_group_exclusions": "All 144 publisher dialogues are reserved solely for evaluation.",
    }


def flatten_public_dialogues(document):
    """Read published JSON as DATA, never execute dataset-provided Python."""
    if not isinstance(document, dict) or not isinstance(document.get("dialogues"), dict):
        raise ValueError("Missing published DiaBLa dialogues JSON")
    for name, dialogue in sorted(document["dialogues"].items()):
        if not isinstance(dialogue, dict) or not isinstance(dialogue.get("utterances"), dict):
            raise ValueError("Malformed published bilingual dialogue")
        history = []
        for identifier, source in dialogue["utterances"].items():
            if not isinstance(source, dict):
                raise ValueError("Malformed published bilingual utterance")
            row = {
                "id": f"{name}_{identifier}",
                "orig": source.get("original_text"),
                "ref": source.get("reference_translation"),
                "mt": source.get("postprocessed_text"),
                "utterance_meta": {"lang": source.get("language")},
                "dialogue_history": list(history),
            }
            yield row
            history.append({"text": source.get("original_text"),
                            "language": source.get("language")})


def main():
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(REPO)
    if not isinstance(info.card_data, dict):
        license_value = getattr(info.card_data, "license", None)
    else:
        license_value = info.card_data.get("license")
    if license_value not in (LICENSE, [LICENSE]):
        raise RuntimeError("Upstream dataset license cannot be verified")
    path = Path(hf_hub_download(
        repo_id=REPO, filename="DiaBLa.json", repo_type="dataset", revision=info.sha))
    raw = path.read_bytes()
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("Unexpectedly large DiaBLa source JSON")
    document = json.loads(raw)
    rows = flatten_public_dialogues(document)
    frozen = freeze(rows, info.sha)
    frozen["source_json_sha256"] = hashlib.sha256(raw).hexdigest()
    frozen["source_file"] = "DiaBLa.json"
    print("FROZEN_DIABLA_EXTERNAL_TEST_MANIFEST")
    print(json.dumps(frozen, sort_keys=True, indent=2))
    # This is only a schema/selection inspection. No raw dialogue, old MT
    # annotations, new translation or final holdout benchmark score is logged.


if __name__ == "__main__":
    main()
