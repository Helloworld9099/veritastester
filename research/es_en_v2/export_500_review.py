#!/usr/bin/env python3
"""Generate exactly 500 ACTUAL Marian model answers for blinded human review.

NOT a final sealed holdout: 40 author-created stress cases, 300 MIT
BilingualBabyLM synthetic dialogue examples, and 160 public Amazon MASSIVE
publisher-VALIDATION utterances. No reference/answer key is exported.

ES->EN: pinned v2 sequentially trained adapter SHA 3bf849...
EN->ES: pinned original Marian base SHA 5bc449... (no learned LoRA).
Both translate ONLY the current utterance, without injecting prior context,
since past-injection previously produced context leakage in outputs.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import random
import re
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/cpu_continuation_20261117"))
sys.path.insert(0, str(ROOT / "research/es_en_v2"))
from research.es_en_v2.baby_dialogue import parse_record
from research.es_en_v2.stress_eval import load as load_model
from research.es_en_v2.stress_eval import cases as load_stress_cases
from research.es_en_v2.inspect_babylm_stream import MAX_BYTES, REPO_ID, DATAFILE
from research.es_en_v2.train_three import aligned

OUT = ROOT / "research/es_en_v2/reviewer_examples_500"
TOTAL = 500
PER_DIRECTION = 250
STRESS_N = 20
DIALOGUE_N = 150
GENERAL_N = 80
SEED = 20261120500
EN_ES_SHA = "5bc4493d463cf000c1f0b50f8d56886a392ed4ab"
ES_EN_ADAPTER_SHA = "3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71"
MASSIVE_REV = "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
BABY_REV = "1093ed2c348313c86b5c4c22563226b65a9b11d1"
BABY_PREFIX_SHA = "1a9eaf0ce3a5ee5dd37aa622a4571b0ac38ae7ab0da68333c5b50ff175a9d6c8"
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
URL = re.compile(r"https?://\S+")


def norm(x: str) -> str:
    return " ".join(x.casefold().strip().split())


def safe(x: str) -> bool:
    return (
        isinstance(x, str) and 4 <= len(x) <= 230
        and 3 <= len(x.split()) <= 37
        and not EMAIL.search(x) and not PHONE.search(x) and not URL.search(x)
        and not any(ord(c) < 32 for c in x)
        and "\ufffd" not in x
    )


def collect_stress() -> list[dict]:
    raw, _ = load_stress_cases()
    cases = []
    for item in raw:
        direction = item["direction"]
        assert item["synthetic_agent_authored"] is True
        cases.append({
            "id": item["case_id"],
            "direction": direction,
            "category": item["category"].replace("_", " "),
            "prior_context": " | ".join(item["previous_source"][-2:]),
            "source": item["source"],
            "source_type": "Synthetic slang/context stress test",
            "source_license": "Original development fixture",
            "group": item["case_id"],
        })
    assert Counter(x["direction"] for x in cases) == {"es-en": STRESS_N, "en-es": STRESS_N}
    return cases


def collect_dialogue() -> list[dict]:
    """Stream the identical hash-pinned first 8 MiB previously inspected."""
    from urllib.request import Request, urlopen
    from huggingface_hub import HfApi
    rev = HfApi().dataset_info(repo_id=REPO_ID).sha
    if rev != BABY_REV:
        raise RuntimeError("BilingualBabyLM revision changed; refusing unpinned source")
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{rev}/{DATAFILE}"
    with urlopen(Request(url, headers={"User-Agent": "BridgeCaption-v2-blind-review-examples/1.0"}), timeout=85) as resp:
        raw = resp.read(MAX_BYTES)
    if len(raw) != MAX_BYTES or hashlib.sha256(raw).hexdigest() != BABY_PREFIX_SHA:
        raise RuntimeError("BilingualBabyLM pinned 8MiB prefix changed")
    records = []
    for line in raw.rsplit(b"\n", 1)[0].splitlines():
        if line.strip():
            records.append(json.loads(line))
        if len(records) == 1500:
            break
    if len(records) != 1500:
        raise RuntimeError("Unexpected sample size")
    grouped = {"es-en": [], "en-es": []}
    # Use the publisher's hashed development groups; never use any private
    # final holdout, and avoid the first N groups that could be used for training.
    for row in records:
        for direction in grouped:
            parsed = parse_record(row, direction=direction)
            if parsed["status"] != "candidate_for_review" or parsed["split"] != "dev":
                continue
            for entry in parsed["rows"]:
                if len(entry["previous_source_turns"]) < 1:
                    continue
                text = entry["source"]
                history = [x["source"] for x in entry["previous_source_turns"][-2:]]
                if not safe(text) or any(not safe(x) for x in history):
                    continue
                grouped[direction].append({
                    "id": "syn_" + entry["case_id"],
                    "direction": direction,
                    "category": "Multi-turn dialogue (synthetic)",
                    "prior_context": " | ".join(history),
                    "source": text,
                    "source_type": "MIT synthetic dialogue: BabyLM dev groups",
                    "source_license": "MIT (upstream GPT-4 synthetic)",
                    "group": entry["original_group_id"],
                })
    selected = []
    for direction, pool in grouped.items():
        pool.sort(key=lambda row: hashlib.sha256(f"{SEED}|{direction}|{row['id']}".encode()).hexdigest())
        # Spread examples over as many different dialogues as possible.
        by_group: dict[str, list[dict]] = {}
        for item in pool:
            by_group.setdefault(item["group"], []).append(item)
        ordered = sorted(by_group, key=lambda g: hashlib.sha256(f"{SEED}|{direction}|{g}".encode()).hexdigest())
        used_texts = set()
        kept = []
        for depth in range(max(map(len, by_group.values()), default=0)):
            for group in ordered:
                if depth >= len(by_group[group]):
                    continue
                item = by_group[group][depth]
                k = norm(item["source"])
                if k in used_texts:
                    continue
                used_texts.add(k)
                kept.append(item)
                if len(kept) == DIALOGUE_N:
                    break
            if len(kept) == DIALOGUE_N:
                break
        if len(kept) != DIALOGUE_N:
            raise RuntimeError(f"Only {len(kept)} unique {direction} synthetic dialogues")
        selected.extend(kept)
    assert len(selected) == 2 * DIALOGUE_N
    return selected


def collect_general(existing: list[dict]) -> list[dict]:
    result = []
    for direction, locale in (("es-en", "es-ES"), ("en-es", "en-US")):
        prior = {norm(x["source"]) for x in existing if x["direction"] == direction}
        raw = aligned(locale, "validation")
        raw.sort(key=lambda row: hashlib.sha256(
            f"{SEED}|{direction}|{row['id']}".encode()).hexdigest())
        selected = []
        for entry in raw:
            t = str(entry["utt"]).strip()
            if not safe(t) or norm(t) in prior:
                continue
            prior.add(norm(t))
            selected.append({
                "id": f"massive_{direction}_{entry['id']}",
                "direction": direction,
                "category": "General speech / instructions",
                "prior_context": "",
                "source": t,
                "source_type": "Amazon MASSIVE author validation",
                "source_license": "CC BY 4.0",
                "group": str(entry["id"]),
            })
            if len(selected) == GENERAL_N:
                break
        if len(selected) != GENERAL_N:
            raise RuntimeError(f"Not enough publisher dev rows for {direction}")
        result.extend(selected)
    assert len(result) == 2 * GENERAL_N
    return result


def make_cases() -> list[dict]:
    selected = collect_stress()
    selected += collect_dialogue()
    selected += collect_general(selected)
    if len(selected) != TOTAL:
        raise RuntimeError(f"Expected {TOTAL} examples, got {len(selected)}")
    if Counter(x["direction"] for x in selected) != {"es-en": PER_DIRECTION, "en-es": PER_DIRECTION}:
        raise RuntimeError("Both directions must contain exactly 250 cases")
    ids = [x["id"] for x in selected]
    if len(set(ids)) != TOTAL:
        raise RuntimeError("Duplicate review example IDs")
    # Keep exact source repeats in tiny synthetic ambiguity contrast groups:
    # different past turns are essential for diagnosing no-context model.
    seen_full = set()
    for x in selected:
        k = (x["direction"], norm(x["source"]), norm(x["prior_context"]))
        if k in seen_full:
            raise RuntimeError("Duplicated identical source/context/direction")
        seen_full.add(k)
    random.Random(SEED).shuffle(selected)
    return selected


def generate_all(cases: list[dict]):
    """Generate real model output, never anything from synthetic references."""
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    results = {}
    timings = {}
    for direction in ("es-en", "en-es"):
        group = [x for x in cases if x["direction"] == direction]
        started = time.monotonic()
        model, activate = load_model(direction == "es-en")
        if direction == "es-en":
            activate()
        # This sends only the CURRENT utterance to the model.
        # History is visible to independent bilingual reviewers but must not
        # be concatenated into a model never trained for this context format.
        outputs = model([x["source"] for x in group])
        if len(outputs) != PER_DIRECTION or any(not x.strip() for x in outputs):
            raise RuntimeError("Real model produced wrong/blank output count")
        for item, output in zip(group, outputs):
            if item["id"] in results:
                raise RuntimeError("Two model generations share one case ID")
            results[item["id"]] = output
        timings[direction] = time.monotonic() - started
        print("REAL_MODEL_TRANSLATED_DIRECTION", direction, len(outputs), "seconds", round(timings[direction], 2), flush=True)
        del model, activate
        gc.collect()
    return results, timings


def save(cases, generated, timings):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    if OUT.exists():
        raise FileExistsError("Refuse to overwrite a prior 500-case evaluation")
    OUT.mkdir(parents=True)
    cols = ("Case ID", "Direction", "Category", "Prior context (reviewer only)",
            "Original spoken text", "Actual model translation",
            "Meaning preserved?", "Critical error?", "Error type(s)",
            "Reviewer notes", "Reviewer ID", "Source type")
    rows = []
    for x in cases:
        rows.append({
            "Case ID": x["id"], "Direction": x["direction"],
            "Category": x["category"], "Prior context (reviewer only)": x["prior_context"],
            "Original spoken text": x["source"], "Actual model translation": generated[x["id"]],
            "Meaning preserved?": "", "Critical error?": "", "Error type(s)": "",
            "Reviewer notes": "", "Reviewer ID": "", "Source type": x["source_type"],
        })

    wb = Workbook()
    intro = wb.active
    intro.title = "READ ME"
    intro.sheet_view.showGridLines = False
    intro.column_dimensions["A"].width = 34
    intro.column_dimensions["B"].width = 104
    intro["A1"] = "BRIDGECAPTION  /  500 REAL MODEL ANSWERS"
    intro["A1"].font = Font(name="Aptos Display", size=18, bold=True, color="FFFFFF")
    intro["A1"].fill = PatternFill("solid", fgColor="172B38")
    intro.merge_cells("A1:B1")
    intro.row_dimensions[1].height = 35
    instructions = [
        ("Purpose", "Bilingual development review of 500 genuinely GENERATED model translations. No answer keys or references are provided."),
        ("Important", "NOT a final independent holdout. Examples are drawn from previously used synthetic development fixtures, public synthetic dialogue groups, and MASSIVE validation."),
        ("Review each row", "Read original speech and any preceding context, then judge only the model's current-utterance translation."),
        ("Pass", "Meaning, intent, numbers, names, negation, slang/idioms, and key details are preserved. Natural paraphrases count."),
        ("Fail", "Wrong idiom or slang sense, invented/omitted factual details, wrong negation/numbers/names/speaker, or misleading meaning."),
        ("Unrateable", "Original utterance too ambiguous to decide with available context; do not guess."),
        ("Meaning preserved?", "Select Pass, Fail, or Unrateable (blue input column)."),
        ("Critical error?", "Select Yes or No. Critical errors include flipped negation, wrong number/name, invented claims, or wrong speaker."),
        ("Error type(s)", "Use short codes: slang, context, idiom, negation, number, name, speaker, hallucination, omission, noisy speech, domain, other."),
        ("ES → EN model", "Actual sequentially trained MASSIVE-derived v2 Spanish checkpoint; pinned SHA-256 " + ES_EN_ADAPTER_SHA),
        ("EN → ES model", "Original pinned Helsinki-NLP/opus-mt-en-es model (NO trained ES→EN adapter); revision " + EN_ES_SHA),
        ("Context limitation", "Existing models translate current speech ONLY. Previous context is shown to reviewer, but was NOT injected into the model."),
        ("Dataset source A", "40 pre-existing AI-authored synthetic dev stress cases; content is not independent human test data."),
        ("Dataset source B", "300 MIT bilingual BabyLM synthetic multi-turn development cases, source: https://huggingface.co/datasets/lindazeng979/bilingual-babyLM"),
        ("Dataset source C", "160 Amazon MASSIVE utterances from publisher VALIDATION, licensed CC BY 4.0: https://huggingface.co/datasets/AmazonScience/massive"),
        ("Do not use as final claim", "500 cases can help identify weaknesses but cannot establish 95% independently validated meaning preservation."),
        ("No answer key", "Workbook, CSV and JSONL include ONLY source text, context, authentic bot output and empty reviewer fields. No reference translations."),
    ]
    for idx, (key, value) in enumerate(instructions, 3):
        a, b = intro.cell(idx, 1, key), intro.cell(idx, 2, value)
        a.font = Font(name="Aptos", size=11, bold=True, color="3A596B")
        b.font = Font(name="Aptos", size=11, color="203948")
        a.alignment = b.alignment = Alignment(vertical="center", wrap_text=True)
        intro.row_dimensions[idx].height = 33 if len(value) > 100 else 26
        if idx % 2:
            for cell in (a, b):
                cell.fill = PatternFill("solid", fgColor="F0F5F7")
    intro.freeze_panes = "A3"

    ws = wb.create_sheet("500 Model Outputs")
    ws.sheet_view.showGridLines = False
    ws.append(cols)
    for row in rows:
        ws.append([row[k] for k in cols])
    widths = [25, 14, 27, 58, 64, 70, 21, 18, 25, 51, 19, 43]
    for i, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = width
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor="172B38")
        cell.font = Font(name="Aptos", bold=True, size=11, color="FFFFFF")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 36
    for i in range(2, TOTAL + 2):
        ws.row_dimensions[i].height = 53
        bg = "FFFFFF" if i % 2 == 0 else "F2F6F8"
        for j in range(1, len(cols) + 1):
            cell = ws.cell(i, j)
            cell.fill = PatternFill("solid", fgColor=("EAF2FF" if j in (7, 8, 9, 10, 11) else bg))
            cell.font = Font(name="Aptos", size=10, color=("1F55A4" if j in (7, 8, 9, 10, 11) else "263F4C"))
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.cell(i, 2).font = Font(name="Aptos", bold=True, size=10, color="137A68")
    ws.freeze_panes = "F2"
    ws.auto_filter.ref = f"A1:L{TOTAL+1}"
    passlist = DataValidation(type="list", formula1='"Pass,Fail,Unrateable"', allow_blank=True)
    criticallist = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    ws.add_data_validation(passlist)
    ws.add_data_validation(criticallist)
    passlist.add(f"G2:G{TOTAL+1}")
    criticallist.add(f"H2:H{TOTAL+1}")

    stats = wb.create_sheet("Reviewer Summary")
    stats.sheet_view.showGridLines = False
    stats.column_dimensions["A"].width = 36
    stats.column_dimensions["B"].width = 30
    stats["A1"], stats["B1"] = "REVIEW PROGRESS", "Value"
    for c in stats[1]:
        c.fill = PatternFill("solid", fgColor="172B38")
        c.font = Font(name="Aptos", bold=True, color="FFFFFF", size=12)
    analysis = [
        ("Cases supplied", "=COUNTA('500 Model Outputs'!A2:A501)"),
        ("Reviewer-rated cases", '=COUNTIF(\\'500 Model Outputs\\'!G2:G501,"Pass")+COUNTIF(\\'500 Model Outputs\\'!G2:G501,"Fail")+COUNTIF(\\'500 Model Outputs\\'!G2:G501,"Unrateable")'),
        ("Passes", '=COUNTIF(\\'500 Model Outputs\\'!G2:G501,"Pass")'),
        ("Fails", '=COUNTIF(\\'500 Model Outputs\\'!G2:G501,"Fail")'),
        ("Unrateable", '=COUNTIF(\\'500 Model Outputs\\'!G2:G501,"Unrateable")'),
        ("Overall pass rate (reviewed only)", '=IFERROR(B4/B3,"")'),
        ("Critical errors", '=COUNTIF(\\'500 Model Outputs\\'!H2:H501,"Yes")'),
        ("ES→EN Passes", '=COUNTIFS(\\'500 Model Outputs\\'!B2:B501,"es-en",\\'500 Model Outputs\\'!G2:G501,"Pass")'),
        ("ES→EN Rated", '=COUNTIFS(\\'500 Model Outputs\\'!B2:B501,"es-en",\\'500 Model Outputs\\'!G2:G501,"Pass")+COUNTIFS(\\'500 Model Outputs\\'!B2:B501,"es-en",\\'500 Model Outputs\\'!G2:G501,"Fail")+COUNTIFS(\\'500 Model Outputs\\'!B2:B501,"es-en",\\'500 Model Outputs\\'!G2:G501,"Unrateable")'),
        ("ES→EN reviewed pass rate", '=IFERROR(B9/B10,"")'),
        ("EN→ES Passes", '=COUNTIFS(\\'500 Model Outputs\\'!B2:B501,"en-es",\\'500 Model Outputs\\'!G2:G501,"Pass")'),
        ("EN→ES Rated", '=COUNTIFS(\\'500 Model Outputs\\'!B2:B501,"en-es",\\'500 Model Outputs\\'!G2:G501,"Pass")+COUNTIFS(\\'500 Model Outputs\\'!B2:B501,"en-es",\\'500 Model Outputs\\'!G2:G501,"Fail")+COUNTIFS(\\'500 Model Outputs\\'!B2:B501,"en-es",\\'500 Model Outputs\\'!G2:G501,"Unrateable")'),
        ("EN→ES reviewed pass rate", '=IFERROR(B12/B13,"")'),
    ]
    for i,(name, formula) in enumerate(analysis, 2):
        stats.cell(i,1,name)
        stats.cell(i,2,formula)
        stats.cell(i,1).font = Font(name="Aptos", bold=i in (2,3), size=11)
        stats.cell(i,2).font = Font(name="Aptos", size=11, color="000000")
        if i in (7, 11, 14):
            stats.cell(i,2).number_format = "0.0%"
    stats["A17"] = "These reviewer-progress calculations are NOT independent final accuracy."
    stats.merge_cells("A17:B17")
    stats["A17"].font = Font(name="Aptos", italic=True, color="BA6E25", size=11)
    wb.active = 0

    xlsx_path = OUT / "BridgeCaption_500_Real_Model_Translations_Review.xlsx"
    wb.save(xlsx_path)
    csv_path = OUT / "BridgeCaption_500_Real_Model_Translations_Review.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    jsonl_path = OUT / "BridgeCaption_500_Real_Model_Translations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False)+"\n")
    result = {
        "status": "500_actually_generated_model_predictions",
        "cases": len(rows),
        "by_direction": dict(Counter(x["direction"] for x in cases)),
        "by_origin": dict(Counter(x["source_type"] for x in cases)),
        "unique_case_ids": len(set(x["id"] for x in cases)),
        "unique_source_context_direction": len(set(
            (x["direction"],norm(x["source"]),norm(x["prior_context"])) for x in cases
        )),
        "unique_output_strings": len(set(norm(x["Actual model translation"]) for x in rows)),
        "research_only_dev_examples": True,
        "never_used_final_holdout": True,
        "reference_translations_in_exports": False,
        "actual_inference_executed": True,
        "es_en_adapter_sha256": ES_EN_ADAPTER_SHA,
        "en_es_base_revision": EN_ES_SHA,
        "prior_context_shown_to_reviewer_not_model": True,
        "execution_seconds_by_direction": timings,
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                         for file in (xlsx_path, csv_path, jsonl_path)},
    }
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FIVE_HUNDRED_ACTUAL_MODEL_ANSWERS", json.dumps(result), flush=True)


def main():
    started = time.monotonic()
    cases = make_cases()
    print("REVIEW_CASE_COMPOSITION", dict(Counter(x["source_type"] for x in cases)), flush=True)
    generated, timings = generate_all(cases)
    save(cases, generated, timings)
    print("TOTAL_RUN_SECONDS", round(time.monotonic()-started, 2), flush=True)


if __name__ == "__main__":
    main()
