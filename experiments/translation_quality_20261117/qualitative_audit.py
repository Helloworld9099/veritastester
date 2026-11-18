#!/usr/bin/env python3
"""Human-readable AI-assisted error audit on ALREADY CONSUMED TICO-19 tests.

Diagnostic only: neither these 12 sentences nor test metrics may be used to
fine-tune, select a checkpoint, or claim an independent 90% adequacy result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "experiments/tico19_20261117"),
                str(REPO / "experiments/local_adapter_review_20261117")]
from eval_tico import DATA_FILES, URL_ROOT, select_rows  # noqa: E402
from translate import translate  # noqa: E402

AUDIT_SAMPLE_N = 12


def choose_audit_samples(pairs, size=AUDIT_SAMPLE_N):
    """Stable illustrative sample, distinct indices, independent of model scores."""
    if len(pairs) < size:
        raise ValueError("Insufficient existing tested pairs")
    positions = sorted(range(len(pairs)),
                       key=lambda i: hashlib.sha256(
                           f"already-consumed-audit-20261118|{i}".encode()).hexdigest())
    return [pairs[i] for i in positions[:size]], positions[:size]


def inspect(language: str):
    if language not in DATA_FILES:
        raise ValueError("Unknown language")
    url = URL_ROOT + "/" + DATA_FILES[language][0]
    with urlopen(Request(url, headers={"User-Agent":"BridgeCaption-diagnostic-only/1.0"}),
                 timeout=60) as stream:
        data = stream.read(5 * 1024 * 1024 + 1)
    if len(data) > 5 * 1024 * 1024:
        raise ValueError("Benchmark source changed size")
    rows, ids = select_rows(data, language)
    audited, positions = choose_audit_samples(rows)
    sources = [s for s,_ in audited]
    refs = [ref for _,ref in audited]
    scores = {}
    for variant in ("base", "massive3"):
        prediction = translate(language, variant, sources,
                               Path("/tmp/bridgecaption-audit-model-cache"), REPO)
        if len(prediction["translations"]) != AUDIT_SAMPLE_N:
            raise RuntimeError("Incomplete inference for qualitative audit")
        scores[variant] = prediction["translations"]
        print("AUDIT_MODEL",language,variant,prediction["adapter_sha256"],flush=True)
    print("AUDIT_WARNING TICO-19 sample ALREADY evaluated and consumed, "
          "not a new blind holdout; never use these cases to train or select.",flush=True)
    for k, (source, ref) in enumerate(audited):
        record={
            "language":language,
            "audit_position_in_previously_scored_set":positions[k],
            "source":source,
            "reference":ref,
            "unmodified_model":scores["base"][k],
            "stage3_model":scores["massive3"][k],
            "not_human_verified":True,
        }
        print("REVIEW_CASE "+json.dumps(record,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", choices=tuple(DATA_FILES),required=True)
    inspect(p.parse_args().language)
