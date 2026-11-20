"""Adaptive model promotion controller for human-verified meaning quality.

Automatic chrF/BLEU may select an *experimental* checkpoint for additional
research, but cannot authorize product promotion. This module has no external
network, billing or model-generation actions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SUBGROUPS=("general","context","idiom","regional_slang",
           "domain_terms","noisy_transcript")
DEFAULT_TRANSLATION_P95_BUDGET_SECONDS=2.0
DEFAULT_PEAK_RSS_KIB_BUDGET=4*1024*1024
MAX_SUBGROUP_REGRESSION_POINTS=1.0
MAX_CRITICAL_ERROR_RATE=.01


@dataclass(frozen=True)
class HumanDevResult:
    """Requires independent human-reviewed development data; NOT chrF/COMET."""
    direction: str
    reviews_n: int
    overall_pass: float
    by_subgroup: dict
    critical_error_rate: float
    reviewer_agreement: float
    dataset_sha256: str
    blinded: bool

    def validate(self):
        if self.direction not in {"es-en","en-es"} or self.reviews_n<200:
            raise ValueError("Insufficient direction-specific human development reviews")
        if not self.blinded:
            raise ValueError("Human development review must be blinded")
        if not isinstance(self.dataset_sha256,str) or len(self.dataset_sha256)!=64:
            raise ValueError("Unverified dev corpus identity")
        for value in [self.overall_pass,self.critical_error_rate,self.reviewer_agreement]:
            if not isinstance(value,(float,int)) or not 0<=value<=1:
                raise ValueError("Invalid human evaluation result")
        if set(self.by_subgroup)!=set(SUBGROUPS):
            raise ValueError("Missing subgroup human ratings")
        for group in SUBGROUPS:
            item=self.by_subgroup[group]
            if not isinstance(item,dict) or item.get("n",0)<20 or not 0<=item.get("pass_rate",-1)<=1:
                raise ValueError("Insufficient reviewed subgroup examples")


def decide_promotion(parent: HumanDevResult | None,
                     candidate: HumanDevResult | None,
                     *, parent_p95: float | None=None,
                     candidate_p95: float | None=None,
                     candidate_peak_rss_kib: int | None=None,
                     seed_variation_gain_floor: float | None=None,
                     code_passed: bool=False) -> dict:
    """No AI score can substitute for reviewer judgments or complete controls."""
    blockers=[]
    if parent is None or candidate is None:
        return {"promote":False,"status":"blocked_missing_independent_human_dev_labels",
                "reasons":["Cannot infer semantic meaning from chrF/BLEU/COMET or synthetic labels"]}
    parent.validate();candidate.validate()
    if parent.direction!=candidate.direction or parent.dataset_sha256!=candidate.dataset_sha256:
        blockers.append("Human dev comparisons are not paired on the same frozen direction/dataset")
    if not code_passed:
        blockers.append("Integration and regression tests incomplete")
    if candidate.critical_error_rate>parent.critical_error_rate:
        blockers.append("Critical error rate increased")
    if candidate.critical_error_rate>MAX_CRITICAL_ERROR_RATE:
        blockers.append("Critical error rate above absolute ceiling")
    for subgroup in SUBGROUPS:
        before=parent.by_subgroup[subgroup]["pass_rate"]
        after=candidate.by_subgroup[subgroup]["pass_rate"]
        if (before-after)*100>MAX_SUBGROUP_REGRESSION_POINTS+1e-9:
            blockers.append(f"Regression in human-judged {subgroup} exceeded 1 percentage point")
    if (seed_variation_gain_floor is None or
        candidate.overall_pass-parent.overall_pass<=seed_variation_gain_floor):
        blockers.append("No demonstrated overall gain above seed variation")
    if (candidate_p95 is None or parent_p95 is None or
        candidate_p95>DEFAULT_TRANSLATION_P95_BUDGET_SECONDS or
        candidate_p95>parent_p95*1.1):
        blockers.append("Unacceptable or unmeasured translation p95")
    if (candidate_peak_rss_kib is None or candidate_peak_rss_kib>
        DEFAULT_PEAK_RSS_KIB_BUDGET):
        blockers.append("Unacceptable or unmeasured peak RSS")
    return {"promote":not blockers,
            "status":"approved_by_prespecified_dev_gate_only" if not blockers else "rejected_or_incomplete",
            "reasons":blockers,
            "final_human_holdout_evaluated":False,
            "production_authorization_granted":False}


def next_experiment(dev_suspected_tags: dict[str,int], available_verified_train: dict[str,int]):
    """Rank *development-only* risk categories; no synthetic unreviewed training."""
    routes=(
        ("negation_disagreement","critical_correctness"),
        ("number_disagreement","critical_correctness"),
        ("contains_slang_needs_review","regional_slang"),
        ("contains_idiom_needs_review","idiom"),
        ("context_sensitivity_needs_review","context"),
        ("possibly_omitted_content","general"),
    )
    ordered=sorted(routes,key=lambda r:(-int(dev_suspected_tags.get(r[0],0)),r[0]))
    selected=next((risk for tag,risk in ordered if dev_suspected_tags.get(tag,0)>0),"general")
    available=int(available_verified_train.get(selected,0))
    return {
        "target":selected,
        "can_train":available>=128,
        "verified_train_examples_in_category":available,
        "plan":"Bilingual review or discover new permitted data first" if available<128
                else "Controlled 1-seed pilot with 20%-40% general replay and dev subgroup gate",
        "no_final_holdout_used":True,
    }


def write_immutable_receipt(report: dict, path: Path):
    """Serialize the checkpoint ledger without silent overwrites."""
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("Checkpoint receipt already exists")
    parent=report.get("parent_sha256")
    candidate=report.get("checkpoint_sha256")
    if not all(isinstance(value,str) and len(value)==64 for value in (parent,candidate)):
        raise ValueError("Every candidate needs complete parent and output SHA256")
    if report.get("human_final_95_verified") is not False:
        raise ValueError("A training receipt may never assert the independent final 95% gate")
    payload=json.dumps(report,indent=2,sort_keys=True)
    path.write_text(payload,encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()
