"""Budget-aware, anti-dead-horse experiment selection for BridgeCaption.

Never turn BLEU/chrF into "95%" semantic accuracy; never keep updating a
checkpoint because the optimizer can still run. Decisions require measured
multi-route quality, critical-error proxies, wall-clock, and rollback.

Run independently of model weights, so it can decide NOT to spend compute.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE_DIR=ROOT/"research/madlad400"


@dataclass(frozen=True)
class Quality:
    examples:int
    chrf:float
    bleu:float
    critical_errors:int
    latency_p95_seconds:float
    ram_mib:int


REQUIRED_CURRENT_DIRECTIONS=("es-en","fr-en","en-es","en-fr","es-fr","fr-es")
EXPANSION_SENTINELS=("en-de","en-pt","en-ar","en-zh")
FAILURE_METHODS=frozenset({"continue_same_schedule","more_random_Tatoeba_updates"})
CANDIDATES=(
    "int8_ctranslate2_pretrained_inference",
    "targeted_quality_filter_and_deduplicate",
    "balanced_replay_vs_low_resource_adaptation",
    "last_decoder_lora_vs_encoder_decoder_lora",
    "past_only_human_verified_context_training",
    "licensed_teacher_distillation_after_dev_label_check",
    "checkpoint_average_only_if_validated",
    "resample_more_diverse_dev_groups",
)


def parse_quality(row:dict)->Quality:
    if any(k not in row for k in ("n","chrF_not_accuracy","BLEU_not_accuracy",
                                 "critical_errors","latency_p95_seconds",
                                 "ram_mib")):
        raise ValueError("Cannot promote models based on missing critical-error, latency or memory evidence")
    q=Quality(examples=int(row["n"]),chrf=float(row["chrF_not_accuracy"]),
              bleu=float(row["BLEU_not_accuracy"]),
              critical_errors=int(row["critical_errors"]),
              latency_p95_seconds=float(row["latency_p95_seconds"]),
              ram_mib=int(row["ram_mib"]))
    if (q.examples<1 or q.critical_errors<0 or q.critical_errors>q.examples
        or not all(math.isfinite(v) for v in
                   (q.chrf,q.bleu,q.latency_p95_seconds))
        or q.latency_p95_seconds<0 or q.ram_mib<0
        or not 0<=q.chrf<=100 or not 0<=q.bleu<=100):
        raise ValueError("Invalid measured quality or resource usage")
    return q


def compare(before:dict,after:dict,*,min_n:int=24,
            max_chrf_drop:float=1.0,max_bleu_drop:float=1.0,
            max_latency_s:float=3.0,max_ram_mib:int=8192)->dict:
    """Scientific gate. Human approval is a separate, mandatory stage."""
    expected=set(REQUIRED_CURRENT_DIRECTIONS+EXPANSION_SENTINELS)
    blockers=[]
    if set(before)!=expected or set(after)!=expected:
        blockers.append("missing_six_original_or_four_expansion_languages")
    routes=sorted(expected & set(before) & set(after))
    quality_deltas={}
    for route in routes:
        parent,child=parse_quality(before[route]),parse_quality(after[route])
        quality_deltas[route]={
            "chrF_change_not_meaning_accuracy":round(child.chrf-parent.chrf,4),
            "BLEU_change_not_meaning_accuracy":round(child.bleu-parent.bleu,4),
            "critical_error_change":child.critical_errors-parent.critical_errors,
            "p95_seconds":child.latency_p95_seconds,
            "ram_mib":child.ram_mib}
        if parent.examples<min_n or child.examples<min_n:
            blockers.append(route+":too_few_development_examples")
        if child.chrf-parent.chrf < -max_chrf_drop:
            blockers.append(route+":chrF_regression")
        if child.bleu-parent.bleu < -max_bleu_drop:
            blockers.append(route+":BLEU_regression")
        if child.critical_errors>parent.critical_errors:
            blockers.append(route+":critical_error_regression")
        if child.latency_p95_seconds>max_latency_s:
            blockers.append(route+":too_slow_for_live_captioning")
        if child.ram_mib>max_ram_mib:
            blockers.append(route+":too_much_ram")
    if routes:
        mean=sum(quality_deltas[route]["chrF_change_not_meaning_accuracy"]
                 for route in routes)/len(routes)
        if mean<0.25:
            blockers.append("insufficient_mean_chrF_gain")
    else:
        mean=None
        blockers.append("no_valid_development_measurements")
    return {
        "research_checkpoint_gate_passed":not blockers,
        "production_approval":False,
        "human_95_meaning_accuracy_verified":False,
        "route_regressions":quality_deltas,
        "mean_chrF_change_not_accuracy":round(mean,4) if mean is not None else None,
        "blockers":sorted(set(blockers)),
        "warning":"Even a passing 10-route quality gate is NOT independent human adequacy or a production release."
    }


def prioritize_next(*,baseline_exists:bool,live_latency_p95_s:float|None,
                    candidate_results:list[dict],remaining_minutes:float,
                    dataset_hard_cases_verified:bool)->dict:
    """Prioritize cheapest new information, prohibit repeated failed methods."""
    if remaining_minutes<=0:
        return {"next":"stop_budget_exhausted","paid_spend_usd":0,
                "allow_new_training":False,"reason":"No authorized compute remains"}
    used={r.get("method") for r in candidate_results if r.get("status")=="completed"}
    rejected={r.get("method") for r in candidate_results if r.get("status")=="rejected"}
    if not baseline_exists:
        action="measure_pretrained_MADLAD_on_development"
        reason="First establish native pretrained quality; no blind LoRA training"
    elif live_latency_p95_s is None or live_latency_p95_s>3.0:
        action="int8_ctranslate2_pretrained_inference"
        reason="Fix translation latency and memory before more gradients"
    elif not dataset_hard_cases_verified:
        action="mine_and_bilingually_verify_context_slang_errors"
        reason="No verified contextual references; more supervised steps would reinforce isolated utterances"
    elif any(m in rejected for m in FAILURE_METHODS):
        action="last_decoder_lora_vs_encoder_decoder_lora"
        reason="Prior strategy failed; compare parameter placement, not more identical steps"
    elif "balanced_replay_vs_low_resource_adaptation" not in used:
        action="balanced_replay_vs_low_resource_adaptation"
        reason="Test whether low-resource adaptation forgets existing languages"
    else:
        action="cost_aware_targeted_ablation_after_new_dev_evidence"
        reason="Re-evaluate hard cases and choose a genuinely different data/training intervention"
    return {"next":action,"reason":reason,"remaining_minutes":remaining_minutes,
            "allow_new_training":bool(baseline_exists and
                                      dataset_hard_cases_verified and
                                      live_latency_p95_s is not None and
                                      live_latency_p95_s<=3.0 and
                                      remaining_minutes>=10),
            "same_failed_schedule_must_not_be_repeated":True,
            "paid_spend_usd":0,
            "claim_95_percent":False}


def analyze_real_prior_results()->dict:
    """Use completed EXPERIMENTS, never claim a planned pilot is completed."""
    prev=ROOT/"research/conversation_growth/experiment_001/report.json"
    if not prev.is_file():
        return {"status":"previous_experiment_report_unavailable"}
    report=json.loads(prev.read_text())
    if (report.get("status")!="actual_additional_licensed_multilingual_training_completed"
        or report.get("actual_optimizer_updates")!=72):
        raise RuntimeError("Unexpected historical training receipt")
    snapshots=report["all_candidate_evaluations"]
    eligible=[name for name,v in snapshots.items()
              if v.get("eligible_for_research_only")]
    all_blocked={name:v["automatic_gate"]["blocked_by"]
                 for name,v in snapshots.items()}
    return {
        "status":"analyzed_actual_historical_training_failure",
        "actual_previous_optimizer_updates":72,
        "historical_accepted_candidates":eligible,
        "historical_rejected_and_why":all_blocked,
        "historical_selected_checkpoint_unchanged":report["selected_sha256"],
        "lessons":[
            "No blindly repeating 72 updates on Tatoeba + MASSIVE; zero accepted candidates",
            "Current 1-sentence-per-direction MADLAD pilot is inadequate for quality claims",
            "Use more diverse licensed, human-verified context/sense data before changing learning rate",
            "Test INT8 live-caption throughput and 10-route forgetting before considering release",
        ],
        "does_not_claim_new_training":True
    }


def main():
    prev=analyze_real_prior_results()
    decision=prioritize_next(
        baseline_exists=False,live_latency_p95_s=None,
        candidate_results=[],remaining_minutes=45,dataset_hard_cases_verified=False)
    result={"status":"completed_optimization_policy_analysis_not_training",
            "prior":prev,"first_next_experiment":decision,
            "candidate_methods":CANDIDATES,
            "language_gate_current":REQUIRED_CURRENT_DIRECTIONS,
            "language_gate_beyond_english_spanish_french":EXPANSION_SENTINELS,
            "no_paid_compute":True,
            "no_automatic_accuracy_promotion":True}
    out=BASE_DIR/"optimization_analysis.json"
    if out.exists():
        raise FileExistsError("Do not overwrite optimization decision evidence")
    out.write_text(json.dumps(result,indent=2)+"\n")
    print("ACTUAL_PREVIOUS_FAILURES_ANALYZED_NEW_EFFICIENT_PATH",json.dumps({
        "prior_successful_trainings":prev.get("historical_accepted_candidates"),
        "next":decision["next"],
        "automatically_start_same_failed_training":False,
        "human_accuracy_claim":False}),flush=True)


if __name__=="__main__":
    main()
