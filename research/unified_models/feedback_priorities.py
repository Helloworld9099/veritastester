"""Translate qualitative 500-response review into a defensible next data budget.

This does NOT convert review flags into accuracy or into supervised training
examples. It chooses research priorities for NEW human-verified, licensed,
group-disjoint examples; reviewed cases remain development-only and cannot be
reused as new fresh final holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

from research.unified_models.registry import ROOT

SUMMARY = (ROOT / "research/es_en_v2/reviewer_feedback_20261120"
           / "processed/summary.json")

# Priority reflects *consequence of error*, not a pretend prevalence estimate.
PRIORITIES = {
    "critical_meaning_or_role": (5, "negation, tense/time, person, names and numbers"),
    "omitted_current_utterance_content": (5, "multi-sentence omissions and short ASR fragments"),
    "repetition_loop": (4, "pathological generation and meaning-safe recovery"),
    "idiom_or_slang_interpretation": (4, "idioms, social slang and regional senses with past-only dialogue"),
    "wrong_lexical_sense_or_action": (4, "polysemy and action grounding"),
    "wrong_speaker_gender_person": (4, "speaker grounding and pronoun roles"),
    "names_brands_entities": (4, "entity retention and proper-name transliteration"),
    "sentence_type_pragmatics": (3, "question versus statement and pragmatic intent"),
    "grammar_fluency_or_untranslated": (2, "natural fluency and loanword handling"),
    "register_formality": (2, "child-directed informal vs formal address"),
    "punctuation_spelling": (1, "Spanish punctuation and accent conventions"),
}


def derive_development_data_priorities(summary: dict) -> dict:
    if summary.get("input_model_responses") != 500:
        raise ValueError("Expected pinned 500-response review")
    if (summary.get("per_row_pass_fail_labels_collected") != 0
        or summary.get("independently_evaluated_95_percent") is not False
        or summary.get("training_examples_added") != 0):
        raise ValueError("Expected partial qualitative review, not human gold")
    by_direction = {"es-en": [], "en-es": []}
    by_tag = summary.get("category_direction_counts", {})
    if set(by_tag) != set(PRIORITIES):
        raise ValueError("Review categories or taxonomy changed")
    for tag, (priority, goal) in PRIORITIES.items():
        counts = by_tag[tag]
        for direction in by_direction:
            flags = counts.get(direction, 0)
            if flags:
                by_direction[direction].append({
                    "category":tag,
                    "reported_row_flags":flags,
                    "impact_weight":priority,
                    "risk_index_not_accuracy":flags*priority,
                    "data_objective":goal,
                    "new_training_examples_available":0,
                    "source":"reviewer-directed development feedback only",
                })
    for direction, priority in by_direction.items():
        priority.sort(key=lambda row:(-row["risk_index_not_accuracy"],row["category"]))
    return {
        "status":"review_informed_not_executed_data_priorities",
        "language_plan":by_direction,
        "new_verified_examples_added":0,
        "all_500_previous_examples_held_out_from_new_training":True,
        "no_final_holdout_scored":True,
        "warning":"This is a severity-weighted ordering of reported concerns, "
                  "not a distribution of real-world error rates. A new, licensed, "
                  "independently verified conversational corpus is still required "
                  "to train and measure meaning improvement.",
    }


def main():
    plan = derive_development_data_priorities(json.loads(SUMMARY.read_text()))
    print(json.dumps(plan,indent=2))


if __name__=="__main__":
    main()
