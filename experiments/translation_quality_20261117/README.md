# A defensible "90% translation correctness" gate

**Purpose:** Measure whether any candidate French->English or Spanish->English model reaches the user-requested **90% semantically correct translations**. No model is certified by chrF, BLEU, train loss, or a small memorized test. A model passing this tool's gate is a *research milestone*, not proof of overall production readiness.

## Independent evaluation requirements

1. Lock the model checkpoint SHA-256 and the evaluation-manifest SHA-256 **before translating** any independent test examples.
2. Reserve **at least 200 genuinely unseen** examples per language from independently sourced, permitted conversational speech. Include at least **20 examples each** of: everyday speech, idioms, slang, multi-turn context, and technical statements; include examples from at least **20 independent speakers/conversations** or disjoint documents.
3. At least **two fluent bilingual human evaluators**, blind to model identity and the development results, must independently mark whether the translation accurately preserves intended meaning without materially wrong additions/omissions. Disagreements require an independent adjudication.
4. Keep original source text, human references, generated translations, and individual reviewers' private annotations **outside Git**. This script reads JSONL locally and writes **only aggregate scoring**.
5. Don't revise the model, sample, judgment rules, or thresholds after inspecting results and then call a rerun "independent." Any retuned model requires a newly reserved test cohort.

A **measured observed rate** of 90/100 or 180/200 is not sufficient evidence for a **reliably at-least-90%** target. This gate requires at least 200 cases, category/group coverage, two raters, and a **95% Wilson lower confidence bound of at least 90%**. Statistics cannot prove unbiased sampling, honest reviewer independence, no pretrained-data leakage, or performance in all applications. Those require methodological review.

## Annotated JSONL shape

Each private line has this schema (replace the literal placeholders with real SHA-256 hashes, actual independently collected IDs, and **actual** bilingual human ratings; the sample here is *illustrative*, not real evidence):

```json
{"sample_id":"sample-0001","language":"french","split":"independent_test","checkpoint_sha256":"<64 hex characters>","evaluation_manifest_sha256":"<64 hex characters>","source_group_sha256":"<64 hex characters>","category":"idiom","rater_a_adequate":true,"rater_b_adequate":true}
```

When ratings disagree, include `"adjudicated_adequate": true` or `false` after an independent third evaluation. Never fabricate or fill reviewer judgments with synthetic labels.

## Run

```bash
python experiments/translation_quality_20261117/score_human.py /secure/location/heldout_ratings.jsonl --output /secure/location/aggregated_heldout_result.json
python -m unittest discover -s experiments/translation_quality_20261117/tests -p 'test_*.py' -v
```

All results should retain model and dataset provenance. No new human annotations were produced by creating this script. Current LoRA development and frozen-test chrF/BLEU scores do **not** demonstrate 90% adequacy.
