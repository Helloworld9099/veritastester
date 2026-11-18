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


## Blinded human review packet (new)

The `collect_reviews.py` utility prepares private, frozen **two-rater** review packets for a trained custom French or Spanish model. It is the bridge from automatic chrF/BLEU metrics to the actual 90% bilingual semantic adequacy goal. **No real human ratings are invented or uploaded.**

First obtain **at least 200 genuinely new, unseen examples** for each language, including at least 20 cases in each of the five categories and at least 20 independent conversation/speaker/document groups. Do not use the MASSIVE, Tatoeba, QED, GlobalVoices, Books, or other examples already seen during model training, checkpoint selection, or earlier scoring. Get the rights to use each utterance, and don't include private personal information without authorization.

Run the frozen checkpoint once to translate each example, then prepare a local JSONL with one record per input:

```json
{"sample_id":"conversation-01-sentence-001","language":"french","source_group_sha256":"<64 hex of the independent conversation group>","category":"idiom","source_text":"J'ai le cafard.","candidate_translation":"I'm feeling down."}
```

Replace all placeholders with **actual separately gathered source phrases and the actual frozen model output**; never manually correct the prediction. Hash group identifiers, keep private input/output text out of Git and public artifacts, and retain exact checkpoint and input-dataset provenance.

```bash
python experiments/translation_quality_20261117/collect_reviews.py prepare \
  /secure/evaluation/predictions.jsonl \
  --checkpoint-sha256 <64-character-final-adapter-sha> \
  --output-dir /secure/evaluation/packet
```

The generated `manifest.json` contains only hashed case identity, grouping metadata, frozen checkpoint provenance, and a manifest SHA-256. The `rater_a.jsonl` and `rater_b.jsonl` files contain **source + candidate translation, with no reference translation and no model variant label**. Give these files to two **different fluent bilingual raters** who independently mark `"adequate": true` only if the intended meaning is preserved with no material addition, omission or mistranslation; otherwise `false`. Keep the data private. A rater should not alter the sample IDs or hashes.

For disagreements, obtain a separate third bilingual judgment in a JSONL file containing the conflicting `sample_id`, `case_hash` and `adequate` boolean. Then aggregate without writing raw sentences or individual human labels to the public repository:

```bash
python experiments/translation_quality_20261117/collect_reviews.py score \
  --manifest /secure/evaluation/packet/manifest.json \
  --rater-a /secure/evaluation/packet/rater_a.jsonl \
  --rater-b /secure/evaluation/packet/rater_b.jsonl \
  --adjudication /secure/evaluation/packet/third.jsonl \
  --output /secure/evaluation/aggregate.json
```

Omit the adjudication argument when there are no disagreements. The score is the existing pre-registered gate from `score_human.py`: at least 200 cases, coverage of all five categories, at least 20 independent source groups, and a **95% Wilson lower bound >= 0.90**. Two human raters must actually be independent; the tool cannot verify their real-world identities or the absence of overlap with any pretrained model. Passing a small artificial unit test does not constitute evidence of model quality.

**Nothing in the repository currently establishes 90% independent bilingual human-rated adequacy.**

## Conversational adequacy must include slang and context

The project requirements in [docs/CONTEXT_SLANG_TRANSLATION_SPEC.md](../../docs/CONTEXT_SLANG_TRANSLATION_SPEC.md) now explicitly define conversational semantic fidelity. The independent 90% adequacy gate requires >=90% **observed human-rated adequacy in each of the five categories**—everyday, idiom, slang, context and technical—in addition to the >=90% 95%-Wilson overall lower bound. A larger number of literal/medical successes cannot mask failures in slang or contextual meaning.

For `category="context"`, the frozen input JSONL must include `"previous_utterances": ["...", "..."]` (1-4 real preceding turns, each <=500 characters). Optional `"topic"` (<=100 characters) helps bilingual reviewers understand the conversation. These fields are included in the frozen case digest and reviewer packets; altering them invalidates the submission.

**Important:** Making context visible to human reviewers does **not** prove the model was given that context. The custom Marian offline inference is currently sentence-only, unlike the context-aware Gemini provider. Record whether context was actually provided and do not call the research model context-aware until train/inference context handling and paired ablation tests are implemented and passed. Preserve strict train/development/test separation by entire conversations, not individual turns alone. Neither AI self-judgment nor BLEU/chrF is a verified human-adequacy percentage.
