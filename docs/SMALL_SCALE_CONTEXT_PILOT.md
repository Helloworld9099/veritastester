# SemEval small-scale context pilot — November 15, 2026

**Research status: negative/inconclusive. Do not claim the idea is proven or scale it based on this test.**

## What was actually trained

A CPU-only, 256-weight logistic classifier for **idiomatic vs. literal** usage on SemEval-2022 Task 2 Subtask A. Its input is the current sentence alone (baseline) versus the same sentence plus **the previous sentence** (context model).

This is **not** a trained multilingual translator, useful-memory predictor, two-grounded-view scene model, JEPA, or reinforcement-learning system. The experiment tests the small model/resource constraint and whether merely adding earlier context is beneficial.

**Source:** [SemEval-2022 Task 2 corpus](https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity), `SubTaskA/Data/dev.csv` and `dev_gold.csv`, matched on original ID; Git blob SHAs `0e7d1ef24b88f898cf11450b6592b9d030bf861c` and `f7c42edbb040e724bc8bc9454a36451174648569`. The official **development set is repartitioned** for an internal pilot: **not** an official SemEval benchmark score.

- 300 real labelled examples, six per each of 50 expression types; source label counts 135 idiomatic / 165 literal
- Expression-type-disjoint partitions: train 180 (30 types), validation 60 (10 types), untouched test 60 (10 types)
- Test cases: 30 English and 30 Portuguese, 29 idiomatic / 31 literal
- Current view: 128 binary hashed word unigram/bigram features
- Optional previous-utterance view: another 128 binary features
- Both classifiers: equal 256-coefficient logistic-regression architecture; baseline's context features are zeros
- Same four regularization candidates tuned on validation only, using macro-F1
- No future (next-utterance) context or reference text is provided as a feature

## Actual held-out results

| Method | Accuracy | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Majority label | 51.67% | 0.3407 | 0.5000 |
| Current sentence only | **56.67%** | **0.5486** | **0.5606** |
| Current + previous sentence | 53.33% | 0.5139 | 0.5273 |

**Context minus baseline macro-F1: -0.0347**, with group-bootstrap 95% interval **[-0.1383, +0.0876]**. This is not statistically decisive. Shuffling previous-sentence features scored 0.5588 macro-F1, another warning that the observed context features did not carry demonstrated useful signal.

An exploratory selective-context gate (lexical overlap >=0.15 and context model no less confident) was tuned on validation: **validation macro-F1 0.5996**, but **test macro-F1 0.5278** (delta -0.0208 vs current only; 95% group-bootstrap interval [-0.1060,+0.0557]). This validation/test discrepancy shows potential overfitting; **the gate did not prove the hypothesis either.**

## Efficiency

- 256 coefficients + intercept per model: about **1,028 bytes of float32 parameters** (excluding software/runtime)
- Four-hyperparameter training/tuning in this test environment: 0.114 seconds (current-only) and 0.024 seconds (context)
- Single case `predict()` after feature extraction: roughly 0.056 ms / 0.059 ms
- Whole Python/scikit-learn process peak RSS about 157 MB, *not* model-specific RAM

These are local observations, not guaranteed user-laptop latency numbers.

## Decision / next experiment

**Do not scale a model that simply concatenates all context.** Obtain **real, independently reviewed helpful/neutral/harmful memory labels** from permissioned multi-turn dialogues and fluent reference translations. Then train an abstaining memory-value predictor and compare (A) no context, (B) all recent context, (C) selective single-view memory, (D) selective two-view memory **with the same fixed translator** on a dialogue-group-disjoint test set. Include misleading/changed topic contexts and end-to-end time/cost.

This small proxy study does not establish whether selective memory or two-view representation learning will eventually help; it does establish that extra text is not automatically beneficial.

**Data/sample limitations:** Label-stratified selection inside each idiom type; small 60-case test across 10 expression types; 128-bit lexical hashing is deliberately resource-constrained and cannot model general semantics; only EN/PT; no natural speech, ASR, latency or real translation tested. Do not quote a conclusive improvement or degradation from this study.

**Execution:** Actual training and tests were run in an isolated container, and numerical results observed. A standalone reproduction script, aggregate JSON reports, gating and efficiency tests were generated as downloadable artifacts in the associated ChatGPT session. The Git repository contains this aggregate summary but **not the source dataset or deployment model weights**. No live app setting, main branch, or deployment was altered.
