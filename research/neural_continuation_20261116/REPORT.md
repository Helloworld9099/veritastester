# Continued neural AI training — 2026-11-16

## Completed scope
Continued actual model-weight training, as requested, without translator integration or activation. Completed two bounded local CPU experiments: a French continuation and a Spanish lower-rate restart. No paid APIs, rented compute, push or deployment; no shared application files or unrelated work changed. All original checkpoints/base-model files retain their recorded hashes.

This is **context-free neural translation research**, not a trained context-utility selector, exact slang-sense model, speech recognizer, or proof of general multilingual quality. Portuguese auxiliary hints remain inactive. The parent's 16 synthetic semantic probes were not opened, optimized against, or used for selection.

## Exact training progress

| Experiment | Initialization | New training examples | Updates completed | Research checkpoint selected |
|---|---|---:|---:|---|
| French -> English | Previous 128-update French adapter | 4,096 | 256/256 | Additional step 128 |
| Spanish -> English | Original base + zero-output adapters | 2,048 | 128/128 | New step 128 |

French's optimizer was restarted; only previous adapter parameters were loaded. This is not a resumed optimizer-state training run. Its selected continuation checkpoint has 128 prior updates plus 128 new updates, and has seen the old 1,024 + new 2,048 = **3,072 distinct fine-tuning pairs**. The later step-256 checkpoint is also preserved; that full trajectory saw all 5,120 pairs but scored lower on development text. Do not confuse the fully trained last checkpoint with the development-selected one.

Spanish's selected checkpoint is a fresh restart on 2,048 pairs; it does not inherit the prior regressed adapter. The prior adapter remains a frozen comparator.

Both use rank-8, alpha-16 LoRA on the same 36 attention query/value projections, **294,912 trainable parameters**, frozen original Marian weights, microbatch 4, accumulation 4 (effective batch 16), AdamW weight decay .01, gradient-norm cap 1, max training length 96, CPU four threads. French LR=.0001; Spanish LR=.00005. Maximum pure training allowance was 600 seconds per job; both completed within it. No synthetic examples or generated translations were training targets.

## Selection registered before scoring test outputs

- New seed: 202611162. Prepared file hashes and source/previous-checkpoint hashes are preserved in each protocol.
- French: compare original base, previous adapter, continued steps 128/256 on **256 new dev examples**. Select greatest dev corpus chrF, tie favoring earliest candidate. Base remained eligible. Scores: base 73.8796, previous 73.9859, new128 **74.2216**, new256 74.0394.
- Spanish: compare base and new64/new128 on **128 new dev examples**. Old adapter is a diagnostic comparator, explicitly ineligible for the selected new recipe. Scores: base 64.3342, new64 64.3570, new128 **64.3653**; old comparator 64.3697.
- Selected weights were saved and selection frozen before test generation. Test outcomes did not change the chosen checkpoint or recipe.
- The held-out cohorts are now consumed. They must not be reused as untouched evidence in later work.

## New data and leakage controls

### French
Retained the data thread's human Tatoeba train/dev/test component partitions, excluding **all components used in the prior pilot**. New samples: 4,096 train, 256 dev, 512 test. Zero exact normalized source/reference overlap between these samples or with the old fitting/dev/test examples.

External guard: **160 DiaBLa turns from the remaining 40 dialogue groups**, four turns/group. Excluded all 40 earlier smoke dialogues and all 64 prior guard dialogues. No exact current-source overlap with the earlier external guard/smoke. One new guard has the same short reference as a prior guard; original cohort results remain visible, and a **159-turn sensitivity cohort** excludes it. This additional rule was recorded before any new external output file existed; no model-selection change. DiaBLa has never been training data or glossary material, and only each current source utterance is supplied to the translator.

### Spanish
Reused the original Europarl v7 contiguous-block partition while excluding **all 1,277 previously sampled blocks**. New samples: 2,048 train, 128 dev, 256 test. Zero exact normalized source/reference overlap between the samples or with the prior fitting/dev/test examples.

External guard: **256 new WMT13 rows**, excluding all previously evaluated 200 + 128 = 328 row indices and their exact source text. Current training/dev/test source overlaps are also excluded. Zero exact source/reference overlap with the preceding WMT guard.

### Limits that remain
Tatoeba components and Europarl blocks are not verified independent authors or speakers. OPUS pretraining may already have included these public corpora, including evaluation material: these are held out from **our adaptation**, not guaranteed unknown to pretraining. All 144 DiaBLa dialogue identities are now accounted for across prior smoke/guard and this new guard; another genuinely untouched dialogue test needs a different source or new reviewed data.

Attribution and local-research restrictions from the data handoff remain in force: Tatoeba CC BY 2.0 / ManyThings / TensorFlow snapshot, DiaBLa CC BY-SA 4.0 / Bawden et al. (2021), and Europarl / Koehn (2005) / European Parliament. Old Tatoeba contributor-ID limitations are not resolved by these experiments. Raw dataset text and model outputs remain local and are omitted from the delivery bundle.

## Frozen test results
Scores are corpus chrF and BLEU, **not percent-correct translation accuracy**. All three arms use identical beam-4 decoding, batches of eight, max source 128 / max output 96 tokens.

| Language / cohort | N | Base chrF | Previous chrF | Selected chrF | Base BLEU | Previous BLEU | Selected BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|
| French / new Tatoeba test | 512 | 74.9047 | 75.0533 | **75.5663** | 62.8125 | 62.8831 | **64.0733** |
| French / new DiaBLa guard | 160 | 57.5809 | 57.4397 | **57.8863** | 37.6328 | 37.7160 | **38.2507** |
| Spanish / new Europarl test | 256 | 66.6940 | 66.6390 | **66.7962** | 43.9981 | 44.0708 | 43.9002 |
| Spanish / new WMT guard | 256 | 58.7914 | 58.9815 | **59.1253** | 32.1493 | 31.9807 | **32.5242** |

French improves both point metrics over previous/base on both new cohorts. The gain is modest and uncertain: selected-minus-previous corpus-chrF group-bootstrap 95% intervals are **[-.203,+1.204]** for the source test and **[-.227,+1.145]** for DiaBLa. The 159-case strict guard gives selected-minus-previous +.4649 chrF / +.5802 BLEU, with chrF interval **[-.208,+1.182]**. Removing the repeated reference does not change the conclusion.

Spanish improves external point metrics over both controls; the source-test BLEU is slightly lower. New WMT selected-minus-base is +.3339 chrF / +.3749 BLEU, interval **[-.033,+.711]**. Selected-minus-previous WMT is +.1439 chrF / +.5435 BLEU, interval **[-.248,+.578]**. All intervals include zero. There is no reliable evidence that the earlier Spanish regression was permanently solved.

Spanish's lower rate, data sample, effective batch and initialization seed changed from the first pilot. This is a corrective recipe experiment, **not a learning-rate-only causal ablation**. The old adapter also behaves differently across new cohorts, supporting caution about small-sample/domain variation rather than a firm diagnosis of catastrophic forgetting. Do not report that LR caused or definitively fixed the earlier error.

The paired intervals use 1,000 group-resampled corpus-chrF comparisons and reproduce the public sacrebleu scores exactly from cached sufficient statistics. Groups: French source 502 components; external 40 dialogues; Spanish source 201 blocks; WMT 256 rows. Intervals are descriptive, conditional on the available groups, without multiple-comparison adjustment.

## Integrity and measured limits

All **35 research-integrity checks passed**: completed update counts, registered dev-only selection, checkpoint replay, preserved old checkpoints/base files, finite nonzero weights, input/output alignment, freeze-before-output ordering, and exact chrF/BLEU reproduction. These do not substitute for application tests, live audio evaluation or independent fluent review.

`length_audit.json`: no selected train/dev/test/guard source exceeds the applied training/inference input limits; no reference exceeds the output-length cap. Zero empty generated outputs. This does not prove semantic correctness.

| Pilot | Pure training compute | Entire run, including controls/evaluation | Peak RSS |
|---|---:|---:|---:|
| French | 173.9 s | 569.1 s | 989.7 MiB |
| Spanish | 177.1 s | 789.9 s | 1,077.5 MiB |

Concurrent CPU evaluation and dataset lengths dominate elapsed time. Batch-amortized timings are descriptive; they are not live single-request latency or evidence that training improved speed.

## Deliverables and reproducibility

Root: `/workspace/scratch/2b30092347a7/translation_continued_20261116`.

- `prepare.py`, `train.py`: exact isolated preparation and training recipes.
- `french/` and `spanish/`: immutable data manifests/protocols, dev histories, training logs, selected/last/intermediate safetensors, frozen evaluations, full local outputs and paired analysis.
- `input_audit.json`, `length_audit.json`, `verification.json`: leakage, length and reproducibility checks.
- `french/strict_guard_addendum.json`: pre-output sensitivity registration.
- `analyze.py`, `audit_inputs.py`, `verify.py`: evidence inspection/replay tools.
- Each selected adapter is 1,187,800 bytes and requires its exact pinned base weights. These are not complete apps or standalone translators.

To reproduce, preserve current files first and run `prepare.py`, then `train.py --language french` and `--language spanish` using the existing isolated CPU dependencies, followed by `analyze.py`, `audit_inputs.py`, and `verify.py`. Paths to pinned original weights, previous adapter and data handoff are explicit. Fresh optimizer warm starts are documented. Refit determinism across different hardware/libraries is not promised; saved-checkpoint/logit replay and metrics are exact in this environment.

## Handoff recommendation
Preserve both new candidates and every original checkpoint. French's selected continuation is the stronger next **research** starting point; the new Spanish checkpoint is a plausible lower-rate candidate, with insufficient evidence for replacement. Nothing is activated or integrated.

Future training can continue on legitimate train/dev data, but stronger generalization claims require additional licensed conversational training data, a distinct conversation-domain dev set, and a newly reserved independent conversation test. The existing DiaBLa guards and parent-authored synthetic probes are consumed; never train on them, tune to them, or call later rescoring untouched validation. No further account, payment or deployment step is required for this completed round.
