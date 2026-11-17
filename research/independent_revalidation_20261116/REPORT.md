# Independent frozen-checkpoint revalidation

2026-11-16 UTC. Scope: the ORIGINAL 128-update French and Spanish pilots in translation_training_20261116/NEURAL_TRAINING_REPORT.md. The continued experiment was read to identify and exclude its data reservations; its adapters were not evaluated or changed by this audit.

## Decision

Keep both original adapters as research artifacts and keep the live base translator unchanged. Historical scores and artifacts are consistent and checkpoint inference is reproducible. The new French holdout does not confirm the earlier in-domain gain. Neither the old mixed dialogue results nor this new in-domain sample establishes a conversation-quality improvement. Spanish remains unsupported for promotion by the combined evidence.

## New evaluation, frozen before inference

The registration and BOTH 256-row selections were saved before importing models or generating outputs. Seed: 2026111603. Each unchanged original base was compared with its original selected step-128 adapter. No fitting, hyperparameter change, checkpoint reselection or output-based exclusion occurred.

French uses 256 unused Tatoeba test components, one example per component, retaining original parent partitions. All original and continued train/dev/test components are excluded, along with exact source/reference text from the original and continued guards and all supplied French DiaBLa text. The selected sample also has zero exact source/reference overlap with the 16 prior synthetic probes; that extra check occurred after inference and changed nothing.

Spanish uses 256 unused WMT13 news rows. All 200 earlier baseline rows, 128 original guard rows and 256 continued guard rows are excluded, along with exact source/reference matches to either pilot's fitting and evaluation text. Within each new sample, source, reference and group identifiers are unique. Normalization: Unicode NFKC, casefold and whitespace collapse.

| Fresh human-reference evaluation | N | Base chrF | Original adapter chrF | Base BLEU | Original adapter BLEU |
|---|---:|---:|---:|---:|---:|
| French / fresh Tatoeba components | 256 | 78.035170 | 77.669802 | 66.285271 | 65.260953 |
| Spanish / fresh WMT13 news rows | 256 | 62.236740 | 62.268198 | 36.708911 | 36.802621 |

French adapter-minus-base: **-0.365368 chrF**, -1.024318 BLEU. Paired 2,000-resample percentile 95% interval for CORPUS chrF difference: [-1.712711, +1.033748]. This supports neither a reliable gain nor a reliable degradation. The earlier +2.134925 chrF / +4.580635 BLEU observation remains true of its original sample, but was not confirmed here.

Spanish adapter-minus-base: **+0.031458 chrF**, +0.093710 BLEU. Corresponding interval: [-0.570301, +0.612538].

Intervals are conditional descriptive bootstrap intervals, without multiple-comparison adjustment. Components do not prove independent authors/templates; Spanish rows do not prove independent news documents. Cached sufficient statistics reproduce the public SacreBLEU corpus-score calculation. These intervals concern corpus chrF; the ORIGINAL report's intervals concern mean sentence chrF, a different statistic. Neither BLEU nor chrF is percent correct translations.

Both arms use beam 4, batch 8, maximum 128 source tokens and 96 generated tokens, float32 CPU, one thread. French: 0 sources exceed the source cap (maximum 24 tokens). Spanish: 0 exceed it (maximum 81 tokens); those inputs are truncated identically in both arms. No sources were removed after outputs. No empty translations occurred. Timings are audit runtime, not controlled live latency.

## Historical evidence independently checked

All 33 original delivery-manifest entries and archived bytes match SHA-256. The archive has no unsafe paths or raw JSONL. All local pinned base files match their manifests. Selected adapters match step 128 byte-for-byte: 72 finite tensors, 294,912 parameters, 36 nonzero LoRA-B tensors each.

Original selected SHA-256:
- French: 7445651cc93617b3c8e4cd00c614371ee7af92ccf541659a9b445f8d9edf372f
- Spanish: 31c33afbfc7f3e5e1e58b361c099237578f214ffd6cce5dd9781c9fe0ff3dd0a

Original logs contain sequential steps 1-128 and finite loss/gradient records. Each saved selection agrees with the greatest dev chrF among base/64/128. Code freezes base parameters before adding LoRA, optimizes only adapters, writes selection before test generation, and supplies only source text to generation. These are code/log checks; full training was not independently rerun.

Both original train/dev/test partitions contain 1,024/128/256 rows, with zero normalized cross-split source, reference or block/component overlap. French samples and the 64-dialogue guard selection reproduce from the handoff; parent hashes, IDs, source text and references match. All 1,408 Spanish pairs match original Europarl lines. WMT guard indices/text match pinned source files and exclude earlier baseline indices.

| Recomputed historical evaluation | N | Base chrF | Adapter chrF | Base BLEU | Adapter BLEU |
|---|---:|---:|---:|---:|---:|
| Spanish / original Europarl | 256 | 66.327950 | 66.204124 | 43.959220 | 44.095750 |
| Spanish / original WMT13 | 128 | 59.744659 | 59.402919 | 36.479811 | 35.735649 |
| French / original Tatoeba | 256 | 74.244751 | 76.379676 | 61.874715 | 66.455349 |
| French / original DiaBLa | 128 | 54.928233 | 55.428430 | 31.174138 | 30.773612 |

All scores match saved reports within 1e-10. Newly loaded original checkpoints regenerated the first eight cases of each historical evaluation in both arms: **64/64 exact output matches** across two languages and four sets. These consumed cases count only as reproducibility checks.

A 256-line Europarl block cannot universally prevent adjacent rows crossing at block boundaries. Among the actual original samples, the nearest cross-split selected lines are 47 apart; document/speaker independence is not established.

## Limitations and handoff

- **No untouched local dialogue holdout remains:** all 144 DiaBLa dialogues are consumed or reserved: 40 smoke, 64 original, 40 continued. Extra turns from those dialogues were not relabeled independent.
- New selections and outputs are now **consumed**. Future tuning must not reuse them and claim fresh validation.
- Held out means held out from project fitting and earlier evaluated/reserved examples. OPUS pretraining exposure is unknown; authors/template independence is unproven. No new bilingual human review exists.
- This evaluates context-free written translation. No contextual memory, slang robustness, speech accents, ASR, live audio/UI pipeline or production latency was validated. Synthetic probes remain diagnostic.
- Exact checkpoint replay is demonstrated, not deterministic full-training reproduction. Optimizer/RNG continuation state is absent. The original bundle omits base weights, dependencies, raw/sampled data, output JSONL and some selection/handoff files. Scripts use sibling workspace paths. The included environment inventory is not a portable lock; neither archive is self-contained.
- Existing Tatoeba attribution gaps and dataset notices remain relevant to redistribution. No new publication/commercial clearance was assessed. Raw selection/output JSONL stays local and is excluded from the handoff archive.
- No app integration, activation, original-artifact overwrite, deployment, push, paid API call or training occurred in this audit. Quantized/integrated behavior would require separate validation.

registration.json records IDs, WMT indices, parent hashes and checkpoint hashes. results.json contains exact scores, metric signatures, intervals, replay and truncation counts, and output hashes. saved_audit.json and supplemental_checks.json contain integrity evidence. The scripts preserve the audit procedures.

Local-only files: fr_selection.jsonl, es_selection.jsonl, fr_outputs.jsonl, es_outputs.jsonl. Preserve them for replay/consumption tracking. Do not add raw sentences to the repository handoff without resolving existing attribution terms.

For replay, copy evaluate_frozen.py into a separate output directory in THIS workspace, then run it with PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Preserve the registered inputs and original checkpoints. Replays are reproducibility checks, not fresh validation.

No execution blocker remains. New independent dialogue data and bilingual human assessment remain missing before conversation-quality or deployment-readiness claims.
