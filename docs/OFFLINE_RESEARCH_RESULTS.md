# BridgeCaption offline research receipt — 2026-11-15

**STATUS: partial.** Dataset preparation and a real auxiliary model experiment
completed. Supervised memory/sense training and live translation validation
remain blocked by missing independent labels and workstation/provider checks.

Base: `Helloworld9099/veritastester`, main
`ff895c353ac8bb2eb1dfbb4e41f1a7ba7f86cfef`.
Working branch: `research/dataset-training-20261115`.
PR #3 was open at inspection; its idiom-review fix and regression tests are
included here without merging it. HTTPS Git authentication was unavailable;
62 repository blobs were retrieved through the connected account and verified
against Git blob hashes. The original Git tree and signed base commit were
reconstructed exactly before editing. No repository AGENTS.md was present.

## Completed, partial and blocked

| Work | Status | Observed result |
| --- | --- | --- |
| Code setup and original failure | Completed | Python 3.12.14 venv; installed documented test/research/eval groups. Original inflected-idiom fix retained. Added real MAGPIE `training` split alias and speaker/turn metadata preservation. |
| Source acquisition | Completed for four sources | MAGPIE, Multi3WOZ, SemEval and DiaBLa acquired. Pinned byte sizes, hashes, licences and revision URLs recorded. Conditional sources deferred/skipped explicitly. |
| Import and leakage screening | Completed | Original partitions retained. MAGPIE strict subset has zero cross-split idiom, document and exact-text conflicts. Lexical near-duplicate screen uses rare word 5-grams, Jaccard >=.9; this is not semantic deduplication. |
| CSVs and candidate JSONL | Completed | 1,600 memory candidates, 400 per EN/FR/AR/TR, spread across 120 aligned training dialogues; 1,000 English sense candidates across 710 BNC documents. All human labels remain null. |
| Auxiliary idiomaticity training | Completed | Two real CPU classifiers trained and scored on 2,301 held-out cases from 140 unseen idiom types. An independent repeat gave identical weights, predictions and metrics. |
| Memory-utility product model | Blocked | No independently reviewed helpful/neutral/harmful labels, grounded scenes or English references. Adjacency is not helpfulness. No fit performed. |
| Exact slang/idiom sense product model | Blocked | No independently reviewed contextual meaning labels. Idiomatic/literal labels were never substituted. No fit performed. |
| Optional adjacency-proxy pilot | Not run | Optional; priority went to the supervised auxiliary baseline and usable review exports. No claim of utility from dialogue identity. |
| Four-arm contextual translation | Blocked | No authorised available provider/model plus no qualified memory model. No chrF, translation improvement or speech/accent/code-switch accuracy is claimed. |
| Live application/overlay | Unverified | No real Ubuntu audio, microphone/monitor capture or workstation privacy/latency validation. One GUI test skips without display/Xvfb. No activation, merge, deployment or paid inference. |

## Measured auxiliary results

The target is **idiomatic versus literal**, not translation or exact meaning.
Same char 3–4-gram HashingVectorizer (32,768 dimensions), balanced SGD logistic
classifier, seed 2026, alpha .0001, maximum 30 iterations, threshold selected
on dev only. Single-view reads current utterance and expression; two-view adds
up to two preceding source sentences. Scene is empty. This does not test the
stronger hypothesis of an independently grounded second description.

| Model | Test macro-F1 | Balanced accuracy | Train seconds | Inference p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Train-majority baseline | 0.4521 | 0.5000 | — | — |
| Single-view | 0.6938 | 0.7216 | 0.653 | 0.3666 / 0.6903 |
| Two-view | 0.6878 | 0.7164 | 1.968 | 0.9382 / 1.3274 |

Two-view minus single-view macro-F1: **-.0060**. Paired bootstrap across held-out
idiom groups, 500 replicates, seed 2026: 95% interval **[-.0316, .0237]**.
This result does not establish a benefit from added context. No configuration
was changed based on test outcomes.

| Split | Rows | Independent idiom types | Idiomatic / literal |
| --- | ---: | ---: | ---: |
| Train | 5,404 | 840 | 4,350 / 1,054 |
| Dev | 1,064 | 85 | 915 / 149 |
| Test | 2,301 | 140 | 1,899 / 402 |

Confusion matrices (rows true literal/idiomatic, columns predicted literal/idiomatic):
single-view `[[241,161],[297,1602]]`; two-view `[[239,163],[307,1592]]`.
Per-class precision/recall/F1 and per-language metrics are in
`results/experiment_summary.json`. This experiment tests English only.

First model artifact: 1,311,525 bytes; weight JSON approximately 348 KB single
and 373 KB two-view. Repeat training process peak RSS 282,248 KiB; wall time
4.947 s; user/system CPU 7.914/0.714 s. Latencies above cover 200 sequential
single-example CPU predictions, excluding translation, ASR and audio capture.
Workstation CPU, cache and contention can change timings. Predictions/weights
are private JSON; no pickle loader or large language model download was used.

## Review files and limitations

- `data/private/memory_needs_review.csv`: 1,600 rows; scene/reference blank.
- `data/private/sense_needs_review.csv`: 1,000 rows; scene blank. The four
  source families with at least ten literal and ten idiomatic candidates are
  first, followed by broader coverage; these still lack verified meanings.
- `data/private/{memory,sense}_review_metadata.csv`: split, assessed rights,
  source hashes, warnings and missing fields. Sense groups use BNC documents;
  the source idiom-family ID is retained separately to prevent the impossible
  setup of holding every occurrence of an expression out of sense training.
- `data/private/{memory,sense}_candidates.jsonl`: inputs to interactive review.
- `data/private/{memory,sense}_queue.csv`: exact-header, **zero-row** ready queues.
  Incomplete candidates were not passed off as ready to annotate/train.
- `templates/{memory,sense}_queue.csv`: empty public-safe templates.

All candidate queues use official training rows only. No original test/dev
examples enter future product-model training queues. Multi3WOZ is culturally
adapted, so parallel turns were not presented as literal reference translations.
The original MAGPIE file has 48,395 rows; the importer excludes 21,876 rows whose
dictionary form cannot be uniquely grounded to a context sentence. This loss
limits generalization to inflected idioms. Another 17,750 imported rows are
removed by the stricter document/text screen. Source IDs/splits are never
reassigned to hide these exclusions.

## Executed checks

```bash
python -m pip install -e '.[test,research,eval]'
python -m compileall -q src tests scripts
python -m pytest -q
bridgecaption-preflight
python scripts/acquire_datasets.py
python scripts/acquire_datasets.py --download
python scripts/run_offline_research.py --prepare
python scripts/run_offline_research.py --train
python scripts/prepare_diabla.py
```

The first preparation attempt stopped on Multi3WOZ split-list IDs absent from
the released data; the adapter now intersects lists with actual dialogues and
records missing counts. A retry exposed repeated hashing inside the row loop;
that run was interrupted, hashing was moved outside the loop, and preparation
was completed. The model run used the corrected MAGPIE import and passed
offline preflight. Reproduction helpers refuse to overwrite prior reviewer or
experiment files. Use a fresh checkout/output tree to rerun preparation.

Final full suite: **138 passed, 1 skipped** (GUI display/Xvfb unavailable).
Offline compilation/preflight passed. Relevant existing tests cover literal
and idiomatic hints, unknown slang, abstention, group leakage, context-free
baseline isolation, empty context, bounded provider timeout settings and
provider error handling. They use fixtures/mocks and do not establish real
speech recognition, accents, code switching or provider uptime.

Exact receipts: `results/foundation_preflight.json`, `results/training_run.log`,
`results/replication_receipt.json`, `results/environment_versions.json`,
`results/*_import.json`, `results/magpie_cleaning.json`.

## Next action

Have independent fluent reviewers complete the private queues using the
commands in [REPRODUCE_OFFLINE_RESEARCH.md](REPRODUCE_OFFLINE_RESEARCH.md).
Reviewers must verify the observed expression and grounded scene and provide
true meanings/references; do not copy the binary source label as the answer.
Then validate sample-size/group/language coverage and held-out quality before
any live audio attestation or opt-in activation. Both product model settings
remain unset in this work. No human review or production readiness is claimed.
