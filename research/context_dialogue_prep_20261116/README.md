# Context-aware dialogue preparation, 2026-11-16

Prepared a separate, reproducible French→English dialogue-data pipeline and evaluation protocol. **No training or model evaluation ran.** No optional translator integration, model activation, paid service, push or deployment occurred. Existing training artifacts and original data are unchanged.

The local preparation produced:

| Partition | Examples | Conversation groups | Source |
|---|---:|---:|---|
| Train | 2,048 | 240 | Official WMT24 training dialogues |
| Development | 256 | 24 | Separate dialogues from the same official training file |
| Memory review queue | 64 unlabelled candidates | Train only | Earlier turn and other-dialogue comparisons |
| Fresh local DiaBLa test | **0** | **0** | All 144 dialogue groups already consumed or reserved |

The pinned WMT24 CSV contains 15,026 turns across 264 dialogues, including 5,093 French→English turns. The publisher README lists 15,027 total training turns; the audit uses the actual parsed snapshot count. Selection is hash-ranked with a fixed seed, capped at 16 examples per dialogue and independent of model outputs.

## Coordinate with training

The original neural report at `../translation_training_20261116/NEURAL_TRAINING_REPORT.md` and the newer `../translation_continued_20261116/REPORT.md` describe **context-free** translation training. This package does not compete with those runs. Their checkpoints remain research candidates; no dialogue-memory improvement is established by this preparation.

The newer continuation used 160 turns from the final 40 DiaBLa dialogues. Combined with 40 smoke and 64 original-guard dialogues, **all 144 are now accounted for**. The independent revalidation registration confirms this. Both translation directions share those same conversation groups. DiaBLa remains evaluation-only, usable as an explicitly consumed regression set; it cannot supply new training, tuning or untouched test evidence. See `artifacts/holdout_registry.json`.

For the next research experiment, use these new WMT train/dev inputs after separately registering the model recipe. Acquire and audit a new conversational test **before fitting**: official WMT24 valid/test are candidates, remain unacquired here, and need conversation-ID and normalized source/reference overlap checks against fitting data. Use only a genuine independent test for a new generalization claim. Do not tune against the old DiaBLa guards or synthetic semantic probes.

## Data sources and permitted scope

- **WMT24 Chat French–English — acquired.** Research-only under the publisher's CC BY-NC 4.0 terms; no commercial-use clearance. Original customer/agent channels, references and dialogue IDs make contextual translation preparation possible. Reference strings are copied from the source, never translated or invented here. See [publisher terms](https://wmt-chat-task.github.io/2024/#data), [source repository](https://github.com/WMT-Chat-task/chat-task-2024-data), and [Mohammed et al. (2024)](https://aclanthology.org/2024.wmt-1.59/). Revision `12a34149267dd9ae98e60e0ec2e415707fad5618`; raw CSV SHA-256 `eff96c25a2b5b5fc436ab85d573d4b2186fb8354a31763445c02e1cc80cc76ad`. README and complete license are retained locally. Do not ship these examples or derivatives as commercially cleared product data.
- **BConTrasT English–German — identified, not acquired.** Taskmaster-derived translated dialogues with published train/dev/test, CC BY-SA 4.0. A possible separately licensed contextual MT research source, but the language does not match the current French/Spanish pilots. Shared Taskmaster lineage means combining the two requires original-conversation overlap checks. See [publisher repository](https://github.com/Unbabel/BConTrasT).
- **Taskmaster-2 English — identified, not acquired.** CC BY 4.0; human role-play dialogue transcripts. Useful for temporal-coherence pretraining or human memory-review candidates, with original conversation IDs. It provides neither French/Spanish translation targets nor human helpful/neutral/harmful memory labels. See [publisher documentation](https://github.com/google-research-datasets/Taskmaster/blob/master/TM-2-2020/README.md).
- **Existing Tatoeba — available sentence translation data.** Preserve the prepared train/dev/test connected components and CC BY 2.0 snapshot attribution limitations. It has no conversation history. MAGPIE is auxiliary idiomaticity supervision, not dialogue translation or memory utility. Synthetic fixtures remain synthetic. Existing data is never relabelled to fill missing human supervision.

## Leakage and input policy

Train/dev conversation IDs are disjoint. All current sources, target references and history strings are checked against the existing Tatoeba dev/test and both DiaBLa directions. Training additionally excludes every source/reference string in all 24 reserved development dialogues, including rows outside the bounded dev sample. Normalization is NFKC, casefold and whitespace normalization; punctuation/paraphrase equivalence and unknown pretraining exposure remain unresolved.

History uses at most six **earlier original-source turns**, with original row indices/language/speaker. It never uses target references or future turns. Protected boilerplate is removed; gaps are retained rather than filled. This assumes access to both original language channels, unlike DiaBLa's original speaker-visible MT history. It is not a monolingual-listener simulation. Speaker identity is unavailable, so these splits are conversation-disjoint, not verified speaker-disjoint; clients, domains and templates can recur.

Inputs and published references are separate JSONL files joined by case ID. `prepare.request_payload` constructs requests through an allow-list, excluding labels, reference translations and extra fields. Training references may be joined for the target loss only. No request is sent by this package.

The 64 review candidates have null `human_judgement`, blank scene and blank independent reference. Earlier does not mean helpful; another dialogue does not mean harmful. These rows are **not eligible memory-training examples**. Fluent reviewers must independently supply grounded pre-current scene, meaning/reference and memory utility, with adjudication. WMT's published references can supervise translation; they cannot supply missing utility judgements.

## Evaluation plan

`artifacts/evaluation_plan.json` specifies no-context, all-history, selected-history and selected-history-plus-scene arms, with identical current sources, base checkpoint, decoding and output budget. Freeze checkpoint/tokenizer/selector hashes, thresholds, exact case membership, history/token budgets and acceptance criteria before fresh inference. Keep current utterances intact when allocating context tokens and record truncation, failures and actual selector overhead.

WMT has no independently grounded scene annotations in this package. The scene arm would equal selected-history until competent reviewers add scenes from information available before the current turn. The supervised selector also remains blocked on genuine utility labels. Start future exploratory comparisons on train/dev; do not manufacture those labels to enable the four-arm test.

Primary evidence should be blinded paired bilingual meaning-preservation review, including antecedents, formality/gender, ellipsis and lexical consistency. Report corpus chrF/BLEU as secondary summaries; resample whole conversations for uncertainty and name the aggregation. Mean sentence-chrF intervals do not describe corpus-chrF deltas. Report harmful changes, helpful rejection/harmful selection only where independent utility labels exist, abstention, median/p95 timing and context-token overhead. Two fluent reviewers should be blinded to model/arm, with disagreements adjudicated. Pre-register meaningful-gain and harm/latency limits with them; no numerical readiness claim is justified now.

Wrong scenes, misleading memories, language switches, negation, prompt-injection text and session resets are separate stress checks. Generated fixtures can test mechanics, not genuine translation accuracy. This package provides no speech/accent, low-resource-language, live application or production-readiness evidence.

## Reproduce and validate

Only the Python standard library is needed. From this directory, with the original handoff and training directories in the parent workspace:

```sh
PYTHONDONTWRITEBYTECODE=1 python fetch_sources.py
PYTHONDONTWRITEBYTECODE=1 python prepare.py --output artifacts
PYTHONDONTWRITEBYTECODE=1 python verify.py
```

`fetch_sources.py` downloads only the pinned training CSV, README and license, at most 8 MiB per file. Existing files must match the saved hashes. `prepare.py` refuses to overwrite an output directory. Use a new direct-child output path for a rebuild. If this source package is placed elsewhere, set `VERITASTESTER_DATA_WORKSPACE` to the workspace containing the original data and training artifacts. Do not rerun the original data-preparation script: it is unnecessary and writes original outputs.

`reports/validation.json` records **9 passed integrity checks**. `reports/reproducibility.json` records byte-for-byte artifact reproduction from two independent preparation runs. The manifests pin all source files; originals were rehashed before and after preparation. Integrity checks do not measure model quality.

## Repository delivery

`SOURCE_DELIVERY.tar.gz` is the ready-to-copy source/report bundle. It omits raw examples, derived train/dev text, review candidates, model outputs and all original private data. Its local source manifest lists every included file and hash. The package `.gitignore` also excludes those text directories. Parent delivery may commit this source package while leaving research corpus text local and preserving unrelated changes. No repository was modified here.
