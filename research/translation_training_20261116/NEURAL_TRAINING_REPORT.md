# Genuine neural translation training: isolated CPU pilots

## What is now completed
The former runtime blocker was resolved: installed an isolated official **PyTorch 2.7.1+cpu** runtime and retrieved the original trainable OPUS Marian checkpoints. The existing CTranslate2 model is an inference export, not trainable weights. No changes were made to it or to the shared BridgeCaption application.

**Actually trained two translation adapters**, each on 1,024 real parallel examples with 128 optimizer updates. This is neural machine-translation fine-tuning, unlike the earlier auxiliary idiom classifiers. It is not conversation-memory training, ASR training, a new foundation model, or proof of multilingual/slang accuracy.

- Spanish -> English: `Helsinki-NLP/opus-mt-es-en` revision `c96e2c5399ebfae4fc43d9669556b9afa74bb69d`.
- French -> English: `Helsinki-NLP/opus-mt-fr-en` revision `c4aed37b318c763fd177aa449b44e3b783cc6c02`.
- Both model cards state Apache-2.0. Weights and tokenizers are pinned and SHA-256 recorded under each `model/download_manifest.json`.
- All work ran on the existing cloud CPU. No paid service, GPU rental, API inference, source push, app setting change or deployment occurred.

## Method
For each model, freeze all original weights and add LoRA to all 36 attention query/value projections. Rank 8, alpha 16; **294,912 trainable parameters**, approximately 0.4% of the model. Each selected adapter is approximately 1.2 MB and requires its exact base checkpoint and architecture; it is not a standalone translator.

Fixed recipe: seed 20261116, CPU 4 threads, AdamW lr .0002, weight decay .01, microbatch 4, two-step gradient accumulation, gradient norm cap 1, max training length 96 tokens. Complete all 128 updates, saving adapters at steps 64 and 128. Compare only baseline/64/128 on the fixed 128-example development set, select greatest development chrF, freeze selection, and evaluate test sets once. Baseline remains eligible. Both pilots selected step 128. No settings were changed after viewing test results.

Decoding is identical for base and selected candidate: beam size 4, batches of 8, max source length 128 tokens, max 96 generated tokens. Evaluation uses **actual generated translations** and published human reference text. Scores are corpus chrF and BLEU, not percentages of sentences translated correctly.

The original zero-output adapters reproduce base logits exactly. Selected safetensors checkpoints reproduce logits after reload. A separate fresh-process French replay translates `Je ne veux pas annuler la réservation.` as `I don't want to cancel the reservation.` This is a synthetic smoke test, not accuracy evidence.

## Data and separation

### Spanish
Downloaded the official Europarl v7 Spanish-English aligned corpus from https://www.statmt.org/europarl/v7/es-en.tgz. Exact SHA-256 and source provenance are in `data_manifest.json`. Research use is supported by the corpus page; acknowledge **Philipp Koehn (2005), Europarl: A Parallel Corpus for Statistical Machine Translation**, and the European Parliament. Do not treat this as blanket commercial redistribution clearance.

Filter to bounded sentence lengths, remove repeated normalized Spanish source text, group contiguous 256-line blocks before selecting 1,024 train, 128 dev and 256 test examples. The sampled partitions also have zero exact normalized English-reference overlap. Block grouping prevents immediate neighboring rows crossing splits; it does **not** guarantee independent speakers or original documents.

Additional guard: 128 WMT13 Spanish-English rows, excluding all 200 indices used in the independent-validation thread and excluding exact current experiment sources. Indices and original file hashes were frozen before training. Neither this WMT sample nor Europarl can be promised unknown to OPUS pretraining; these are held out from **our adapter fitting**, not necessarily the foundation model.

### French
Used the data thread's French-English Tatoeba splits without moving any example between original partitions. Fixed samples: 1,024 train, 128 dev, 256 test. Parent split construction keeps connected duplicate/alternate-translation components together and removes overlaps with DiaBLa. Samples have zero exact normalized source, reference or component overlap.

External dialogue guard: 128 French-English DiaBLa turns, two randomly selected turns from each of 64 dialogues. Excluded every dialogue in the data thread's 40-dialogue smoke subset. **DiaBLa was never trained on**, never added to a glossary, and references/history/future text were never used as model inputs. This context-free model sees only each current French utterance. Initial unseen-to-project status does not guarantee unknown pretraining.

Tatoeba snapshot attribution is CC BY 2.0, Tatoeba contributors and ManyThings/Anki via the TensorFlow mirror. The old mirror lacks contributor IDs; preserve the data thread's redistribution caveats. DiaBLa is CC BY-SA 4.0: Bawden, Bilinski, Lavergne and Rosset (2021), DOI 10.1007/s10579-020-09514-4. Raw sentences are not included in this delivery bundle. Local files remain separate with attribution.

## Frozen-model test results

| Model / evaluation | N | Baseline chrF | Adapter chrF | Baseline BLEU | Adapter BLEU |
|---|---:|---:|---:|---:|---:|
| Spanish / Europarl test | 256 | 66.328 | 66.204 | 43.959 | 44.096 |
| Spanish / WMT13 guard | 128 | 59.745 | 59.403 | 36.480 | 35.736 |
| French / Tatoeba test | 256 | 74.245 | **76.380** | 61.875 | **66.455** |
| French / DiaBLa guard | 128 | 54.928 | 55.428 | 31.174 | 30.774 |

**Spanish: do not promote.** The slight development improvement did not carry through to either test set. Its adapter is retained for reproducibility, not activated.

**French: improved the in-domain test; external dialogue results remain mixed.** Tatoeba improved +2.135 corpus chrF and +4.581 BLEU. DiaBLa improved +.500 corpus chrF but lost .401 BLEU. A dialogue-group bootstrap of mean sentence-chrF differences gives [-3.023,+1.157]; it does not establish a reliable dialogue benefit. The mean sentence difference is -.944 despite positive corpus chrF: length/aggregation matters. Do not describe it as a general conversation-quality gain.

A component-group bootstrap of mean sentence-chrF differences on Tatoeba gives [.578,3.058] with mean +1.784. This supports an in-domain gain on the sampled split, with pretraining, authorship and paraphrase limitations. It is a different aggregation from corpus chrF. Complete uncertainty reports, evaluation outputs and unchanged selected checkpoints are saved. No threshold/hyperparameter search was performed against either external guard.

## Measured compute

| Pilot | Updates | Training interval including intermediate dev checks | Entire run including evaluation | Peak process RSS |
|---|---:|---:|---:|---:|
| Spanish | 128 | 142.5 seconds | 415.9 seconds | 1,262 MiB |
| French | 128 | 87.2 seconds | 239.0 seconds | 986 MiB |

These pilots used about 1 GB RAM per process, 4 CPU threads and no accelerator. Training is practical at this scale. Baseline/candidate batch-amortized inference ranged about .12–.38 seconds/sentence, with both jobs sharing the machine. This is not single-request latency or a speed improvement claim; French adapter timings being lower in one batch reflect uncontrolled contention/cache/order, not proof it is faster. The earlier int8 CTranslate2 baseline is a different runtime and cannot be directly compared as a training gain.

## Verification and artifacts
`verification.json` checks pinned base-file hashes, exact source/reference/group isolation, all 128 updates, finite nonzero selected adapter weights, initial-backbone/reload equality, output counts and exact metric reproduction. It also checks that the WMT guard excludes old indices and the DiaBLa guard excludes smoke dialogues. These are research-integrity checks, not app, audio, live UI or deployment tests.

Directories:
- `/workspace/scratch/2b30092347a7/translation_training_20261116`: Spanish pilot, isolated PyTorch deps, preparation/training/replay scripts, protocol, checkpoints, test outputs, report and uncertainty.
- `/workspace/scratch/2b30092347a7/translation_fr_training_20261116`: French pilot, matching source files, adapter checkpoints, protocol, frozen evaluation and fresh-process replay.

Minimal candidate replay, from either directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python run_candidate.py --text 'language-appropriate public example'
```

Add `--baseline` to disable LoRA and use the unchanged original weights. The helper stays offline and requires its pinned base-model files. The Spanish directory's `deps/` contains PyTorch; both use the independent-validation dependencies by absolute path. Full rebuild commands are the scripts `prepare_europarl.py` / `prepare_from_handoff.py`, followed by `train_adapter.py`. Do not overwrite previous experiment outputs without preserving them. Source hashes and chosen indices support replay.

## Recommendation to parent
Keep **all live settings unchanged**, and keep Portuguese hints inactive. Preserve French as a promising **research checkpoint**, not a verified app upgrade. A larger pre-registered dialogue test and fluent human review are needed before choosing it over the base for real conversations. Any next training run should use additional legally sourced conversational training data, a new dev set for that domain and a new untouched dialogue holdout. Do not tune against these now-consumed guard sets or use synthetic labels to satisfy human-review requirements.

For runtime integration, merge a validated adapter into a copy of its base and convert that copy to CTranslate2 int8, then re-evaluate the quantized model and pipeline invariants; do not assume float32 results transfer exactly. No merged app artifact, provider default or deployment was made in this task.
