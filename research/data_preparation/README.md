# BridgeCaption data-preparation handoff

This directory is separate from `veritastester`. No application files were changed, no paid service was used, no model was trained/activated, and nothing was pushed or published.

## Ready-to-use local research data

| Files under private/ | Training | Development | Test | Meaning |
|---|---:|---:|---:|---|
| `tatoeba.fr-en.*.jsonl` | 133,204 | 16,584 | 16,202 | Real community-authored French -> English sentence pairs |
| `magpie.*.jsonl` | 4,391 | 1,007 | 4,840 | Human crowdsourced idiomatic/literal labels; auxiliary task, not translation |
| `diabla.fr-en.*.test.jsonl` | 0 | 0 | 2,883 | Source-published human-reference dialogue translation evaluation |
| `diabla.en-fr.*.test.jsonl` | 0 | 0 | 2,865 | Source-published human-reference dialogue translation evaluation |
| `synthetic.context.*.jsonl` | 54 | 18 | 18 | AI-authored development fixtures, 15 expression families; not real evidence |

`private/tatoeba.fr-en.train.quick12000.jsonl` is a deterministic 12,000-row subset for small-compute experiments. It is NOT an additional dataset or a test set. It is NOT length-sorted or chosen using model scores.

Each DiaBLa direction has a 40-dialogue smoke-input subset. Use it for connectivity/latency only, and record that it has been examined. Never tune against DiaBLa or add it to a glossary. Full references are in separate files joined by case_id, to keep gold targets out of provider requests.

## First use

1. Read `reports/data_audit.json` and `reports/validation.json`.
2. Fit translation experiments only on Tatoeba train (or quick12000); choose configurations on dev. Test once after freezing a candidate. No translation runtime or weights are included here.
3. MAGPIE labels use **1=idiomatic, 0=literal**. Existing SemEval raw labels use the OPPOSITE convention. Merge semantic label names, never raw numeric values. Train/dev/test keep original type splits but remove documents and exact context sentences that cross split boundaries, prioritizing test then dev.
4. Evaluate genuine dialogue translation on DiaBLa only after freezing the model/configuration. Use identical source inputs and translation engine for no-context, all-history, selected-history and selected-history-plus-scene arms.
5. Do not feed `reference_translation`, label, authored meaning, model output, or future turns to the selector/translator as input. Do not report source-dataset MT judgments as context-memory usefulness labels.
6. Synthetic data is a separate experiment. Report its results separately. It does not satisfy the repository's human-review/live-verification gates and must not be relabelled as human data to pass them.

## Integrity and leakage precautions

- Tatoeba duplicate pairs removed: 159. All rows connected by an identical normalized French OR English sentence are assigned together, preventing alternate-translation leakage.
- Another 981 pairs were removed with their connected components because they overlap a DiaBLa original or reference. Final exact overlap is zero.
- Tatoeba group isolation does NOT imply author, topic or paraphrase independence; names/templates may recur. DiaBLa is independently sourced, but prior foundation-model training contamination is unknown.
- DiaBLa retains all 144 dialogues. Previous context contains only earlier turns visible to the original speaker. Opposite-language turns use the original source system's MT, NOT the newly created human gold reference. This reflects realistic noisy history.
- DiaBLa scene input contains only the shared scene and this speaker's assigned role, not the other participant's private role, feedback or future content. Browser/user-agent/demographic metadata is excluded from derived inputs.
- MAGPIE targets use source `context[2]` and the validated source character offsets. Simple dictionary-string matching is unsafe because legitimate idioms have inflection, insertion and passive forms. This preparation does NOT use the repository adapter's string-search target locator.
- All MAGPIE test rows are preserved; strict document isolation removes much of train/dev. Report the smaller, shifted sample and its class counts. Do not compare its score directly to published official-split results.
- Derived files are local-only with owner-only permissions. No user conversations, credentials or emails were used.

## Licenses, provenance and known source restrictions

### Tatoeba / ManyThings / TensorFlow mirror

Downloaded `https://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip` (3,423,204 bytes), SHA-256 `036e7dc4e83d393db24720ca0c7b7ddcb4b6b6053960438a5c0e1d6d8f2528f5`.

The archive's `_about.txt` dates the snapshot **2019-04-07**, says it derives from Tatoeba, provides a **CC BY 2.0** license notice and credits `www.manythings.org/anki` and `tatoeba.org`. That notice is preserved in `raw/tatoeba_mirror_about.txt`, and attribution is carried on every derived pair. The current ManyThings download returned HTTP 406; this older public training-data mirror was used instead. The two-column snapshot lacks individual contributor IDs, so we make no claim that redistribution or publication of trained artifacts has been fully cleared. Local research does not publish those sentences. Before redistribution, verify/restore attribution and retain source terms. Source translation quality is community quality, NOT a professional accuracy guarantee.

Sources: https://www.manythings.org/anki/ and https://tatoeba.org/en/terms_of_use .

### DiaBLa

Bawden, Bilinski, Lavergne and Rosset (2021), *DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues for Machine Translation*, Language Resources and Evaluation 55, 635–660. DOI: 10.1007/s10579-020-09514-4.

Source: https://github.com/rbawden/DiaBLa-dataset at commit `c9f3ae2db7b7c9248ec5dd04d51680c60aa4e7c5`. **CC BY-SA 4.0**. The original license and README are retained. Derived evaluation inputs/references retain attribution, note their filtering/field transformations, and remain subject to CC BY-SA 4.0. This is informal written French/English role-play, NOT accented speech, ASR, Yoruba, Swahili or workplace field validation. It has no helpful/neutral/harmful context utility labels. Both directions are from the same conversations, not independent samples of people.

### MAGPIE

Haagsma, Bos and Nissim (2020), *MAGPIE: A Large Corpus of Potentially Idiomatic Expressions*, LREC, 279–287. Source: https://github.com/hslh/magpie-corpus at commit `7fa677b82b9a772dfa54bbdd0fb414412d73db3b`. **CC BY 4.0** per publisher. BNC-derived; retain its license and attribution if redistributing derivatives. Snapshot and license are retained in raw/. Only its type-based split is used. Labels say idiomatic/literal, not exact meanings or translation quality.

### Excluded sources

- FLORES+ currently requires accepting a Hugging Face dataset gate. No gate was bypassed and no files were downloaded. Do not claim it is in this bundle.
- SlangInterp/Urban Dictionary research-only material was not acquired.
- Existing SemEval development results were previously inspected, so they are not described as an untouched evaluation here.
- No one invented human labels or gave blanket approval for downstream model redistribution.

## Reproduce and validate

With existing raw inputs and ordinary Python (no package installation):

    python scripts/prepare_data.py
    python scripts/validate_data.py

Preparation is deterministic. `reports/data_audit.json` contains SHA-256 and row count for every main output. `raw/download_manifest.json` records successful URLs plus failed attempts. Tests check source target alignment, label mapping, group/text/document isolation, no future/gold history, reference joins, synthetic separation and hashes. The eight checks passed; they are data-integrity checks, NOT measured translation accuracy.

## Remaining work

No independent bilingual human-review set of helpful/neutral/harmful memories or exact slang senses exists. The missing labels cannot be obtained by calling synthetic examples "real." Acquire consented dialogues or a separately permitted real conversational training corpus, then have competent reviewers annotate memory utility and reference translations without seeing the candidate model's answers. DiaBLa stays evaluation-only. Do not claim support or measured generalization for Yoruba/Swahili, speech accents or live latency based on this French/English text preparation.

## Native training compatibility

`python scripts/export_native_magpie.py` writes `private/magpie.normalized.{train,dev,test}.jsonl`. Those files use the app's `bridgecaption_corpus_v1` schema, semantic idiomatic/literal labels, string history, rights metadata and validated source offsets. The app's existing `bridgecaption.idiom_ablation` CLI accepted all 10,238 records in validation-only mode; no training gates were overridden. See `reports/native_adapter_validation.txt`.

A supplemental `private/memory_review_queue.csv` contains **128 unlabelled, real train-only context candidates** (64 source contexts, each paired with an adjacent and an other-document candidate). `situation` and `reference_translation` are deliberately empty, and there is no `human_judgement`. See `reports/review_queue.json`. An other-document candidate is NOT automatically harmful; this queue cannot train the memory model until a reviewer supplies the missing information. These English corpus passages are not multilingual workplace conversation data.
