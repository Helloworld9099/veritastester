# Corpus preparation for BridgeCaption — eight data sources

**No downloading, model training, cloud inference, or uploads occur in these preparation commands.** You supply dataset files that you acquired under each source's terms. Records and annotations are written only to local, gitignored `data/private/` or `results/`, with owner-only file permissions on POSIX systems. `--write` is explicit, and existing files are never overwritten.

The attached RL hypothesis proposes two complementary views, a predictor of useful previous memories, and avoiding negative transfer. In language translation this remains a **testable analogy**, not a demonstrated JEPA or reinforcement-learning method. Use an honest staged pipeline:

1. **Two-view auxiliary idiomaticity:** distinguish a current utterance alone from the same utterance plus surrounding context. MAGPIE and SemEval provide actual idiomatic/literal labels for this auxiliary task.
2. **Predictive memory:** extract previous turns from real dialogues, pair some with other conversations as candidate distractors, then require **human judgments** of helpful/neutral/harmful, independently verified English reference translations, and a grounded second scene description. A different dialogue is NOT automatically a harmful memory.
3. **Slang senses:** use candidate dictionary definitions and idiomaticity labels to propose review cases. A human supplies the **actual meaning in the situation**; idiomatic/literal labels and tweet sentiment cannot masquerade as meaning annotations.
4. **Held-out evaluation:** DiaBLa is test-only. Compare the same translator with no context, always-on context, selective single-view memory, and selective two-view memory; report harmful memory selections and end-to-end latency.
5. **Optional self-supervised pretext:** train a future continuity/contrastive encoder on genuine same-dialogue-adjacent versus other-dialogue turns. These are valid **adjacency proxy labels**, not proof of semantic equivalence or useful transfer. Use this only if the simple supervised baselines justify the extra computation.

## Install and initialize

From a checkout containing `research/dataset-training-20261115` (see [the execution receipt](OFFLINE_RESEARCH_RESULTS.md)):

~~~bash
python -m pip install -e '.[test,research]'
python -m pytest -q
bridgecaption-preflight
mkdir -p data/private ~/bridgecaption-datasets
~~~

Preparation has **dry-run mode by default**. Every successful import reports source roles, train/dev/test counts, source rights metadata, cross-split group conflicts and warnings about duplicate long utterances. Each normalized row contains its original-source SHA-256 and filename (not a machine-specific absolute path).

### Dataset mapping

| Corpus | Local input | Output role | Safe use / hard limit |
| --- | --- | --- | --- |
| MAGPIE | Official filtered **type-based split** JSONL | Binary idiomaticity | Labels i/l mean idiomatic/literal, not the exact idiom meaning; do not use filtered random split for zero-shot idiom claims |
| SemEval-2022 Task 2 | Subtask A CSV with DataID, Language, MWE, Previous, Target, Label | Binary idiomaticity | 0=idiomatic, 1=non-idiomatic; preserve provided train/dev/test files and zero-shot/one-shot distinction |
| SlangInterp | Data/UD_train.csv, UD_dev.csv, UD_test.csv | **Unreviewed meaning candidates** | Columns Slang, Definition, Usage Context. Dataset is copyrighted Urban Dictionary material, expressly restricted to research use; do not redistribute records or artifacts without authorization |
| Multi3WOZ | Extracted JSON dialogues with dialogue_id and turns/log entries | Conversation context candidates | Languages en/ar/fr/tr. Align same dialogue ID across languages; cultural adaptation is not a literal translation reference. May need a small field map depending on extracted version |
| Bangor Miami | A **normalized CHAT/CLAN export** (TSV/CSV) with conversation_id, speaker, language, utterance | Code-switching dialogue context | Do not feed raw .cha CHAT markup into the generic text loader; preserve speaker/conversation IDs and comply with TalkBank participant/ethics restrictions |
| DiaBLa | HF plain_text records exported to JSONL **with a verified dialogue_id** | **Test only** EN/FR | Human reference translation exists; never train or tune thresholds on DiaBLa; do not treat utterance id as dialogue id |
| NaijaSenti | CSV/JSONL exported from a language's dataset split | Informal-language *review only* | Tweet sentiment is not a slang-definition or translation target. Respect platform, copyright, privacy and language-specific restrictions |
| Okwu corpus | Local JSONL/JSON with source_lang, source_text, english_translation | Parallel-text translation only | No prior dialogue or memory labels; review source quality even when a row claims verified=true |

Authoritative source pages:
- MAGPIE: https://github.com/hslh/magpie-corpus (CC BY 4.0 corpus; source based on BNC)
- SemEval: https://sites.google.com/view/semeval2022task2-idiomaticity
- SlangInterp: https://github.com/zhewei-sun/slanginterp (Data/README.txt contains research-only restriction)
- Multi3WOZ: https://github.com/cambridgeltl/multi3woz
- Bangor Miami: https://talkbank.org/biling/access/Bangor/Miami.html
- DiaBLa: https://huggingface.co/datasets/rbawden/DiaBLa (CC BY-SA 4.0)
- NaijaSenti: https://github.com/hausanlp/NaijaSenti
- Okwu: https://github.com/Okwu-AI/corpus (CDLA-Permissive-2.0 data; check the specific release)

**This code does not assert legal rights**: review and comply with each dataset's current terms, source attribution, platform consent and derived-model restrictions before training or publishing outputs. Private storage alone does not grant rights.

## Example imports (adjust filenames to the actual downloaded releases)

**MAGPIE** — prefer type-based holdout:

~~~bash
bridgecaption-prep-corpus --source magpie --language en \
  --input ~/bridgecaption-datasets/magpie-corpus/MAGPIE_filtered_split_typebased.jsonl \
  --limit 200000 --skip-invalid --rights cc-by-4.0-reviewed \
  --out data/private/magpie.jsonl --write
~~~

Review the reported rejected rows. Invalid target alignment must be quarantined, not guessed. The original split labels are retained.

**SemEval** — prepare each official split separately (example with train only):

~~~bash
bridgecaption-prep-corpus --source semeval \
  --input ~/bridgecaption-datasets/semeval/train_zero_shot.csv \
  --source-split train --rights verify-semeval-terms \
  --out data/private/semeval-train.jsonl --write
~~~

**SlangInterp** — the real CSV headers are recognized without mapping. Source filenames like UD_train/UD_dev/UD_test preserve their official splits:

~~~bash
bridgecaption-prep-corpus --source slanginterp --language en \
  --input ~/bridgecaption-datasets/slanginterp/Data/UD_train.csv \
  --skip-invalid --rights research-only-acknowledged \
  --out data/private/slanginterp-train.jsonl --write
~~~

Only use this corpus for permitted research; do **not** publish raw entries or derived proprietary text.

**Multi3WOZ** — unzip the source's 90MB `data.zip` outside the repo; locate actual language/split JSON files:

~~~bash
cd ~/bridgecaption-datasets/multi3woz
unzip -n data.zip
find . -name '*.json' | head
cd /path/to/veritastester
bridgecaption-prep-corpus --source multi3woz --language en \
  --input ~/bridgecaption-datasets/multi3woz/path/to/en/train.json \
  --source-split train --rights multi3woz-reviewed \
  --out data/private/multi3woz-en-train.jsonl --write
~~~

The generic adapter recognizes **dialogue_id + turns/log/dialogue/messages** and **utterance/text/content**. Actual extracted files can use different wrapping. If the importer rejects the schema, inspect the JSON and supply an explicit `--columns` mapping or convert it to a normalized dialogue JSONL first. Never silently flatten the whole ZIP or lose dialogue IDs.

**Bangor Miami** — after using TalkBank-compatible tools to export CHAT utterance tiers into a tab-separated file with header `conversation_id,speaker,language,utterance` (TSV fields separated by tabs):

~~~bash
bridgecaption-prep-corpus --source bangor \
  --input ~/bridgecaption-datasets/miami/clean_dialogues.tsv \
  --rights talkbank-ethics-reviewed \
  --out data/private/bangor-dialogues.jsonl --write
~~~

The tool **does not claim to parse raw .cha files directly**.

**DiaBLa** — export the Hugging Face dataset locally with its `id,orig,ref,utterance_meta` fields; add a **verified real conversation ID from source metadata** as `dialogue_id` before importing:

~~~bash
bridgecaption-prep-corpus --source diabla \
  --input ~/bridgecaption-datasets/diabla/verified_grouped_test.jsonl \
  --rights cc-by-sa-4.0-reviewed \
  --out data/private/diabla-heldout.jsonl --write
~~~

Every DiaBLa record remains test-only regardless of command options.

**NaijaSenti** — export an official Nigerian Pidgin or Yoruba split to local CSV, with one row per social-media post:

~~~bash
bridgecaption-prep-corpus --source naijasenti --language pcm \
  --input ~/bridgecaption-datasets/naijasenti/pidgin_train.csv \
  --source-split train --columns '{"utterance":"text","sentiment":"label"}' \
  --rights platform-and-dataset-terms-reviewed \
  --out data/private/naijasenti-pcm.jsonl --write
~~~

**Okwu** — export the dataset locally from Hugging Face to JSONL. The adapter reads the published fields `record_id,source_lang,source_text,english_translation,verified`:

~~~bash
bridgecaption-prep-corpus --source okwu \
  --input ~/bridgecaption-datasets/okwu/records.jsonl \
  --rights cdla-permissive-2.0-reviewed \
  --out data/private/okwu.jsonl --write
~~~

A published `verified` flag is preserved as a claim, not proof that every translation is correct; samples should be independently checked.

## Multi-corpus audit, contrastive candidates and human review

~~~bash
bridgecaption-corpus-curriculum --task audit --inputs \
  data/private/magpie.jsonl data/private/slanginterp-train.jsonl \
  data/private/multi3woz-en-train.jsonl

# Two alternative views of conversation continuity, NOT useful-memory labels
bridgecaption-corpus-curriculum --task alignment \
  --inputs data/private/multi3woz-en-train.jsonl \
  --out data/private/temporal_pairs.jsonl --write

# Existing/concurrent dialogue memory vs a same-language distractor
bridgecaption-corpus-curriculum --task memory \
  --inputs data/private/multi3woz-en-train.jsonl data/private/bangor-dialogues.jsonl \
  --out data/private/memory_candidates.jsonl --write

# Source slang/idiom examples with NO fabricated meaning labels
bridgecaption-corpus-curriculum --task sense \
  --inputs data/private/slanginterp-train.jsonl data/private/magpie.jsonl \
  --out data/private/sense_candidates.jsonl --write

# Human reviewer fills a grounded scene, reference meaning/translation,
# and helpful/neutral/harmful label. q exits safely; --append resumes.
bridgecaption-corpus-review --task memory \
  --candidates data/private/memory_candidates.jsonl \
  --out data/private/reviewed_memory.jsonl

bridgecaption-corpus-review --task sense \
  --candidates data/private/sense_candidates.jsonl \
  --out data/private/reviewed_senses.jsonl
~~~

You can include curated slang expressions in a separate CSV with `language,expression` columns via `--terms` for NaijaSenti and other informal corpora. This identifies **review candidates**, not their meanings.

The review tools deliberately refuse test/holdout records. They never generate a "grounded situation" from a target/reference label: the reviewer must provide one based only on permitted scene evidence.

## Optional supervised auxiliary experiment (explicit opt-in)

~~~bash
# Schema/split verification only, no model fit:
bridgecaption-train-idiom --inputs \
  data/private/magpie.jsonl data/private/semeval-train.jsonl \
  data/private/semeval-dev.jsonl data/private/semeval-test.jsonl

# Actual training only when all permissions and foundation gates pass:
bridgecaption-train-idiom --inputs \
  data/private/magpie.jsonl data/private/semeval-train.jsonl \
  data/private/semeval-dev.jsonl data/private/semeval-test.jsonl \
  --train --confirm-dataset-rights
~~~

This auxiliary classifier trains **two equal-budget linear models**, one reading the utterance only and the other reading preceding context, and compares held-out *idiomatic versus literal* classification macro-F1. It uses CPU-friendly hashed character n-grams and an optional scikit-learn install. It **does not fine-tune Whisper or the translation LLM, and does not output the exact meaning of slang**.

The model training command validates source-provided train/dev/test grouping, licenses confirmed by the operator, a successful offline preflight and minimum data. It writes a JSON-only artifact under `results/`, not an unsafe pickle file.

## Core research checks

- Four arms A (no context), B (all context), C (selected single-view), D (selected two-view) on identical held-out spoken utterances and the **same translation engine**; always-on context is a strong practical baseline.
- Negative transfer cases must be judged independently. Words appearing similar or coming from different conversations alone do **not** establish whether transferring them helps.
- Compare one vs two grounded descriptions, genuine different perspectives vs mere paraphrases, and misleading scene descriptions; do not feed reference translations into predictor features.
- Preserve original corpus splits, aligned multi-language dialogue IDs and independent speaker/conversation families. Reject group leakage, review exact-text duplicates and MAGPIE cross-document overlap warnings.
- Report per-language macro-F1 for idiomaticity, sense accuracy/abstention, helpful/harmful memory selection, translation chrF and human meaning preservation, median/p95 latency and missed audio chunks, plus training/CPU cost.
- **A passing offline test is not evidence of translation performance.** A negative result should be reported, not hidden behind new claims of superior AI.

No source corpus is bundled in Git and **no real training begins automatically**.
