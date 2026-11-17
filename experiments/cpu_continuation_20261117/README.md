# BridgeCaption experimental CPU training — 17 Nov 2026

This is a **free, bounded, research-only** fine-tuning experiment for the existing French->English and Spanish->English Marian LoRA translation adapters. It does **not** train a new foundation model, replace the in-app Gemini/provider selection, or establish production readiness.

## First completed French run (GitHub Actions #44173650150)

- 96 real AdamW optimizer updates completed on an unmodified CPU runner, training on 1,536 French-English Tatoeba/ManyThings pairs. 128 pairs were reserved for **internal development scoring only**.
- Existing adapter: chrF **76.4287**, BLEU **63.2285**.
- Step 48: chrF **76.5090**, BLEU **63.4110**.
- Step 96: chrF **76.4298**, BLEU **63.1424**.
- **No candidate qualified.** The existing selected research adapter remains unchanged. This result does not demonstrate improved real-world translation.
- The run uploaded rejected research checkpoints and aggregate results to GitHub Actions artifacts (14-day retention). The original adapter is permanently preserved under `research/neural_continuation_20261116/french/`.

## Current reproducible experiment

The next workflow revision runs French and Spanish in separate CPU jobs, each with a pinned base model, checksum-verified historical research adapter, and a locked evaluation gate. For each language:

1. Download and SHA-256 verify the exact original public Marian base files from the model's pinned Hugging Face revision.
2. Download source-only Tatoeba parallel sentences through ManyThings.org; record the archive SHA-256 and remove exact duplicate sources and references before splitting 1,536 train / 128 development examples.
3. Evaluate the **existing adapter**, then run 96 LoRA optimizer updates with fresh AdamW state (rank 8, alpha 16, effective batch 16, LR 2.5e-5). Record candidates at steps 48 and 96.
4. Mark a candidate for *further research* only if it exceeds the prior adapter by at least **0.25 chrF** on development and does not reduce BLEU. No checkpoint is installed into the live app.
5. Preserve aggregate scores and candidate adapter weights as private-to-workflow artifacts. **Do not upload raw Tatoeba sentences or full pretrained weights.**

## Reproduce (free CPU, no provider credentials)

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/requirements-replay.txt
python -m pip install 'huggingface-hub>=0.34,<1.0'
python -m unittest discover -s experiments/cpu_continuation_20261117/tests -p 'test_*.py'
python experiments/cpu_continuation_20261117/train.py --language french --output-dir /tmp/fr-cpu-experiment
python experiments/cpu_continuation_20261117/train.py --language spanish --output-dir /tmp/es-cpu-experiment
```

The output folder must not exist in advance. Model downloads require Internet access. These steps do not need a GPU.

## Run the saved custom models locally (no GPU or API key)

These models are **experimental** French/Spanish to English text translators. The command downloads checksum-pinned original Marian weights to the user's local cache, verifies the existing selected adapters, and prints the model's translation. No cloud transcript provider is used, no audio is recorded, and no application setting is changed.

```bash
python experiments/cpu_continuation_20261117/local_translate.py --language french --text "Je veux reserver une table."
python experiments/cpu_continuation_20261117/local_translate.py --language spanish --text "Quiero reservar una mesa."
# To compare with the unchanged base model:
python experiments/cpu_continuation_20261117/local_translate.py --language french --text "Je veux reserver une table." --baseline
```

Requires the dependencies listed above, Internet for first-time pinned base downloads, and approximately 350 MB per language in a user-owned model cache. This is **not** a live speech-recognition system or an activated BridgeCaption GUI provider. A trained LoRA adapter is not a standalone language model.

## Evidence and limits

- Tatoeba/ManyThings.org English-French and English-Spanish datasets require attribution and license review before redistribution. Existing pinned Marian model cards/weights remain subject to their upstream terms.
- **A new random split from the same publicly available corpus does not guarantee independence from earlier adaptation, other Tatoeba sentences, or original OPUS pretraining.** Development results are screening signals, not untouched-test or independent human-validation results.
- Historical external evaluation sets have already been used; this run does not recycle them for tuning. No guaranteed generalization to slang, long dialogue, live audio or other languages.
- The historical `research/CHECKSUMS.json` describes an immutable prior delivery. New experiments are deliberately in `experiments/`, so tests verifying the historical delivery remain meaningful.
- A GPU alone cannot fix insufficiently distinct data or lack of blinded human review. Before deploying, reserve new external conversation-domain evaluation examples and compare the frozen model against the unchanged base, previous selected adapter, and current live provider.

## Work-mode handoff

Use this branch/PR as the reproducible starting point. Review GitHub Actions logs and artifact reports, save any genuinely improved candidate with a new immutable checksum and provenance, obtain independent *new* test data, and only then consider adding an explicitly opt-in local provider. Do not claim that a passing CI run proves a production-ready model.
