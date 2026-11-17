# BridgeCaption: document-disjoint OPUS Books adaptation

**Status: research only.** French->English and Spanish->English Marian LoRA experiments initialized from the existing checksum-verified adapters merged in PR #5. This experiment does not change the live caption app, provider selection, or original checkpoint files.

## Why another corpus?

The previous Tatoeba/MultiThings tuning rounds reached small and inconsistent gains. Here the model sees a **different domain**: aligned literary text from OPUS Books v1. The parser uses the corpus's paired source/target texts **and its `.ids` file** to partition by bilingual book/document rather than by sentence. Source-text and English-reference exact duplicates are removed before splitting.

**Data source / rights:** OPUS Books v1 contains bilingual book alignments collected by Andras Farkas and distributed by OPUS/Jorg Tiedemann. It is free for **personal, educational, and research use**, but **commercial use and mass redistribution are not granted without permission**. The workflow downloads the corpus transiently and records its authoritative SHA-256, and does **not** upload text, copyrighted examples, or base model weights. OPUS upstream source: https://opus.nlpl.eu/datasets/Books ; original archived downloads documented by Helsinki-NLP's `opus_books` dataset-info manifest.

## Protocol frozen in code before scoring

- Each language uses its exact pinned base Marian revision and its unchanged, checksum-verified selected LoRA adapter.
- 3,072 aligned literary training pairs; 128 development pairs; 256 entirely separate, **book-group-disjoint** held-out pairs.
- 192 new updates, from the previous adapter with a **fresh** AdamW optimizer (effective batch 16, LoRA rank 8 / alpha 16, LR 2.5e-5). Model selection may consider only development chrF/BLEU at 48, 96, 144, or 192 updates.
- New weights are marked *research candidate* only when development chrF rises at least 0.25 relative to the original selected adapter and BLEU does not decline. No change to production.
- **Only after model selection is frozen** does the workflow compare the original base, previous adapter, and selected research candidate (if different) on the book-heldout test set.
- The book-heldout test is consumed once that comparison is run. Don't reuse it later to tune hyperparameters, choose checkpoints, or make a second "fresh test" claim.
- Development and test data are book-separated from *this* new training run but may be represented in the underlying OPUS base model pretraining. The results are conditional on a literary domain and do not establish slang, speech or contextual translation quality.

Run from repository root in a Python environment with `research/requirements-replay.txt`, `huggingface_hub` and CPU PyTorch installed:

```bash
python -m unittest discover -s experiments/opus_books_20261117/tests -p 'test_*.py' -v
python experiments/opus_books_20261117/train_books.py --language french --output-dir /tmp/books-french-1
python experiments/opus_books_20261117/train_books.py --language spanish --output-dir /tmp/books-spanish-1
```

The output directories must not already exist. The workflow at `.github/workflows/books-research.yml` runs free Linux CPU jobs, tests the parsing logic, and retains candidate LoRA weights and **aggregate metrics only** as GitHub Actions artifacts. It does not commit any training data or automatically deploy a model.

## What "90%" would mean

The requested **90% correctly translated** target cannot be established by chrF, BLEU, training loss, or exact match against one reference alone. The workflow reports those as **separately named automatic metrics**, never as a percent of correct translations. For a defensible 90% quality milestone, use a freshly collected, genuinely unseen conversation-domain evaluation with independent fluent bilingual human judgments on semantic adequacy. This run creates better model/data infrastructure but cannot certify that target.

## Handoff to Work mode

Review the CI job and model-selection report. Preserve all previously trained checkpoints and every candidate artifact needed for independent comparison. Require new licensing and dataset provenance checks before expanding to other corpora. If a new model beats the original on *development and an independent evaluation*, enable it behind an explicit opt-in research provider—not as a silent replacement for the live Gemini model.
