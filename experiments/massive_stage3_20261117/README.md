# MASSIVE stage 3 — new samples, verified second-stage checkpoints

**Research experiment, not deployment.** This round continues French→English and Spanish→English Marian LoRA adapters from the **saved second-stage MASSIVE checkpoints**, while protecting the older checkpoints and every previous evaluation sample.

## Current baseline (stage 2)

| Model | 256-sample held-out chrF | BLEU |
| --- | ---: | ---: |
| French stage-1 MASSIVE | 68.995 | 43.412 |
| **French stage 2** | **70.747** | **48.378** |
| Spanish stage-1 MASSIVE | 63.473 | 38.414 |
| **Spanish stage 2** | **64.432** | **38.727** |

These are automatic reference-similarity scores on *MASSIVE speech-assistant text*, not percentage translation accuracy. No independent bilingual human adequacy assessment has been performed.

## Data integrity and provenance

- **Source:** AmazonScience/massive, pinned dataset revision `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`, CC BY 4.0. Attribute FitzGerald et al. (2023), Amazon Science and SLURP authors (Bastianelli et al.).
- Verify stage-one original author train/development/test SLURP ID hashes against permanent stage-one saved evidence.
- Reconstruct stage-two's **exact used IDs and paired source/English text** using the unchanged old selection algorithm and stage-two random seed. Verify its hashes against the permanently saved second-stage report and frozen evaluation. **Do not score the historical tests again.**
- Exclude all historical IDs, normalized French/Spanish sources and normalized English references from future fitting, development and testing. Keep official author's train, validation and test partitions.
- **New stage-three data** per language: 1,536 fresh official-train examples; 256 fresh validation examples for checkpoint selection; 256 fresh official-test examples that were *never evaluated in stages 1 or 2*.
- Initialize from **versioned checkpoint SHA-256**: French `0219365d7d51492f8319ebb8b6a8fccbf6da25ee3525505663ac52810b0cbcba`, Spanish `990fa755ab733a8a3a600615f2d9ed47c8ea24935cb88258e0f5a9f00888d791`.
- Frozen Marian base (model revision from historical manifest), rank 8/alpha 16 LoRA, effective batch 16, new AdamW optimizer at learning rate 2.5e-5; **96 updates** and development scoring at steps 24, 48, 72, 96.
- Accept a new *research* checkpoint only if development chrF exceeds the saved stage-two checkpoint by >=0.25 with no decrease in development BLEU. Otherwise preserve the stage-two checkpoint as preferred.
- Only after dev selection is frozen, compare the unchanged pretrained base, stage-two adapter, and selected stage-three candidate on the new 256-item official test **once**. The test is then considered consumed. No raw corpus, generated translations or personal data are uploaded.

## Run with free GitHub Actions CPU

The workflow `.github/workflows/massive-stage3-research.yml` starts automatically when this experiment branch changes, with separate French and Spanish jobs and CPU-only PyTorch. No GPU, provider credentials or paid API calls.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/requirements-replay.txt 'huggingface-hub>=0.34,<1.0' 'datasets==2.21.0'
python -m unittest discover -s experiments/massive_stage3_20261117/tests -p 'test_*.py' -v
python -u experiments/massive_stage3_20261117/train_stage3.py --language french --output-dir /tmp/massive3-fr
python -u experiments/massive_stage3_20261117/train_stage3.py --language spanish --output-dir /tmp/massive3-es
```

Experiments use distinct output folders and never overwrite previous adapters. Save every candidate model and aggregate evidence with checksums after successful CI before merging.

**Key scientific limit:** MASSIVE tests are disjoint from the **adaptation examples** in this repository, but cannot prove exclusion from Marian's original large-scale OPUS pretraining. The user-requested 90% independently bilingual-human-rated correctness remains unverified. A model passing the development gate or reaching 90 chrF is not necessarily 90% semantically correct. Use the separate blind bilingual evaluation protocol in `experiments/translation_quality_20261117` to assess that milestone.
