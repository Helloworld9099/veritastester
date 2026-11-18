# MASSIVE Stage 2: Continue from saved spoken-command adapters

**Research only.** This iteration trains the BridgeCaption French→English and Spanish→English adapters from the *already trained* MASSIVE stage-one selected checkpoints rather than restarting from the older model. This is not a new foundation model and does not change the live translation provider.

## Why continue MASSIVE?

Stage one delivered a measurable gain on 256 held-out MASSIVE utterances from each language:

| Language | Previous chrF | Stage-one chrF | Previous BLEU | Stage-one BLEU |
|---|---:|---:|---:|---:|
| French | 63.768 | **66.091** | 35.667 | **43.771** |
| Spanish | 61.114 | **63.945** | 32.151 | **39.256** |

These are **automatic corpus scores, not percent-correct translations**. They justify further controlled experiments, not a 90% semantic-adequacy claim.

## Reproducibility and non-leakage

1. Use **AmazonScience/massive** at the exact author-modelled dataset revision `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740` (CC BY 4.0; attribute FitzGerald et al., Amazon Science and original SLURP authors). French `fr-FR`, Spanish `es-ES`, English `en-US`.
2. Verify the saved stage-one adapter against its immutable code-pinned SHA-256:
   - French: `5c5094a3f8dfc9d783f6e8482ae8ed2fad7e632b81b31f20f54fa8f288538ad8`
   - Spanish: `e60fbadae2037686f9693af4be2e7e59b128aa44f4be8c287f40b7f893b0a004`
3. Reconstruct the **exact original MASSIVE stage-one train/dev/test SLURP IDs** and verify their SHA-256 hashes against the saved report before selecting more data.
4. Select *new* source/reference/ID-disjoint examples, retaining the author's original partition boundaries:
   - 5,120 additional official-**train** utterances (no stage-one train, dev or test replay).
   - 256 fresh official-**validation** examples for model selection.
   - 256 fresh official-**test** examples **never evaluated in stage one**, to be assessed **once**, only after checkpoint selection.
5. Use the already pinned Marian base model and a fresh optimizer to perform 320 additional LoRA updates, with evaluations at steps 64/128/192/256/320. Choose a research candidate only when development chrF improves at least 0.25 over the saved stage-one adapter and development BLEU does not fall. Never use test scores to select or tune.
6. After selection, evaluate the unchanged base, the **stage-one MASSIVE checkpoint**, and the selected stage-two candidate (when eligible) on the *new* 256-item official-test sample. These examples become **consumed** after that evaluation and must not be recycled for a later "independent test".
7. Retain *all* candidate checkpoints, the chosen model (if any), SHA-256 manifest and aggregate-only results, without uploading raw utterances, individual inference output, or private data.

## Limitations

This experiment isolates held-out MASSIVE examples from BridgeCaption's adaptation process. It cannot prove non-exposure during original Marian pretraining; MASSIVE localizations can differ from literal translation, and no end-to-end speech recognition or multilingual human evaluation is performed.

**The requested 90% correctness milestone remains unverified** until independent human bilingual adequacy annotations establish it using the separate `experiments/translation_quality_20261117/score_human.py` rubric. Neither 90 chrF nor 90 BLEU is an accuracy percentage.

## Run on free CPU

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/requirements-replay.txt
python -m pip install 'huggingface-hub>=0.34,<1.0' 'datasets==2.21.0'
python -m unittest discover -s experiments/massive_stage2_20261117/tests -p 'test_*.py' -v
python -u experiments/massive_stage2_20261117/train_stage2.py --language french --output-dir /tmp/massive-stage2-fr
python -u experiments/massive_stage2_20261117/train_stage2.py --language spanish --output-dir /tmp/massive-stage2-es
```

The GitHub Actions workflow `.github/workflows/massive-stage2-research.yml` runs both languages on standard free CPU runners, retaining aggregate reports and LoRA weights as **research-only** outputs. Prior checkpoints are untouched.
