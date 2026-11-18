> **Evaluation-integrity warning (November 18, 2026):** A qualitative audit found several unrelated French source/reference pairs in the previously scored TICO-19 cohort. The historical frozen numbers remain for reproducibility but **must not be used as reliable model-ranking evidence or a 90% adequacy claim**. See [REFERENCE_ALIGNMENT_WARNING.md](REFERENCE_ALIGNMENT_WARNING.md).\n\n# TICO-19 independent-domain evaluation — research only

## Purpose

BridgeCaption has made successive LoRA improvements on **MASSIVE short voice-assistant utterances**, but the same dataset family can exaggerate apparent progress. To assess transfer to a *different domain*, evaluate the unchanged Marian base, MASSIVE stage 1, MASSIVE stage 2, and MASSIVE stage 3 **without any additional model updates** on the TICO-19 medical/crisis bilingual translation benchmark.

**This is an external automatic reference benchmark, NOT proof of 90% bilingual human adequacy or production readiness.**

## Data rights and provenance

- TICO-19: [Translation Initiative for COVID-19](https://tico-19.github.io/), described by Anastasopoulos et al. The original project publishes benchmark content under **Creative Commons CC0**.
- Downloaded directly from `gmnlp/tico19` public Hugging Face dataset at exact revision `55d70dc0b1d1d0b2151c5e22815d823fedac3f2f`.
- Pinned files: `test/test.en-fr.tsv` and `test/test.en-es-LA.tsv`. They contain English→French/Latin-American-Spanish translation pairs. **Direction is reversed before evaluation** to French/Spanish→English, matching the original BridgeCaption Marian models.
- The source archive's SHA-256, exact selected case-ID hash and distinct document/source-group count are recorded. Select 256 examples per language deterministically before running model inference; do **not** use the TICO test or dev splits for training, checkpoint choice or hyperparameter tuning.
- Text from the benchmark is downloaded and processed on the GitHub Actions runner but **no raw source, reference, output translation or full pretrained weights** are uploaded or committed.
- Some medical text may overlap the original Marian pretraining corpus. Use the results as *out-of-domain diagnostic evidence*, not independent human correctness or a proof of zero pretrained leakage.

## Versions compared

`base`, `massive` (stage 1), `massive2` and `massive3`, for both French and Spanish. Every adapter must pass its immutable SHA-256 and training-provenance checks in `experiments/local_adapter_review_20261117/translate.py` before inference.

The workflow reports corpus chrF, BLEU and reference-exact agreement separately. **None is equivalent to a percentage of semantically correct translations.** A 95%-confidence, independent two-fluent-bilingual-human review remains necessary for the user's actual 90% goal; use `experiments/translation_quality_20261117` on a **different, previously unexposed conversation corpus**.

## Reproduction on a free CPU runner

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/requirements-replay.txt
python -m pip install 'huggingface-hub>=0.34,<1.0'
python -m unittest discover -s experiments/tico19_20261117/tests -p 'test_*.py' -v
python experiments/tico19_20261117/eval_tico.py --language french --output-dir /tmp/tico-fr
python experiments/tico19_20261117/eval_tico.py --language spanish --output-dir /tmp/tico-es
```

`.github/workflows/tico19-external.yml` runs this on free GitHub Actions CPU runners with separate French and Spanish jobs, preserving **only aggregate JSON metrics and pinned cohort hashes** as research artifacts. Do not re-run with different hyperparameters after seeing the test results and call it an independent holdout.
