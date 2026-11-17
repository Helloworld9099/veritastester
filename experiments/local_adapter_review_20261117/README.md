# Local, pinned research-model comparison

This optional command evaluates BridgeCaption's **saved custom-trained French/Spanish translation adapters** on your own short text. It is intentionally separate from the live microphone/caption app and does not change Gemini or other providers.

## Supported variants

- `base`: Original unchanged Helsinki-NLP Marian translation model.
- `historical`: Prior French or Spanish research LoRA checkpoint (the latest selected prior to OPUS Books).
- `books`: New saved OPUS Books v1 LoRA candidate from PR #11 (French checkpoint step 144; Spanish step 96). Research-only and not independently validated for production use.

Every invocation verifies pinned original base-model download hashes and the selected adapter's SHA-256 against the saved experiment report / checksums; weights use safetensors, not arbitrary Python deserialization. All inference after first-time model download is local CPU inference.

## Install and run (repository root)

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/requirements-replay.txt
python -m pip install 'huggingface-hub>=0.34,<1.0'

python experiments/local_adapter_review_20261117/translate.py \
  --language french --variant books --text "Je veux reserver une table."
python experiments/local_adapter_review_20261117/translate.py \
  --language spanish --variant historical --text "Quiero reservar una mesa."
python experiments/local_adapter_review_20261117/translate.py \
  --language spanish --variant base --text "Quiero reservar una mesa."
```

For all six language/variant combinations, repeat the command with each explicitly selected `--variant`. A successful sample translation is **not** independent evidence of 90% semantic adequacy. The tool handles only text input; it does not perform ASR, speaker diarization, contextual memory, or live graphical overlay.

The training sets and original pretrained model may overlap; do not score these ad-hoc examples as untouched quality tests. For the user-requested 90% correctness milestone, use the independent two-rater human adequacy evaluation rubric under `experiments/translation_quality_20261117/`.

The runner stores only public base weights in a user-owned cache, never sends the input text to an external API, and never modifies production configuration. License terms of upstream Marian base weights and research-only corpus-derived adapters remain applicable.
