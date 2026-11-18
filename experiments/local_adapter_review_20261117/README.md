# Local, pinned research-model comparison

This optional command evaluates BridgeCaption's **saved custom-trained French/Spanish translation adapters** on your own short text. It is intentionally separate from the live microphone/caption app and does not change Gemini or other providers.

## Supported variants

- `base`: Original unchanged Helsinki-NLP Marian translation model.
- `historical`: Prior French or Spanish research LoRA checkpoint (the latest selected prior to OPUS Books).
- `books`: New saved OPUS Books v1 LoRA candidate from PR #11 (French checkpoint step 144; Spanish step 96). Research-only and not independently validated for production use.
- `massive`: Selected MASSIVE stage-one adapter from PR #16 (French and Spanish step 256). Most promising so far for short spoken-command text; **not** independently validated as 90% correct.
- `qed`: QED v2.0a educational-subtitles research adapter, **Spanish only**; French failed the pre-registered selection gate. QED training content is **research-use only**.
- `globalvoices`: GlobalVoices translated-news research adapter, **French only**; Spanish failed the pre-registered selection gate. Article-level copyright/attribution constraints apply.

Rejected candidates are intentionally unavailable as `--variant` options in this validated CLI. Their weights remain preserved in the research directories and can be examined separately.

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
python experiments/local_adapter_review_20261117/translate.py \
  --language french --variant massive --text "Mets une alarme demain matin."
python experiments/local_adapter_review_20261117/translate.py \
  --language spanish --variant massive --text "Pon una alarma mañana."
python experiments/local_adapter_review_20261117/translate.py \
  --language spanish --variant qed --text "Quiero entender esta lección."
```

Try only the explicitly supported language/variant combinations; some research branches have no accepted checkpoint for one language. A successful sample translation is **not** independent evidence of 90% semantic adequacy. The tool handles only text input; it does not perform ASR, speaker diarization, contextual memory, or live graphical overlay.

The training sets and original pretrained model may overlap; do not score these ad-hoc examples as untouched quality tests. For the user-requested 90% correctness milestone, use the independent two-rater human adequacy evaluation rubric under `experiments/translation_quality_20261117/`.

The runner stores only public base weights in a user-owned cache, never sends the input text to an external API, and never modifies production configuration. License terms of upstream Marian base weights and research-only corpus-derived adapters remain applicable.


## Efficient batch predictions for independent bilingual review

For a **new, appropriately licensed, never-used-before** set of French or Spanish utterances, prepare a **private** JSONL containing `sample_id`, `category` (`everyday`, `idiom`, `slang`, `context`, or `technical`), `source_group_sha256` (SHA-256 of the conversation/speaker group ID), and `source_text`. No reference translation is needed in the input and no private text is committed to Git.

Example input line (illustrative only):

```json
{"sample_id":"speaker-01-case-001","category":"idiom","source_group_sha256":"<64 lowercase hex>","source_text":"J'ai le cafard."}
```

Then generate **frozen real model predictions** in one model load, processing batches of up to 256 utterances with an 8-example inference microbatch:

```bash
python experiments/local_adapter_review_20261117/translate.py \
  --language french --variant massive \
  --input-jsonl /secure/review/independent_sources.jsonl \
  --output-jsonl /secure/review/frozen_predictions.jsonl
```

The CLI prints the selected adapter's exact SHA-256 on **stderr**, and outputs a JSONL with each original source and `candidate_translation` (never silently rewritten by another service). It refuses to overwrite an existing output file. Full source/translation text remains on your local computer. **Do not upload the private review packets or personal information to GitHub.**

Finally, hand the predictions to `experiments/translation_quality_20261117/collect_reviews.py prepare`, passing the printed `--checkpoint-sha256`. This freezes blind review packets for two independent fluent bilingual human raters; use its `score` subcommand only after receiving actual ratings. No AI-only metric can certify the requested 90% human adequacy.

This batch feature optimizes inference efficiency, **not model quality**, and does not affect the microphone/overlay or active translation provider.
