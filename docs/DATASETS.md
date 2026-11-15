# Speech and translation datasets

The repository contains only original, synthetic text examples. It does not contain public dataset audio or any recorded meetings.

## FLEURS: speech recognition across languages

https://huggingface.co/datasets/google/fleurs

Use held-out audio+reference transcripts for Spanish, Swahili and Yoruba (verify each language's current dataset config). Start with 30–50 samples per language. Create a CSV manifest with columns file, language, transcript.

Run: python -m bridgecaption.asr_eval --manifest /path/to/manifest.csv

Report WER separately per language. For scripts without word spaces, consider CER rather than naive whitespace WER. Avoid using the test samples for improving transcription prompts.

## Mozilla Common Voice: accent evaluation

https://commonvoice.mozilla.org/en/datasets

Use public, appropriately licensed speech. Compare groups only when accent labels exist and group sizes are adequate. Do not infer accents from demographic characteristics. Compare manually specified vs automatically detected language on identical clips as a simpler alternative experiment.

## FLORES+: translation evaluation

https://huggingface.co/datasets/openlanguagedata/flores_plus

Use aligned sentence pairs for Spanish, Swahili and Yoruba -> English. Verify licensing and language tags. Keep development data separate from devtest. Score with sacrebleu chrF on held-out translations and inspect errors.

FLORES+ contains isolated sentences, not full conversation turns. Do not claim it proves contextual meeting translation improves; use independent contextual examples evaluated by competent speakers.

## Optional CoVoST 2: end-to-end speech translation

https://github.com/facebookresearch/covost

Use only available language pairs, respect the license, and keep a held-out evaluation set. It can test speech->English translations rather than the ASR and text-translator stages separately.

## Slang and contextual evaluation

data/context_demo.jsonl is a tiny, synthetic demonstration benchmark. The examples include target glossaries and are NOT sufficient to validate generalization.

For meaningful results:
1. Gather 30+ independently written or permissioned examples with surrounding conversational turns.
2. Have a fluent speaker provide a reference translation.
3. Do not use held-out examples to write the final glossary.
4. Run bridgecaption-eval --data independent_examples.jsonl --out results/context.csv.
5. Compare the same model with context disabled/enabled, and manually judge slang meaning and terminology preservation.
6. Break down results by source language, include failures, and measure latency.

Do not claim to have trained Whisper or Gemini. Only the optional topic classifier in bridgecaption.training is trained locally.
