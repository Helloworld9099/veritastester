# First real local speech pilot

This is a small **development diagnostic**, not the matched M6 selection gate. It tests whether one local checkpoint can produce source transcripts and English translations from shared speech representations. It does not choose a product model, test incremental streaming, call Gemini, use test clips for inference, train a model, or change the app.

## Inputs and method

- **Speech and references:** [Google FLEURS, pinned release](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd), publisher DEV, CC BY 4.0. Four distinct public read-speech sentence IDs each from es_419 and fr_fr, selected in archive order before inference. Match the audio filename to its source DEV TSV and the language-independent sentence ID to the first en_us DEV raw transcription for that ID. The [FLEURS authors document the parallel design](https://simran-khanuja.github.io/assets/fleurs_slt.pdf). The [input receipt](../research/unified_models/fleurs_dev_pilot_inputs.json) contains IDs and hashes, not audio, transcripts or translations.
- **Checkpoint:** [OpenAI Whisper Base multilingual, pinned Transformers conversion](https://huggingface.co/openai/whisper-base/tree/e37978b90ca9030d5170a5c07aadb050351a65bb). All eleven local files matched publisher Git or SHA-256 LFS identities. [Whisper](https://github.com/openai/whisper) supports multilingual source transcription and translation into English. The pilot uses one FP32 CPU model, two intra-op threads, one inter-op thread, and complete 16 kHz mono PCM16 clips. An encoder hook counts actual calls; source transcription and English translation are separate non-sampling decoder passes. Closed-clip sharing does not establish live streaming.
- **Comparator:** Feed the same measured Whisper source transcript into the existing archived, pinned Marian adapter for that direction. This is a local proxy, **not the Gemini-backed app**. Both translation paths use the same paired English references and no external conversational context. WER and chrF2 are automatic diagnostics, not human percentages of correct meaning.

## Reproduce locally

Extract the exact listed DEV clips from their pinned language audio archives. Convert to 16 kHz mono PCM16 with FFmpeg and write canonical WAV headers; check both hashes in the input receipt. Supply a local JSON list with id, language, wav, source, reference, sha256 and duration; use exact source/English DEV references and a relative WAV name. Keep the manifest and media out of Git.

Use an existing Torch/Transformers/SacreBLEU environment and the eleven exact file pins in the runner. The archived Marian proxy needs all three historical arguments together; omit them for the one-checkpoint diagnostic alone.

```sh
python -m research.unified_models.whisper_fleurs_dev \
  --manifest "$DEV_MANIFEST" --audio-root "$AUDIO" --model-dir "$MODEL" \
  --report "$REPORT" --max-clips-per-direction 4 --threads 2 \
  --historical-repo "$PWD" --historical-base-cache "$HISTORICAL_BASES" \
  --historical-artifact-root "$HISTORICAL_ADAPTERS"
```

The runner does not download weights or audio. Timings are closed-clip compute with a single observation per clip, not live end-to-end latency or a speedup claim. The Marian calls include cold loads. The sample is tiny, is not necessarily outside model pretraining, and provides no evidence on slang, caption revision, noisy conversations, language switching or hour-long stability.

## Observed results

| Direction | Clips | Source WER | Direct Whisper chrF2 | Archived Marian proxy chrF2 |
| --- | ---: | ---: | ---: | ---: |
| Spanish→English | 4 | 8.60% (8 / 93) | 48.48 | 49.72 |
| French→English | 4 | 33.33% (32 / 96) | 49.00 | 51.26 |

The run recorded **eight encoder forwards for eight clips** and two decoding passes per clip. Both outputs were nonempty on all clips. Closed-clip computation for the two outputs ranged from 1.51 to 3.35 seconds on this CPU. Whisper-only process peak RSS before the proxy loaded was 730,084 KiB (about 713 MiB, including the Python/Torch process). See the [redacted measured receipt](../research/unified_models/fleurs_dev_pilot_report.json).

The direct translations scored lower than the local proxy in both tiny directions. Do not promote Whisper Base based on this result. The formal incumbent gate is still **not verified**: no matched repetitions against the live Gemini-backed path, equivalent streaming latencies, or independent bilingual evaluation. The existing app and weights remain untouched.
