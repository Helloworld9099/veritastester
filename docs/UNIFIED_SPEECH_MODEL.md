# BridgeCaption's unified speech model target

The target is **one multilingual streaming speech model** that takes live audio and produces both original-language captions and translated captions. The reference design shares an incremental audio encoder and model state across transcription, language identification, and speech translation. It may still need separate output heads or decoding passes. One checkpoint does not imply one decoding pass, equal quality in every direction, or half the compute.

```text
Live audio → shared streaming encoder → original transcript ↘
                                   └→ translated transcript → live overlay
```

No per-language model reload is planned. Source and target language are routing **inputs to the same verified model**, not selectors for different weights. Unsupported directions must fail explicitly. The existing Whisper-plus-translator app stays unchanged until this path wins comparable tests and the user approves deployment. Historical Spanish, French and Mandarin artifacts stay intact: use licensed, approved specialists for isolated benchmarks and potential teacher signals, not runtime switching or direct weight merges. Restricted lineages stay isolated. Text-only examples cannot teach audio recognition; future audio evaluation/training needs appropriately licensed audio, transcripts, references and timing annotations.

## Prototype status

M5 adds an [opt-in PCM and caption-event contract](STREAMING_SPEECH_EXPERIMENT.md) and bounded synthetic checks. It does not claim a real streaming model or switch the app. An unchanged incumbent pipeline is still the baseline. M6 adds [local evidence and opt-in preparation tools](OPT_IN_MULTILINGUAL_RESEARCH.md); it does not select a model. The [first real local DEV pilot](FLEURS_SPEECH_PILOT.md) measures eight read clips and verifies the shared-encoder mechanics, not native streaming or the full incumbent gate.

## A separate research track, in the same PR

Keep the completed infrastructure numbering (inventory, evaluation, training coordination, local workflow) intact. Treat the proposed speech milestones as a **parallel research sequence**, not completed work and not replacements already made:

1. Benchmark pinned unified speech candidates against the existing Whisper-plus-translation baseline, with equal audio, boundaries, directions and hardware. Start with small, rights-reviewed, CPU-feasible dev comparisons. Do not pick a model based on advertised language count.
2. Select a shared encoder/decoder only after measuring model size, rights, streaming behavior and meaning/speech metrics. Compare true incremental processing with chunk-based baselines.
3. Implement provisional and stable caption events, measure revision rate and queue growth, and don't translate a fragment earlier than the language permits.
4. Evaluate rights-cleared distillation or auxiliary text objectives; then jointly evaluate recognition and translation and profile CPU, memory and long-session behavior.
5. Consider a reversible feature flag only after independently evaluated directions pass. Keep the existing app as rollback, not an automatic specialist switch. Extended live sessions and independent bilingual review are required before promotion.

Initial engineering targets are under one second to a first provisional caption, under two seconds to stable translation where linguistically feasible, and at least 60 minutes without a growing backlog, memory growth or a model reload on language change. These are goals, not measurements. Track source-caption quality against current ASR and translation quality per direction, including context, slang and meaning-changing errors. A shared text-only runtime can help with development experiments but is not the end-to-end speech interface.
