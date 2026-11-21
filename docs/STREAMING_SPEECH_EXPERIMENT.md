# M5: opt-in streaming speech experiment

This is the research interface for the [one streaming speech model target](UNIFIED_SPEECH_MODEL.md), not a product launch. It does not ship a model or loader, collect audio, modify the app, add a provider choice, use GitHub Actions, or select a fallback. Current audio, Whisper, provider, consent, installer and desktop update paths remain the incumbent baseline.

## Boundary

The research runtime accepts small, ordered mono PCM frames from a caller that already has permission to use the audio. A reviewed local loader gives the runtime one immutable model session; changing a supported language direction opens new stream state on that same loaded model, not another model. No candidate, automatic language choice, or context capability is presumed. Unknown or unverified directions (including unlisted locale variants), sequence gaps, and invalid frame sizes fail closed. Raw PCM has no header, so the caller must supply actual 16 kHz mono signed 16-bit little-endian samples; byte alignment cannot prove sample rate or channel count. A stream with no PCM cannot produce a caption.

An output carries paired source and translated text. Provisional outputs can be revised; stable outputs are append-only for that stream. The research bridge translates stable events into the app's existing finalized caption event shape and keeps provisional events on a separate sink. It is not attached to the desktop controller, cannot claim device consent, and does not read the microphone or make network calls. The caller must marshal callbacks to a UI thread if it uses a UI. Synchronous framing in this prototype must run off the UI thread in any future integration; it is not an admission of a real-time production capture loop.

### Local developer entry points

- `SingleStreamingSpeechRuntime` starts disabled. For research only, give it a `SpeechQualification` constructed from a separately pinned speech-specific development receipt, one reviewed loader, and explicit opt-in.
- Call `start_stream(source=..., target=...)`, push consecutive `PCMFrame` objects, and finish or abort before changing direction. New decoder state reuses the same loaded backend. The loader must independently verify actual immutable weights, config and tokenizer bytes and isolate model execution. Checking the reported checkpoint identity is not a weight attestation.
- Put `CaptionEventBridge` between the admitted stream and an explicit event sink in a local research harness. It derives the language from the admitted stream. Its caption delay is only since the triggering frame ended, not end-of-utterance or user-visible translation latency.
- `stream_metrics.measure_stream` accepts supplied integer millisecond evidence for utterance start, capture end, provisional and stable events, and observed pending audio. It reports sample counts, latency percentiles, extra provisional emissions, and backlog summaries without text or utterance IDs.

## Measure the right things

The delivered synthetic harness checks ordering, finality, failure isolation, bounded frame/session state, and model identity across directions. It records redacted timings only when provided with real event boundaries by a caller. Synthetic event times show that calculations behave correctly, not that a model reaches the goals or actually performs incremental decoding.

A later matched comparison must keep audio, noise, clip boundaries, time source, direction, platform and CPU settings fixed. Report time to first provisional **and** time from reference utterance end to stable translated output; track revisions, skipped/gapped input, peak and late backlog, memory, and uninterrupted session duration. Compare ASR errors against current Whisper and bilingual meaning against independently judged baselines by direction. Separately measure cold start and end-to-end playback, and keep raw conversations, personal metadata and sealed references out of Git and logs.

## What still blocks activation

There is no selected checkpoint, verified streaming implementation, production consent flow for this route, or real audio performance and language-coverage result. The under-one-second / under-two-second latency and 60-minute stability goals are not measured. Historical specialists remain offline references, not substitutes when the experimental stream fails. Any future production opt-in must be reversible, separately approved and evaluated against the incumbent. Rollback is disabling the research adapter; existing app paths and stored weights do not need undoing.
