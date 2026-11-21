# Paced audio replay: exposing the live-path timing gap

The [closed-clip pilot](FLEURS_SPEECH_PILOT.md) measured computation with audio already available. This research-only follow-up releases the same checksum-verified FLEURS development clips at the rate they would arrive from a 16 kHz mono source. It records elapsed time to a paired draft, elapsed time from the end of the audio to the final pair, and how much newly arrived audio had not yet been processed. Use these measures to diagnose responsiveness, not to extrapolate from closed-clip throughput.

This is **not native incremental inference**. Every dispatched prefix gets a new Whisper encoder forward pass, shared by two decodes (original transcription and English translation). One model is loaded throughout; there is no specialist routing or second encoder for the translation task. No microphone, user conversation, live overlay, app provider, Gemini, training, or model promotion is involved. No tool requires GitHub Actions.

## Measurement boundaries

- The capture time origin is the monotonic instant replay starts. Draft latency includes waiting for the triggering prefix to arrive and both decodes. Stable lag is measured from the exact end of the audio to the delivered final pair, not from the wall time at which a final job happens to start.
- A single worker can fall behind incoming audio. The scheduler keeps no growing queue and can coalesce stale provisional opportunities; a missed or empty draft is not counted as a delivered first caption. Replay still measures capture lead and final overrun rather than converting dropped provisional work into a claimed latency win.
- Timing is measured on this CPU host with publisher read speech. It does not measure microphone capture, UI rendering, production end-to-end latency, independent bilingual meaning, noisy/spontaneous speech, or a sustained live session. No head-to-head comparison to the unchanged Gemini-backed app was run.
- The eventual model needs true incremental state and an authorized matched baseline. Replay itself does not change the product or choose any checkpoint for deployment.

## Reproduce locally

Prepare the pinned DEV manifest, audio and eleven Whisper model files as for the closed-clip pilot. Keep the manifest, audio, weights and any unredacted inputs outside Git. Use an independently saved lowercase SHA-256 for the exact manifest you intend to run.

```sh
PYTHONPATH=.:src timeout --signal=TERM --kill-after=5s 115s \
  python -m research.unified_models.whisper_paced_replay \
  --manifest "$DEV_MANIFEST" --audio-root "$AUDIO" \
  --expected-manifest-sha256 "$DEV_MANIFEST_SHA" --model-dir "$MODEL" \
  --report "$NEW_REPORT" --threads 2 --cadence-ms 3000 \
  --max-clips-per-direction 1 --clip-deadline-seconds 45
```

The per-clip deadline is cooperative; the outer process watchdog is needed for a blocked decoder. The runner writes a new redacted JSON report rather than overwriting a prior run. Capture lead is observed at dispatch and decode completion: released audio beyond the prefix supplied to that decode. It is not a continuous queue measurement. Final processing always uses the whole clip if its deadline allows a call; an expired or empty final is recorded as a failure rather than skipped or replaced.

## Observed CPU diagnostics

At 3.0-second prefix cadence, two threads and one clip per direction ([redacted receipt](../research/unified_models/fleurs_paced_dev_3000ms_report.json)):

| Direction | Clip length | Capture start → first paired draft | Capture end → final pair |
| --- | ---: | ---: | ---: |
| Spanish→English | 8.40 s | 4.35 s | 2.11 s |
| French→English | 14.46 s | 4.31 s | 3.87 s |

Eight encoder forwards produced sixteen decode calls. Both clips delivered finals. The French 12-second prefix completed after the clip ended, was suppressed, and delayed dispatch of the full-clip final by 0.45 seconds. Capture lead at decode completion reached 1.66 seconds in the Spanish clip and 2.46 seconds in the French clip. The percentiles in the receipt have one observation per direction and are not population estimates. The 3-second cadence alone precludes a sub-second first draft; this run does not mean a faster cadence or a genuinely incremental encoder has the same limit.

An independent process replayed those same two clips at 1.5-second cadence ([redacted receipt](../research/unified_models/fleurs_paced_dev_1500ms_report.json)):

| Direction | Capture start → first paired draft | Capture end → final pair |
| --- | ---: | ---: |
| Spanish→English | 2.42 s | 2.74 s |
| French→English | 2.51 s | 4.41 s |

The more frequent replay gave earlier drafts but later finals in both observations. It increased computation to 14 encoder forwards and 28 decodes, with seven computed drafts discarded as stale and two scheduled drafts coalesced without decoding. Both final pairs arrived. Final WER and chrF2 were identical by direction across the two runs. These are single observations per direction and cadence, not a repeatable latency ranking or an incumbent comparison. Neither cadence met both targets on these clips. The practical next direction is a genuinely incremental candidate with a bounded finalization budget, not simply shortening the window between full-prefix re-encodes.
