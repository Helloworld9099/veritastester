# M14: Capture-paced native Q8 ASR replay

This research-only replay feeds the same eight pinned Spanish and French FLEURS DEV clips at capture speed, alternating languages. It then repeats the initial Spanish/French pair after six intervening clips on the same resident recognizer to spot reset problems. It measures source ASR only, not the two-head translation product, microphone-device latency or GUI rendering. No training, new recording, holdout access, production change, or merge.

## Timing and invariants

The runner verifies the [official NVIDIA release](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/tag/v0.3.0), its C API header, the [pinned Q8 model](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf), and all WAVs before replay. The [native C API](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/20999cc921d2b14aa58708f61756a8b2e787ba20/include/nemo_speech/asr.h) provides one recognizer, separate streams and an explicit finish operation. The helper sets two CPU cores of affinity and an OpenMP ceiling of two threads; the C ABI leaves native backend worker count at its default. Do not equate this with M13's explicit two backend threads. Backend context is three tokens and the submission block is 320 ms (5,120 PCM16 samples), with the remainder at the exact scheduled EOF.

- Schedule absolute releases from a monotonic epoch for each clip, when the corresponding audio block would have arrived. A slow call cannot move the physical release clock. Pass each sample once, drain results after each submission, and explicitly finalize EOF. Each clip opens a new stream with its own locale prompt; the model is loaded once. Timing starts after the stream opens and includes native initial inference; model load is separate.
- First caption means the first nonempty native interim text measured from the start of capture, including any initial quiet. Final delay is the final native result's observed availability minus the scheduled EOF. No stability/translation claim is attached to these ASR events.
- Backlog is the duration of **complete, already-due audio blocks still waiting to be submitted** when the current block finishes draining. A block available partway through a decode is counted at step completion if still waiting. Max dispatch lateness tracks smaller delays. Native processed-audio progress can lag submitted audio due to lookahead; it is measured separately only when the native API emits a partial. The harness refuses missing samples, early submission, malformed event order and missing or duplicate final output.
- The redacted receipt contains no transcript, token, media, path, custom microphone capture, or third-party provider data. Repeated clips are a same-model language-reset check, not independent quality observations. The harness checks exact repeat finals locally and computes reference WER with the same normalization as M12–M13.

## Run locally

Use the exact [M13 native CPU SDK and model](NEMOTRON_Q8_CPU.md), GNU C++, and the exact [M12 DEV input manifest and normalized WAVs](NEMOTRON_CACHE_SMOKE.md). Use absolute canonical paths and a new report path.

```sh
PYTHONPATH=.:src timeout --signal=TERM --kill-after=5s 290s \
  python -m research.unified_models.nemotron_q8_paced \
  --exe "$NEMO_SPEECH_CPU" --model "$NEMOTRON_Q8_GGUF" \
  --manifest "$DEV_MANIFEST" --audio-root "$AUDIO" --report "$NEW_REPORT"
```

The helper links to the vendor C library and a static GNU C++ runtime to avoid mixing the host compiler's C++ ABI with the vendor bundle's older libstdc++. No application components, training assets, or downloaded media are installed by this command.

## Observed results

One test session used a single resident recognizer and ten serial streams: all eight exact clips (80.22 seconds of audio), then the original Spanish/French pair again (22.86 seconds), without microphone or translation. The [redacted receipt](../research/unified_models/nemotron_q8_paced_report.json) records timing and word-count evidence.

| Source, primary clips only | First nonempty ASR partial, capture start | Final ASR result, after EOF | ASR word errors |
| --- | ---: | ---: | ---: |
| Spanish, 4 | 2,041–2,390 ms | 202–338 ms | 2 / 93 |
| French, 4 | 1,101–3,315 ms | 270–302 ms | 8 / 96 |
| All 8, median | 2,193 ms | 299 ms | Not pooled for qualification |

All eight final results arrived within two seconds of EOF. **None** produced an interim caption within one second from capture start. This start point is not speech onset; clip lead-in and the model's emission policy both influence the first-caption result. Model creation returned in 40 ms on this run, but the cold first native decode step took **801 ms**, with 640 ms of audio waiting and 481 ms worst dispatch lateness. The seven subsequent primary clips had at most 60 ms of a due block waiting at native step completion; the two repeated streams peaked at 80 ms. This does not prove hour-long stability. Native output reported up to 70 ms between partial progress and the submitted audio frontier, which is distinct from unsent audio.

One recognizer was created and all ten streams finalized, with no drops. The two repeated final strings matched their original streams exactly after switching languages. Each primary clip had the same normalized ASR error count as M13; cross-run exact transcript parity between paced and unpaced was not tested. The two repeat clips are not additional quality examples. Peak child RSS across helper compilation and replay was approximately 0.87 GiB; it is not sustained-memory evidence.

**Decision:** Keep native Q8 as the CPU development baseline, but don't call the first-caption target met. Next isolate first-caption delay: compare prewarming to cold starts and test shorter *supported* chunk/lookahead settings against the same ASR accuracy checks. Report capture-start and speech-onset timing separately before adding a second head or increasing complexity. Do not infer translation or product readiness from ASR final timing.
