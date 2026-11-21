# M13: Native 8-bit Nemotron CPU replay

Research-only ASR diagnostic on the same eight pinned Spanish and French FLEURS DEV recordings used for [M12's FP32 experiment](NEMOTRON_CACHE_SMOKE.md). No model training, new data, translation head, app change or final holdout.

## Reproduce

Use NVIDIA's official [NeMo Speech C++ v0.3.0 Linux x86-64 CPU bundle](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/tag/v0.3.0) (tag code commit `20999cc921d2b14aa58708f61756a8b2e787ba20`) and [publisher Q8_0 GGUF at the same model revision used in M12](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf). The release archive matched both the vendor checksum file and the GitHub release digest. The runner verifies the local executable and the complete top-level bundled runtime library inventory, including symlink destinations, before passing it any data. The model, input manifest, original WAVs, and canonical replay WAV bytes are also checked. Keep all runtime/model binaries, transcripts and WAVs out of Git. Preserve NVIDIA/OpenMDW notices if redistributing anything covered under that license.

```sh
PYTHONPATH=.:src timeout --signal=TERM --kill-after=5s 330s \
  python -m research.unified_models.nemotron_q8_smoke \
  --exe "$NEMO_SPEECH_CPU" --model "$NEMOTRON_Q8_GGUF" \
  --manifest "$DEV_MANIFEST" --audio-root "$AUDIO" --report "$NEW_REPORT"
```

Use absolute canonical input paths. Each clip runs in a fresh local CLI process with two ASR backend threads and concurrency one, its fixed M12 locale prompt, 320 ms input blocks, right context three and two measured repeats. Native `bench asr --mode stream` feeds recorded audio without a real-time wait and measures each item after the engine's warmup. No microphone, companion model, service, or transcript upload is used. Only the local temporary benchmark directory receives final text for scoring. Timed transcription output is checked for repeat mismatches and is not saved in the report. WER uses the M12 accent-preserving normalization, same references, and same clip order. Full-process wall time is recorded separately and includes setup plus both repeats.

The FP32 reference uses a Python/Transformers implementation with whole-clip mel precomputation; the Q8 runtime uses native C++ with its own streaming frontend and model conversion. Both request two backend threads, but this is a comparison of **two configurations**, not a controlled estimate of quantization's effect alone. The sample is tiny, read speech, and drawn from a dataset listed by the foundation model publisher. It does not qualify quality on independent conversation, sustained memory use, live device end-to-end latency or translation. Revert this research-only commit or stop invoking the runner to roll back.

## Observed on the eight clips

| Source | Audio | FP32 compute, M12 | Native Q8 compute, M13 | WER both runs |
| --- | ---: | ---: | ---: | ---: |
| Spanish (4 clips) | 43.38 s | 51.95 s | 14.65 s | 2 / 93 = 2.15% |
| French (4 clips) | 36.84 s | 43.78 s | 12.41 s | 8 / 96 = 8.33% |
| Total | 80.22 s | 95.73 s | 27.06 s | Not pooled for qualification |

The [redacted Q8 receipt](../research/unified_models/nemotron_q8_smoke_report.json) shows approximately **3.54×** the measured compute throughput of the M12 configuration and a per-clip mean of 0.329–0.349 compute seconds per audio second. This leaves compute headroom in unpaced ASR replay, not a measurement of first/final captions. Both Q8 repeats gave identical final text for each clip. WER matched FP32 on **every** clip by error count; no token/string parity across runtimes was asserted. The downloaded Q8 model is 742.09 MB vs. 2,552.06 MB decimal for the FP32 artifact. Linux peak child RSS during the Q8 benchmark was 914,408 KiB (0.87 GiB); the reference process that ran both FP32 paths peaked at 3,127,892 KiB (2.98 GiB), different process scopes. These figures do not isolate the effect of quantization from the native runtime.

Recommendation: keep Q8 as a **development** CPU candidate. Next run capture-paced PCM with one resident runtime and verify first partial, finalization, long-session backlog, and clean state resets across language changes. The native runtime's encoder interface must independently satisfy the single-checkpoint dual-head contract before it can be used to ship a translator.
