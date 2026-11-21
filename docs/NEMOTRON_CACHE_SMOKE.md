# M12: Native Nemotron ASR and bounded FLEURS TRAIN sample

This is a research-only development step. The pinned model is a transcriber, not a speech translator. It was not trained, packaged or wired into the app. No final holdout was used.

## Inputs

- Model: [NVIDIA Nemotron 3.5 ASR Streaming 0.6B, exact publisher release](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/ea30d66debe3740a08b573244286791d423d6b3e); the 2,552,062,944-byte safetensors matched publisher SHA-256. Five small configuration/processor/tokenizer files also matched their publisher Git blobs and were independently SHA-256 pinned. No quantized or NeMo weights substituted. Load only verified local files; [license](https://openmdw.ai/license/1-1/) and third-party obligations in M11 still apply.
- Speech: the eight previously used and independently pinned [FLEURS DEV clips](FLEURS_SPEECH_PILOT.md), four Spanish and four French. This small, read-speech subset is not held out from upstream pretraining or fit to measure casual conversation. No provider calls, tuning or new recordings.

## TRAIN slice: integrity and limitations

A separate [redacted staging receipt](../research/unified_models/fleurs_train_audio_slice_report.json) records 16 publisher TRAIN clips: eight es_419 (79.92 s) and eight fr_fr (95.52 s). Both publisher ranges returned HTTP 206 with the first 12,000,000 bytes and expected declared total length at the exact [pinned FLEURS revision](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd). Each selected regular WAV member matched the stage M11 TRAIN label manifest by exact member name with the same source and English references. Per-clip original, normalized WAV and decoded PCM hashes were recorded locally. The checked-in receipt does not contain media, member names or transcripts.

To reconstruct: obtain the exact [M11 TRAIN manifest](STREAMING_ADAPTATION_PREP.md), request range `bytes=0-11999999` from `data/es_419/audio/train.tar.gz` and `data/fr_fr/audio/train.tar.gz`, and verify the response range, total size and prefix digest against the receipt. Iterate tar members without writing archive paths to disk. Select the first eight unique matched sentence IDs per locale with 1–15 seconds of audio, cap source member size to 2,000,000 bytes, and stop at eight. Normalize with `ffmpeg -nostdin -hide_banner -loglevel error -i pipe:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le -f s16le pipe:1`; wrap the result in a deterministic mono PCM16 WAV header. Keep original/normalized/PCM hashes, duration, declared TRAIN split, and joined dual labels in a local manifest only. The source-archive LFS SHA is publisher metadata: **the whole archive was not fetched**, so a partial hash is not a verification of the complete training corpus. M12 rechecked all 16 normalized WAV hashes and found zero exact decoded PCM matches against the eight M7 DEV WAVs. This does not establish near-duplicate, speaker or foundation-pretraining independence. The small slice is for a later engineering dry run, not representative ASR/translation training.

## CPU cache method and reproduction

The [official Transformers integration](https://huggingface.co/docs/transformers/v5.21.0/en/model_doc/nemotron3_5_asr) accepts acoustic cache and precomputed encoder outputs. The runner feeds disjoint log-mel slices to NVIDIA's greedy generation path, then independently passes each slice through the same encoder and invokes the unchanged RNNT head on precomputed projected frames. Encoder hooks check slice identity, reuse of attention and convolution state, and forward-call counts on both paths. Parity compares blank-inclusive token IDs and RNNT frame advances. It does not imply equivalent results to full-context decoding or a qualified translation path.

Use CPU PyTorch 2.7.1, Transformers 5.21.0, and librosa 0.11.0, outside the app's environment. Prepare the exact M8 local DEV manifest and audio plus the six local model files pinned in the runner. Run from the repository checkout with absolute canonical input paths and a new report path.

```sh
PYTHONPATH=.:src timeout --signal=TERM --kill-after=5s 240s \
  python -m research.unified_models.nemotron_cache_smoke \
  --model-dir "$MODEL" --manifest "$DEV_MANIFEST" --audio-root "$AUDIO" \
  --report "$NEW_REPORT" --threads 2 --clips-per-language 4
```

Whole-clip feature extraction happens before the model streams its mel slices. Any terminal padding is explicitly counted. This is not a paced live microphone benchmark, an end-to-end latency result, a proof of constant-memory operation, or translation accuracy. Keep the app, checkpoint and all holdouts unchanged; revert the research-only commit or stop using the runner to roll back.

## Observed results: parity passed, CPU speed did not keep up

One local x86-64 Linux run with two CPU threads, FP32, and the model's 320 ms nominal chunk setting. Times are summed per track from prepared features through each encoder and ASR decoder; they exclude source-paced waiting and are not a GUI or live-latency benchmark.

| Source | Clips | Audio | Vendor compute | Staged compute | WER on these clips |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spanish | 4 | 43.38 s | 51.95 s | 50.93 s | 2.15% (2 / 93 words) |
| French | 4 | 36.84 s | 43.78 s | 43.18 s | 8.33% (8 / 96 words) |

On both paths, all eight clips produced identical token IDs and frame advances. **Each path** made 255 encoder forwards, with 247 verified reuses of the previous attention and convolution state. The [redacted full receipt](../research/unified_models/nemotron_cache_smoke_report.json) records the clip aggregates and flags no translation head, no training, and no live latency measurement. Vendor compute took **95.73 seconds for 80.22 seconds of audio** (1.19 seconds compute per audio second); all eight vendor runs were slower than their clip durations, even with precomputed features. The staged path also did not keep up. Whole-process peak RSS was approximately 2.98 GiB for both passes in one process; this does not isolate each path's memory or demonstrate sustained stability. No formal WER comparison to another runtime, translation quality percentage, or production performance claim follows.

Next recommendation: compare a pinned CPU-optimized runtime on the same clips and track the ASR quality/speed tradeoff before spending compute fitting a translation head. The measured FP32 configuration should not be treated as a viable real-time default. Keep the existing app unchanged.
