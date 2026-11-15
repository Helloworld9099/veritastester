# Pre-training checkpoint — November 14, 2026

The BridgeCaption prototype from PR #1 **has been merged to main** (merge commit 854212b3c3df3b89e25b11e3e17c7fe3e7ef3646). This document tracks the subsequent foundation hardening on the fix/translation-foundation-and-training-gates branch.

## Current implementation

- Visible, movable Linux Tkinter caption window with system-audio and microphone source discovery.
- Continuous audio capture, independent Whisper and Gemini workers, bounded queues, and visible backlog counters.
- Each listening session has isolated short-term context. Context reset now discards queued older utterances and suppresses in-flight captions from the prior context generation.
- The UI clears displayed captions on Stop and Clear Context; no raw meeting audio is intentionally stored.
- Trusted translation rules use the Gemini system-instruction field while transcripts and glossaries are serialized as untrusted data.
- Bounded glossary entries, optional translation timeout, and free local same-language transcription/captions.
- The previously experimental and poorly performing synthetic topic classifier **does not influence live translations**. Explicit user-selected topics and Gemini context remain.
- Opt-in offline utility-predictor training scaffold with grouped conversation/speaker splits, independent labels, negative-transfer accounting and an abstention threshold.

## Verification and constraints

- Local isolated Python tests for the new offline research scaffold and provider/pipeline behaviors passed using synthetic fixtures; those checks are **not** a full checkout or end-to-end run of this GitHub revision.
- Run `bridgecaption-preflight` from a **fresh checkout on the user's own Linux workstation**. It executes compile + pytest and reports dependencies. It deliberately does not imply that real audio capture or Google API calls were verified.
- Real audio-device interactions, Gemini API credential access, desktop UX, multilingual speech recognition quality, and sustained latency remain unverified.
- No real model has been trained or fine-tuned. The supervised research predictor is separate from the live app and requires verified data and opt-in `--train`.

## Before training or submitting

1. Run all offline tests/compilation successfully and test a real live audio session with start/stop, errors and session privacy isolation.
2. Measure recognition and contextual translation separately on held-out permitted examples, with at least two languages and some difficult slang/dialect cases.
3. Obtain independent human references and judge whether candidate memories truly help; avoid examples used to write prompts or glossaries.
4. Follow [training-readiness gates](TRAINING_READINESS.md), including fixed-model four-arm ablation tests and total compute/latency accounting.
5. Verify GitHub code access for judges and accurately disclose the use of pretrained models, AI coding and all unverified claims.

**Training readiness: NOT YET VERIFIED.** There is no justified basis to claim the translator is bug-free or meets accuracy/latency targets until the device/API integration tests are performed.
