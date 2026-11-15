# Pre-training checkpoint — November 14, 2026

Implementation branch: `feat/bridgecaption-hackathon` (PR #1). The `main` branch intentionally still holds only an initial README and has NOT been merged.

## What is in this branch

- Linux Tkinter overlay with window-manager controls, draggable title row and always-on-top toggle.
- PulseAudio/PipeWire microphone and playback-monitor enumeration using `pactl` and FFmpeg.
- Continuous audio capture, distinct ASR and translation workers, bounded queues, timestamped chunks and visible lag/drop counters (`session.py`).
- Local faster-whisper inference with model caching, language validation and configurable audio chunk duration.
- Context-sensitive Gemini translation, conversation history, glossary and slang hints. Latin-language slang uses word boundaries to avoid matching fragments of unrelated words.
- A consent notice before sending any recognized transcript, glossary and context to the Gemini API. Audio itself remains local. Each listening session has independent context and explicitly clears it on finish.
- Text-only translation demo, baseline-vs-context evaluation command, per-language WER/CER evaluation command, synthetic training CSV for an optional future topic model.

## Testing evidence

Focused local tests on the streaming controller and a matching subset of language-normalization functions: **9 passed** with faked audio/ASR/translations and no network/API access.

A Tk 8.6 Linux/Xvfb check confirms a normally decorated window can be minimized, unlike the old override-redirect implementation.

**Not verified:** a complete install and test run of this exact GitHub revision; real GTK/Tk desktop usability; FFmpeg capture from an actual meeting; downloaded Whisper weights; authenticated Gemini calls; real-world latency/quality and language coverage.

The full repository has more tests. Before any merge/submission, run `python -m pytest -q` from a fresh checkout and fix any failures.

## Next phase: training and honest evaluation (deliberately not started)

1. Collect authorized/public speech with held-out reference transcripts using FLEURS and optionally Common Voice. Avoid consent/privacy violations.
2. Prepare independent context and slang samples with native-speaker references. Do not evaluate against the very examples used to write glossary hints.
3. Run per-language WER/CER and translation chrF; compare the same Gemini model with/without context.
4. Only then decide whether improving a classifier or fine-tuning any model adds value. The synthetic topic classifier is NOT equivalent to training the translation model.
5. Include at least one failure, model/dataset acknowledgments and AI-assisted development attribution in the Veritas submission.

## Caveats and privacy

- The hard-coded language selection is **not** the full Google Translate language catalog. Some speech codes may have weaker recognition and Gemini translation performance; measure before advertising accurate support.
- Slow CPU or API processing can cause audio chunks to be dropped to keep near-live captions. The UI reports skipped chunks; this is not lossless transcription.
- Free-tier Gemini usage may involve provider data use for product improvement. The consent notice does not replace permission from meeting participants or an organization's privacy policy.
- Do not store API keys or real recordings in Git.
- No model was trained in this change. Existing training scripts and synthetic samples were retained for the following phase.
