# BridgeCaption

**A visible, movable AI translation overlay for Linux meetings, videos and lectures.** Built for the **2026 Veritas AI International High School Hackathon — AI + Languages** track.

BridgeCaption opens an always-on-top caption window with normal Linux window decorations and minimize controls. Select a **System audio** monitor to translate what is playing on your computer, or a microphone to translate nearby speech. The app transcribes speech locally with a pretrained multilingual Whisper model, then sends **transcribed text** (not raw audio) to Google's Gemini translation model with short-term conversation context, glossary terms and contextual slang hints. It displays the original language and translated captions side by side.

This is an early working-code prototype, **not a verified production service**. Actual live caption quality depends on the user's machine, Gemini quota, network and source language. Translations process approximately four-second chunks of audio, so there will be a delay; it is not simultaneous word-by-word interpretation. Capture, recognition and translation run independently with bounded queues; if the translation service is too slow, older queued chunks are skipped and the app reports this instead of silently accumulating minutes of latency.

## Quick start on Ubuntu / Linux

Prerequisites: Python 3.10+, FFmpeg, PulseAudio-compatible PipeWire or PulseAudio, an audio output or microphone, and a Gemini Developer API key.

~~~bash
git clone --branch feat/bridgecaption-hackathon --single-branch https://github.com/Helloworld9099/veritastester.git
cd veritastester
sudo apt install ffmpeg pulseaudio-utils python3-tk python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,eval]'
cp .env.example .env
# Open .env in an editor and set GEMINI_API_KEY
bridgecaption
~~~

Get a key at https://aistudio.google.com/apikey. The default translation model is gemini-3.5-flash-lite, and the model setting is configurable. If a model ID is unavailable for your account, change GEMINI_MODEL in .env to an available text-generation model. The first speech-model run downloads weights to the local machine.

Use **System audio (current output)** for meetings, browser videos and other playback, or a microphone. Read and accept the visible disclosure about transmitting transcript text to Gemini, then click **START LISTENING** to capture speech. Click **STOP LISTENING** to end capture. Drag the top header to move the window; PINNED toggles always-on-top mode.

**To try without audio**, select **Spanish** rather than Auto detect, type "Ese código está chido" and click TRANSLATE TEXT. Open GLOSSARY and enter "chido = cool", or select a specific meeting topic. Toggle Context + slang to compare context-enabled behavior.

If output devices are missing, run `pactl list short sources`. A device whose name ends in `.monitor` generally captures output playback. On some Wayland compositors, always-on-top behavior may be restricted. If CPU recognition is too slow, set WHISPER_MODEL=tiny in .env, which can reduce accuracy. To change the listening interval, set AUDIO_CHUNK_SECONDS to between 1 and 12.

## Language coverage

The UI offers a broad language selector, including Spanish, French, Mandarin Chinese, Swahili, Yoruba and other languages. **It does not yet include every language supported by Google Translate, and not every listed language or translation pair has been empirically validated.** The speech model and translation model have different language coverage. The target selector accepts any listed language but successful output depends on the chosen model's actual capabilities.

## Core AI design

Audio from selected device -> **Faster-Whisper ASR** -> transcript and language -> **Gemini translation** -> original and translated captions.

Each listening session has independent, memory-only context that is cleared when the session ends. The translation request can use the last four preceding utterances, a chosen topic, a small term glossary, and conditional slang-meaning hints. Hints are not unconditional replacements: slang can have literal meanings too. Prior transcript text is treated as untrusted data and cannot legitimately instruct the AI.

An older, unvalidated **small supervised Naive Bayes topic classifier** is retained for demonstration, but is disabled in the live translation pipeline. Train it from scratch using original synthetic labels, but do not confuse it with training Whisper or Gemini:

~~~bash
bridgecaption-train --data data/topic_train.csv --out data/topic_model.json --metrics data/topic_metrics.json
~~~

The model may perform poorly on real meetings and can be overridden with an explicit topic. It is optional; without trained artifacts, the app falls back to General. The central intelligence is the speech and translation models, which are pretrained and **not fine-tuned in this repository**.

## Foundation checks and future memory training

Run `bridgecaption-preflight` from the repository root after installing the test dependencies. It runs compile and offline unit tests, checks whether desktop/audio dependencies are available, and writes a local report to `results/foundation_preflight.json`. It **cannot** establish real microphone, system-audio, Gemini API, language coverage or desktop usability. Those require testing on your own workstation.

The research hypotheses in the provided document concern two-view representation learning, predicting whether old experiences are useful, and avoiding harmful transfer. BridgeCaption adapts them cautiously to translation: an opt-in supervised predictor can **abstain** from reusing earlier dialogue when that dialogue appears likely to distort a new translation. The experimental code is **not JEPA, reinforcement learning or fine-tuning of Whisper/Gemini**, and is not yet connected to the live application.

After collecting independent permitted and human-reviewed conversation examples:

~~~bash
# Validates JSONL schema and grouped train/dev/test splits. Does NOT train.
bridgecaption-train-memory --data data/private/annotated_context.jsonl

# Only after all readiness gates have been verified, opt in to training:
bridgecaption-train-memory --data data/private/annotated_context.jsonl --train --confirm-live-verified
~~~

Read the [training hypotheses, dataset schema, experimental baselines, and pre-training acceptance gates](docs/TRAINING_READINESS.md). No real model training or hackathon accuracy benchmark is claimed in the current revision.

## Test and evaluate

~~~bash
python -m pytest -q
# Focused session/normalization tests can run without model weights or an API key.
# Requires GEMINI_API_KEY, actually calls the remote model:
bridgecaption-eval --data data/context_demo.jsonl --out results/demo_results.csv
# Requires authorized held-out audio clips and a CSV manifest:
python -m bridgecaption.asr_eval --manifest /path/to/manifest.csv
~~~

The synthetic contextual examples are for testing software mechanics, not a validated competitive benchmark. For genuine results, use **FLEURS** and **Common Voice** for multilingual/accent recognition, **FLORES+** for parallel text translation, and optionally **CoVoST 2** for speech translation. See [Dataset Guide](docs/DATASETS.md).

Report **actual WER for each language**, translation chrF by language, latency, slang/term accuracy on independent human references, examples of errors and comparisons against the same model without context. No real audio, Gemini, or translation-performance results have been collected by this repository's authoring environment yet. Small synthetic topic-training examples are included, but the training step is deliberately pending. See [Pre-training implementation status](docs/PRETRAINING_STATUS.md).

## Privacy

Capture starts **only** after an explicit click and consent notice. There is no hidden/stealth recording or saving of audio files by BridgeCaption. Microphone/system audio stays local for recognition. **Text from the conversation, recent context and terminology are sent to the Gemini API**, subject to Google's data-handling policies and quota. Do not use confidential meetings or record other people without permission. The overlay is visible, not designed to evade screensharing or monitoring.

.env, recordings and local evaluation results are ignored by git. No secrets should be committed.

## Hackathon integrity

Pretrained speech and translation models, public datasets and AI-assisted coding are permitted by the event rules, provided they are credited and the team understands its work. This application was generated with extensive assistance from ChatGPT GPT-6 during the hackathon. The work does not claim to train a new multilingual foundation model. The trained topic classifier is a limited supplemental experiment. See [Hackathon checklist](docs/HACKATHON.md) for evaluation and disclosure requirements.

## Sources and credits

- [OpenAI Whisper](https://github.com/openai/whisper) — pretrained multilingual speech recognition
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — optimized local inference
- [Gemini Developer API](https://ai.google.dev/gemini-api/docs) — cloud translation; model and quotas are configuration-dependent
- [FFmpeg](https://ffmpeg.org/) — local audio capture
- [FLEURS](https://huggingface.co/datasets/google/fleurs), [Common Voice](https://commonvoice.mozilla.org/en/datasets), [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus), [CoVoST 2](https://github.com/facebookresearch/covost) — optional external evaluation datasets, not included

MIT license applies to original project code. Respect separately licensed libraries, models and datasets.
