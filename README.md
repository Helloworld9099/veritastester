# BridgeCaption

A **visible, always-on-top Linux desktop translation overlay** for videos, online meetings and lectures. Built for the **2026 Veritas AI Hackathon — AI + Languages** track.

**Status:** Prototype code with offline tests; no claim of production readiness or verified multilingual accuracy. User-approved audio sources are processed by local Faster-Whisper. A separate provider translates text, with context selection designed to avoid misleading previous statements. Audio is not deliberately saved.

## What it does

- Select microphone or PulseAudio/PipeWire **system audio monitor**; read subtitles over other desktop windows.
- Translate with **local Ollama (no API key)**, Groq, Gemini or Microsoft Azure Translator. Provider switching is explicit: **no silent cloud fallbacks**.
- Use an in-memory recent-conversation buffer, topical hints, slang ambiguity hints and a user glossary. A context selector can **abstain** from passing unrelated older text. This is a conservative untrained heuristic until independently trained predictor weights are supplied.
- Display original and translated text and report processing lag/skipped audio chunks. Three independent workers capture audio, transcribe it and translate it.
- Clear private context between sessions; Clear Context discards old queued audio/captions. The UI requests disclosure/consent about the selected transcript destination.
- Test baseline vs. always-on context vs. learned one-view and two-view memory selection without mixing speaker/conversation groups.

**Language coverage is not equivalent to Google Translate's entire list.** Listed speech and translation languages vary by model; Swahili, Yoruba, accented speech and slang need actual tests.

## Ubuntu installation

~~~bash
git clone --branch fix/translation-foundation-and-training-gates \
  https://github.com/Helloworld9099/veritastester.git
cd veritastester
sudo apt update
sudo apt install -y ffmpeg pulseaudio-utils python3-tk python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,eval]'
cp .env.example .env
~~~

The clone may require GitHub login because this repository is private.

### Select the translation engine

**Local: Ollama (no provider key).** Install Ollama using its official documentation, start the Ollama service, and run `ollama pull qwen3:4b`. Keep the service running. With `BRIDGE_TRANSLATOR=ollama` in `.env`, the recognized speech is translated locally at `127.0.0.1:11434`. CPU performance may be slow: this is a privacy/no-quota option, not a speed guarantee.

**Groq (cloud):** Set `BRIDGE_TRANSLATOR=groq` and `GROQ_API_KEY`; the default model is `openai/gpt-oss-20b`. This supports instructions and selected context, but free-tier quotas still apply.

**Gemini (cloud):** Set `BRIDGE_TRANSLATOR=gemini` and `GEMINI_API_KEY`. Model defaults to `gemini-3.5-flash-lite`; verify availability for your account.

**Azure (cloud baseline):** Set `BRIDGE_TRANSLATOR=azure`, `AZURE_TRANSLATOR_KEY` and, where needed, `AZURE_TRANSLATOR_REGION`. This endpoint only receives the current transcript. It **does not support our contextual prompts or glossary** and should be treated as a conventional translation baseline.

Run:

~~~bash
bridgecaption
~~~

The **Translation Engine** control inside the app changes providers; you can also choose a provider in `.env`. The privacy dialog identifies the current destination. The app does not send text to other translation providers if one fails.

To test translation without audio, select Spanish as source, type *Ese código está chido*, and click **TRANSLATE TEXT**. Click **CHECK ENGINE** to try a public sample sentence without accessing your meeting. With system audio, choose a monitor such as **System audio (current output)**, then click **START LISTENING**. Consent and participant authorization are required. Use **STOP LISTENING** when done.

The first speech recognition run downloads Whisper model weights. `WHISPER_MODEL=tiny` may reduce CPU load but usually hurts accuracy. Set `AUDIO_CHUNK_SECONDS` between 1 and 12 (default 4) and `TRANSLATION_TIMEOUT_SECONDS` within 3–30 seconds. Results depend on hardware, provider and model.

## AI and context architecture

~~~text
Authorized audio -> faster-whisper (local speech recognition)
                         |
                         v
                Current transcript
                         |
               Context intelligence
             (select memories or abstain)
                    /         \
       optional trained        untrained conservative
       utility predictor       context heuristic
                    \         /
                         |
              Model instructions + data
                         |
              Explicit translator provider
             Ollama | Groq | Gemini | Azure*
                         |
               Original + translated captions

* Azure does not receive optional dialogue context.
~~~

The local context selector uses relevant prior utterances, potential slang and manually provided meeting topics. The **supervised memory-utility predictor** is a small trainable baseline using eight shared lexical features. Its training experiments follow the uploaded two-view/transfer-value hypothesis **by analogy**, not as a reinforcement-learning or JEPA implementation. The predictor does **not** generate meaning from nothing; the translation language model still performs contextual language interpretation. The predictor's only learned task is which earlier memory might help or harm a translation.

No reliable pretrained local memory model or trained semantic encoder is bundled. Before enabling a trained artifact, it must outperform no-context and always-on-context baselines on independently reviewed, held-out data; see [training design and readiness gates](docs/TRAINING_READINESS.md).

## Optional supervised slang-sense training

A separate CPU-friendly **character n-gram classifier** can learn which labelled meaning of a slang expression is appropriate given earlier speech and a grounded meeting description. It trains with conversation-group splits and abstains when predicted confidence is too low. This is **not a pretrained multilingual semantic encoder**: without a validated dataset it supplies **no learned hints**. The translation model is never forced to use a predicted sense.

Annotated JSONL records contain `case_id`, `group_id`, `language`, `expression`, `utterance`, `previous`, `situation`, `meaning`, and `provenance`. For an ambiguous expression such as *sick*, independent reviewers should label both the medical and informal-positive meanings in different conversations. Keep all examples from the same speaker/conversation in one split.

~~~bash
# Validate syntax/groups without training:
bridgecaption-train-sense --data data/private/sense_labels.jsonl

# Only after code, real-audio, API and privacy verification:
bridgecaption-train-sense --data data/private/sense_labels.jsonl \
  --train --confirm-live-verified

# Explicitly enable a separately validated artifact:
# BRIDGE_SENSE_MODEL=results/sense_model.json
~~~

The model stores character n-gram statistics in a local JSON artifact. Since these features can encode fragments of private speech, treat the artifact as **confidential** and do not commit it. Compare the learned model with no-sense and literal-context baselines before enabling it for real users. See [training readiness](docs/TRAINING_READINESS.md).

## Tests and foundation preflight

~~~bash
python -m compileall -q src tests
python -m pytest -q
bridgecaption-preflight
~~~

Preflight reports offline compilation, dependencies and tests only. **It does not verify live audio, provider responses, model quality, participant consent or UI behavior.**

For independent, permissioned contextual datasets:

~~~bash
# Validate data, do not train:
bridgecaption-train-memory --data data/private/annotated_context.jsonl

# After successful preflight + real audio/provider tests, explicit opt-in:
bridgecaption-train-memory --data data/private/annotated_context.jsonl \
  --train --confirm-live-verified

# Experimental model is never automatically activated:
# BRIDGE_CONTEXT_MODEL=results/memory_research.json

# Compare all four methods on test groups without making API calls:
bridgecaption-eval-context --data data/private/annotated_context.jsonl \
  --artifact results/memory_research.json

# Only after confirming account quota and data sharing permissions:
bridgecaption-eval-context --data data/private/annotated_context.jsonl \
  --artifact results/memory_research.json --engine ollama --execute
~~~

The four-arm comparison uses the **same translation engine** in every condition, and reports per-language chrF and request latency. It requires real fluent reference translations and speaker/conversation-disjoint splits. Synthetic examples and unit tests are **not evidence of real-world accuracy**.

ASR can be evaluated on held-out public or consented clips via `python -m bridgecaption.asr_eval --manifest /path/to/manifest.csv`. See [dataset guidance](docs/DATASETS.md) for FLEURS, Common Voice, FLORES+, and CoVoST 2.

## Security, privacy, limitations

Raw audio is decoded locally; chosen text and selected prior context are sent to the **explicitly selected** cloud provider, or only to local Ollama. Do not use confidential material or record meeting participants without authorization. Recordings and API credentials must not be committed; `.env`, `data/private/`, and `results/` are ignored by Git. For a complete data-policy assessment, consult the selected provider's terms.

The app can lag, skip audio under load, misrecognize accents or speakers, and generate incorrect translations. No certified real-time guarantee, complete Google language parity, model fine-tuning, or sustained Ubuntu field test is claimed here. Running Ollama and Whisper on CPU can be considerably slower than cloud translation.

### Attribution

[OpenAI Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Ollama](https://docs.ollama.com/api), [Groq](https://console.groq.com/docs/openai), [Gemini](https://ai.google.dev/gemini-api/docs), [Azure Translator](https://learn.microsoft.com/en-us/azure/ai-services/translator/text-translation/reference/v3/reference), and [FFmpeg](https://ffmpeg.org/) are third-party dependencies or services. Major code and documentation have been developed with ChatGPT assistance; disclose this in the competition submission. Original project code is MIT licensed.
