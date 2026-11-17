# BridgeCaption

A **visible, always-on-top Linux desktop translation overlay** for videos, online meetings and lectures. Built for the **2026 Veritas AI Hackathon — AI + Languages** track.

**Status:** Prototype code with offline tests; no claim of production readiness or verified multilingual accuracy. User-approved audio sources are processed by local Faster-Whisper. A separate provider translates text, with context selection designed to avoid misleading previous statements. Audio is not deliberately saved.

## For judges

**[Judge guide: demo, verified results, limits and disclosures](docs/JUDGE_GUIDE.md)** · [Full experiment report](docs/OFFLINE_RESEARCH_RESULTS.md) · [Dataset sources and licences](docs/DATASET_SOURCE_INVENTORY.md)

In an offline English idiom-classification experiment on **2,301 held-out examples**, a majority-class baseline scored **0.4521 macro-F1**, a sentence-only classifier **0.6938**, and a context-added classifier **0.6878**. Additional preceding text did **not** demonstrate a reliable benefit. These scores **are not** claims about translation accuracy, speech recognition or real-time caption performance. The trained idiom classifier is not an activated translation model.

The [tests](tests/), [reproduction steps](docs/REPRODUCE_OFFLINE_RESEARCH.md) and [inconclusive/negative findings](docs/SMALL_SCALE_CONTEXT_PILOT.md) remain available for inspection.

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
git clone --branch main \
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

**Azure (cloud baseline):** Set `BRIDGE_TRANSLATOR=azure`, `AZURE_TRANSLATOR_KEY` and, where needed, `AZURE_TRANSLATOR_REGION`. Select **Translator F0** in the portal for its free character allotment. See [Azure key setup](docs/AZURE_SETUP.md) for the exact resource, pricing tier and configuration. The standard Azure endpoint receives only the current transcript and **does not support our contextual prompts or glossary**.

Run:

~~~bash
bridgecaption
~~~

The app starts in a **compact, floating, lightly transparent caption view** (92% opacity). **START/STOP** and **SETTINGS** remain visible. Select **SETTINGS** (or press Ctrl+,) to expand the full controls, change the microphone/system monitor, choose your translation engine, edit your glossary, test typed text, or change transparency with the 82–100% slider. Hide settings to return to the compact overlay. The window remains draggable and pinnable; transparency is compositor-dependent on Linux/Wayland. This is a normal Tkinter desktop window, not click-through.

Settings group the labeled audio and language selectors, context tools, and translation engine. The summary shows Ready/Listening state; the engine note identifies local versus cloud text translation. **Full caption** opens a scrollable, read-only snapshot of the complete last result without another translation request. Clearing context, changing engines, stopping capture, or replacing the current caption with new speech closes the snapshot. See [UI review and verification](docs/UI_REVIEW.md) for desktop checks and current limits.

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

## Testing the translator and measuring actual speech clips

You reported that the offline tests passed on Ubuntu. Next, test the selected provider using **public synthetic text**, without exposing credentials or conversation content:

~~~bash
# No network request
bridgecaption-check --engine azure
# One real, synthetic Spanish-to-English request using your private .env key
bridgecaption-check --engine azure --execute
~~~

To evaluate real speech, prepare a CSV in the same directory as your own permitted audio clips (file,language,transcript; optionally translation,previous_json,topic). The source clip path must stay inside that CSV's folder. A sample with genuine reference transcripts and translations can quantify WER/CER and chrF; unverified synthetic references cannot establish accuracy.

~~~bash
# Validate manifest only
bridgecaption-eval-audio --manifest /path/to/examples.csv
# Actually decode audio and run local Whisper ASR; no provider request
bridgecaption-eval-audio --manifest /path/to/examples.csv --execute
# Include Azure translation, only with participant permission
bridgecaption-eval-audio --manifest /path/to/examples.csv --execute \
  --translate --engine azure --allow-cloud-transcripts
~~~

The output is **aggregate-only** (per-language ASR WER/CER, clip inference timings and optional translation chrF), and goes into ignored results/. It does not store audio or full recognized transcripts. Clip replay is **not** a test of multi-minute live performance. See [Azure setup](docs/AZURE_SETUP.md) and [training readiness](docs/TRAINING_READINESS.md).

## Prepare public research corpora (no automatic training)

BridgeCaption now includes **offline adapters for MAGPIE, SemEval-2022 Task 2, SlangInterp, Multi3WOZ, Bangor Miami (normalized CHAT export), DiaBLa, NaijaSenti and Okwu**. Their data roles differ: idiomaticity is not an exact meaning, sentiment is not slang interpretation, and a prior utterance is not automatically a helpful memory. The import process preserves source identifiers, local file fingerprints, train/dev/test boundaries and attribution metadata.

Start with the [step-by-step eight-dataset preparation guide](docs/CORPUS_PREPARATION.md). It includes the real source field mappings, license restrictions, commands, source holdout rules and the optional two-view idiomaticity classifier. All import tools dry-run by default; private outputs require explicit `--write`.

~~~bash
bridgecaption-prep-corpus --help
bridgecaption-corpus-curriculum --help
bridgecaption-corpus-review --help
bridgecaption-train-idiom --help
~~~

The curriculum tool can create human-review candidates for **helpful/neutral/harmful memory utility** and **slang meanings**, plus separately labelled same-dialogue versus cross-dialogue adjacency pairs for future self-supervised pretraining. No automatic labels of translation usefulness or slang meaning are fabricated. The human review CLI requires a grounded second situation description and a verified reference meaning/translation.

**No raw third-party corpora, private review queues or training weights are committed here.** A separate auxiliary English idiomaticity classifier was trained offline, with actual metrics and limitations recorded in [the research report](docs/OFFLINE_RESEARCH_RESULTS.md); it is not enabled in the live app. Some sources (notably SlangInterp/Urban Dictionary) are restricted to research use, and DiaBLa was held out for evaluation only.

## Collecting real context and slang labels

The models must train from **independently reviewed examples**, not from invented AI labels. We added a local-only annotation command for this purpose. Prepare a CSV queue using permitted conversations; include speaker/conversation family IDs so the train/test split cannot reuse the same person or meeting.

For **memory-usefulness labels**, the CSV header is:

~~~text
case_id,group_id,language,utterance,previous,situation,candidate_memory,reference_translation,provenance
~~~

For **slang-sense labels**, use:

~~~text
case_id,group_id,language,expression,utterance,previous,situation,provenance
~~~

Reviewers see the speech and grounded situation **locally**, and provide a label interactively. The CLI never generates its own ground-truth label, sends the text to an API, or records audio. 'q' stops without guessing. Outputs are written as private JSONL with owner-only permissions on POSIX systems.

~~~bash
bridgecaption-annotate --task memory \
  --queue data/private/memory_queue.csv --out data/private/memory_labels.jsonl

bridgecaption-annotate --task sense \
  --queue data/private/sense_queue.csv --out data/private/sense_labels.jsonl

# Continue an interrupted annotation without duplicating labelled cases:
bridgecaption-annotate --task sense \
  --queue data/private/sense_queue.csv --out data/private/sense_labels.jsonl --append
~~~

Memory labels are **helpful**, **neutral**, or **harmful**; sense labels are concise meanings such as *excellent* versus *ill*. Both require actual independent human review. The output under data/private/ is ignored by Git. See [training readiness](docs/TRAINING_READINESS.md) before fitting either model. No verified training dataset has yet been produced in this repository.

## Security, privacy, limitations

Raw audio is decoded locally; chosen text and selected prior context are sent to the **explicitly selected** cloud provider, or only to local Ollama. Do not use confidential material or record meeting participants without authorization. Recordings and API credentials must not be committed; `.env`, `data/private/`, and `results/` are ignored by Git. For a complete data-policy assessment, consult the selected provider's terms.

The app can lag, skip audio under load, misrecognize accents or speakers, and generate incorrect translations. No certified real-time guarantee, complete Google language parity, model fine-tuning, or sustained Ubuntu field test is claimed here. Running Ollama and Whisper on CPU can be considerably slower than cloud translation.

### Attribution

[OpenAI Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Ollama](https://docs.ollama.com/api), [Groq](https://console.groq.com/docs/openai), [Gemini](https://ai.google.dev/gemini-api/docs), [Azure Translator](https://learn.microsoft.com/en-us/azure/ai-services/translator/text-translation/reference/v3/reference), and [FFmpeg](https://ffmpeg.org/) are third-party dependencies or services. Major code and documentation have been developed with ChatGPT assistance; disclose this in the competition submission. Original project code is MIT licensed. See the [judge-facing disclosures](docs/JUDGE_GUIDE.md) and [dataset licence inventory](docs/DATASET_SOURCE_INVENTORY.md).
