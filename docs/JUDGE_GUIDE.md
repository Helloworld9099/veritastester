# BridgeCaption — judge guide

**Track:** 2026 Veritas AI Hackathon — AI + Languages  
**Repository:** [BridgeCaption](https://github.com/Helloworld9099/veritastester)  
**Status:** Linux prototype with measured offline research; live multilingual translation quality and sustained caption latency remain unverified.

## Pitch

BridgeCaption is a real-time AI translation overlay that uses conversation context to help interpret slang and idioms, making discussions across languages easier to follow.

## What the project does

A floating Linux window can capture authorized microphone or system audio, transcribe speech locally with faster-whisper, and show original text next to translations from an **explicitly selected** engine: local Ollama, Groq, Gemini, or Azure Translator. It offers a glossary, conversation history, a text-only test, and context clearing. It does **not** silently switch to a cloud provider.

The live context selector currently uses a conservative **untrained heuristic** to omit unrelated earlier statements. Optional memory-usefulness and slang-sense trainers require reviewed labels; neither is currently established as a trained product feature. The offline idiom classifier described below is a **separate research experiment**, not fine-tuning Whisper or a translation model.

```text
Authorized mic/system audio -> faster-whisper (local)
      -> current transcript -> conservative context selection
      -> explicitly chosen translator -> original + translated captions
```

*Azure's standard text endpoint is a context-free baseline and does not receive optional dialogue prompts.*

## Two-minute demo

1. Start `bridgecaption` and show the floating window. Select a microphone or PulseAudio/PipeWire audio monitor, source and target languages, and a provider.
2. For a safe text-only demo, enter `Ese código está chido` with Spanish as source and press **TRANSLATE TEXT**. This still requires a configured working translation provider.
3. If authorized test audio and a working provider are available, show **START LISTENING**, bilingual captions, glossary editing, **CLEAR CONTEXT** and **STOP LISTENING**.
4. Explain the trained idiom-classification result below and its limits. A text-only demo is **not** evidence of live microphone or subtitle accuracy.

See the [Ubuntu installation and provider instructions](../README.md#ubuntu-installation).

## Measured results

**Actual task:** distinguish English **idiomatic versus literal** usage, using cleaned MAGPIE examples and held-out idiom types. A hashed character 3–4-gram logistic classifier was trained on **5,404** examples, developed on **1,064**, and assessed on **2,301 held-out** examples.

| Approach | Test macro-F1 | Balanced accuracy |
| --- | ---: | ---: |
| Majority-class baseline | 0.4521 | 0.5000 |
| Sentence only | **0.6938** | **0.7216** |
| Sentence plus preceding context | 0.6878 | 0.7164 |

The sentence-only model improved macro-F1 by **0.2417 absolute points** over the baseline. Context changed macro-F1 by **−0.0060** versus sentence-only, with a grouped bootstrap 95% confidence interval of **[−0.0316, 0.0237]**. This does **not** establish that adding context helps.

The research checkout recorded **138 passed, 1 skipped** offline tests (the skip required a graphical display). This is a **historical test receipt**, not a claim that current branches, providers or the running GUI have been tested.

The team prepared **1,600** memory-usefulness and **1,000** contextual-sense candidates for independent review, with **zero verified product-task human labels** at that checkpoint. Those product models were **not trained or activated**.

**Not measured by this experiment:** actual translation quality, speech-recognition WER, accent/code-switch coverage, real-time lag, or an improvement in caption translations. Do not present these classifier scores as those results.

For methods, exclusions, confusion matrices, timing and uncertainty see [full offline results](OFFLINE_RESEARCH_RESULTS.md). See [reproduction instructions](REPRODUCE_OFFLINE_RESEARCH.md) for the source-based experiment. The [small SemEval context pilot](SMALL_SCALE_CONTEXT_PILOT.md) is retained, including its negative/inconclusive findings.

## Data, privacy and attribution

The training research used public research sources and synthetic examples. No private meeting recordings or API credentials are committed. Audio is decoded locally, while text is sent only to the selected cloud provider (or stays with local Ollama). Record only with participants' permission and follow the provider's data policy.

**Datasets and papers:** MAGPIE (the scored English experiment); Multi3WOZ, SemEval-2022 and DiaBLa (other prepared/held-out sources). Specific roles, copyrights, licences, citations, and unused candidate sources are recorded in [dataset source inventory](DATASET_SOURCE_INVENTORY.md). Public availability is not permission to redistribute everything.

**Pretrained models, tools, reference documentation:** OpenAI Whisper and faster-whisper; optional Ollama (README example: `qwen3:4b`), Groq, Gemini and Microsoft Azure Translator; scikit-learn, Python, FFmpeg and dependencies in [pyproject.toml](../pyproject.toml). Official platform/model documentation is linked in [README](../README.md), [Azure setup](AZURE_SETUP.md), and the research preparation guides. Supported provider options were **not** used to calculate the idiom-classification scores.

**AI assistance:** ChatGPT (GPT-6) and Codex contributed to coding, tests, research utilities and documentation. Additional tutorials, models or assets used by the human team must be added to the final attribution if applicable.

**Team integrity declarations:** The team states that every dataset, pretrained model, tutorial and AI tool used is credited; only public or synthetic data was used and no one's personal information was collected without permission; all reported results are actual measurements; and everyone named on the submission contributed. The submitting team must confirm names and the final complete credit list.

## Further reading

- [Code, installation, and run commands](../README.md)
- [Full observed results and limitations](OFFLINE_RESEARCH_RESULTS.md)
- [Dataset provenance and rights](DATASET_SOURCE_INVENTORY.md)
- [Offline research reproduction](REPRODUCE_OFFLINE_RESEARCH.md)
- [Training readiness and proposed four-arm evaluation](TRAINING_READINESS.md)
- [Regression tests](../tests/) and [synthetic examples](../data/context_demo.jsonl)
