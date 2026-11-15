# Training theories and foundation readiness

**Status: HOLD. Real model training has not started.** This is a research plan and an opt-in offline predictor prototype. Never treat the prototype's unit tests as proof of accurate live translation.

## 1. Scientific translation of the uploaded RL theories

The proposed research originally targeted **reinforcement-learning training speed** using two complementary descriptions of an experience and a predictor of beneficial old memories. That does **not** mean a DQN, MiniGrid agent, or JEPA is appropriate for speech translation.

Our analogous hypothesis: given a spoken utterance and relevant old dialogue, can a small predictor determine whether context will **improve** rather than **harm** an AI translation? A second, independently grounded description of the meeting may help the predictor avoid false associations. The current small predictor is supervised logistic regression over lexical similarities, **not** a JEPA or trained language encoder. We will not call it either.

This follows the uploaded research's logic (2x2 ablations; positive/negative transfer; matched model/data budgets; test-time abstention; real computational cost) while clearly separating a proposed analogy from established results.

## 2. Experimental arms: identical primary AI models

| Arm | Two-view context | Selective memory | Meaning |
| --- | --- | --- | --- |
| A | No | No | Gemini translates only the current transcript |
| B | Yes | No | Gemini sees the current transcript and a grounded description of conversational situation |
| C | No | Yes | A trained predictor decides whether to include a prior utterance |
| D | Yes | Yes | The situation view and learned memory selector are combined |

All four arms must use the **same fixed** Whisper and Gemini models, language settings, evaluation clips, source transcripts and decoding parameters. Use exactly the same source audio for end-to-end comparisons. Include a stronger practical baseline that always passes the last three relevant utterances to Gemini; a weaker baseline alone can inflate the apparent improvement.

Do **not** equate more prompt context or tokens with a better representation. Include the extra context tokens, training duration, inference time and missed audio chunks in cost comparisons.

## 3. Two grounded views

- **View 1:** the actual source-language transcript and permitted preceding utterances.
- **View 2:** a concise human-verified scene or meeting-topic description derived from preceding speech, not from the answer or held-out reference translation.
- **Memory candidate:** an older utterance, previous technical term or slang instance that may help interpretation.
- **Negative transfer:** a superficially similar memory with a different meaning. Examples include English "sick" as "ill" versus "excellent", Spanish "vale" as agreement or value, and coding terms used outside software discussions.
- **Abstention:** skip a memory unless the predictor's development-set-selected threshold says its expected value is positive. If uncertain, translate without it.

The source transcript, situation and glossary are untrusted; never treat content inside them as commands to the translation model.

## 4. Data sources and collection

Use FLEURS and licensed/appropriate Common Voice audio to measure multilingual ASR separately. Use FLORES+ for conventional **text** translation, not as evidence of conversational context improvement. Add your own independent, permissioned conversational/slang test cases with fluent-speaker references. No recordings of meeting participants without permission.

Each JSONL training row has:

~~~json
{
  "case_id": "demo-001",
  "group_id": "speaker-or-conversation-family-001",
  "language": "es",
  "utterance": "Ese codigo esta chido",
  "previous": "We were discussing the new app interface",
  "situation": "A peer reviewing software interface quality",
  "candidate_memory": "The other speaker complimented the design",
  "reference_translation": "That code is cool",
  "human_judgement": "helpful",
  "provenance": "synthetic demo; not eligible as a real evaluation result"
}
~~~

The example above is **illustrative only**; it is not an actual measured improvement. All reference translations and usefulness judgements must be produced or checked independently of the predictor and Gemini prompt-tuning process. The dataset must include **harmful** and **neutral** past memories, not only beneficial examples.

Use group_id to keep all utterances from a speaker or conversation family in the **same split**. Assign independent groups to train/dev/test with a fixed seed. For cross-accent generalization, hold out speakers and accents as well as conversation topics when feasible.

Keep identifying information, private meeting audio, and proprietary transcripts **outside Git**. Local private data under data/private/ is ignored by the repository.

## 5. Offline prediction model

The new `bridgecaption.research_training` module:

- checks missing fields, duplicate cases, permitted usefulness labels and minimum example counts;
- divides data by **conversation/speaker group**, never randomly by sentence;
- compares **single-view** versus **two-view** feature sets with the **same eight-parameter logistic predictor architecture** and the same number of training epochs;
- fits helpful-context probability from independently reviewed utility labels and evaluates harmful-memory selection;
- tunes the decision threshold on a **development split**, with higher penalty for negative transfer;
- reports held-out results **by language** and saves only summary weights, not private utterances;
- has an **abstain** option at a threshold of 1.0;
- does not send any audio or text to Gemini and does not alter the live translation model.

**This is a low-cost pilot for the predictor**, not a trained encoder. An actual multi-view contrastive or predictive embedding model is a separate, later hypothesis that should be justified by results.

### Commands

~~~bash
# First install test dependencies, configure the app and run the offline foundation gate.
python -m pip install -e '.[test,eval]'
bridgecaption-preflight

# Validate an independent, permissioned and annotated dataset.
bridgecaption-train-memory --data data/private/annotated_context.jsonl

# Only AFTER live end-to-end checks and dataset validation:
bridgecaption-train-memory --data data/private/annotated_context.jsonl --train --confirm-live-verified
~~~

**No training was executed on real user data during this implementation.** The last command is intentionally opt-in and must not be run until the acceptance gates below are met.

## 6. Required foundation gates before training

- [ ] Fresh checkout: package installation, `python -m compileall -q src tests`, and complete `python -m pytest -q` pass.
- [ ] Verified Linux desktop launch, window controls, minimize/restore, drag, always-on-top, and shutdown.
- [ ] Authorized microphone **and** system-output monitor produce actual transcripts.
- [ ] On a real clip, the chosen Gemini API model successfully translates; handle invalid key, rate limiting and offline mode.
- [ ] Five-minute continuous playback: measure median and p95 caption latency, skipped chunks, CPU/RAM and failure recovery.
- [ ] Stop/start a second meeting: old context, queued speech, displayed captions and glossary data do not leak between sessions.
- [ ] Test malicious spoken instructions: translated as untrusted content rather than followed.
- [ ] Per-language transcription WER/CER on held-out audio, including at least one low-resource language and accent tests where valid labels exist.
- [ ] Independently reviewed baseline translations, contextual test examples and negative-transfer examples collected.
- [ ] Use the same evaluation data and model configuration across every ablation. Verify no conversation/speaker is in more than one split.
- [ ] Match inference/training budgets where applicable; record total elapsed time, resource overhead and whether added context improves quality.
- [ ] Confirm third-party model and dataset licenses, participant permission, data privacy, AI-tool attribution and access for judges.

`bridgecaption-preflight` only verifies local files, dependencies and offline tests. **It deliberately reports foundation_fully_verified=false** until actual desktop, audio, API and evaluation checks are performed; it cannot magically test a user's microphone or API credential from another computer.

## 7. Acceptance metrics and honest reporting

Report per language, not only an overall average: ASR WER/CER, translation chrF or human-rated meaning preservation, technical-term and slang accuracy, harmful-memory selection rate, helpful-context rejection rate, median/p95 latency and missed chunks.

Also test: wrong scene descriptions, misleading similar slang, unexpectedly changing languages, low audio levels, absent API credentials, no network and an API call taking longer than the audio queue.

A result is **successful** only if the combined approach improves held-out translation quality without unacceptable added delay, hallucinations, privacy issues or training/inference cost. If it doesn't, keep the simpler baseline and report the negative result.

## 8. Current verification limitations

The training scaffold's isolated Python unit tests passed locally with generated synthetic data. They did **not** establish performance of the live translator. Main-repository end-to-end tests, real Gemini calls, downloaded speech model performance and multilingual held-out accuracy are still unverified at the time this document was written.

Do not claim a 100%-bug-free foundation. The appropriate statement is: no known high-priority blockers **after** the above critical tests pass, with clearly disclosed residual limitations.

## 9. Implementation status (updated November 15, 2026)

The live `Pipeline` now invokes `ContextSelector` before contextual translation. Without `BRIDGE_CONTEXT_MODEL`, selection uses **a conservative untrained heuristic**, which is not a trained semantic interpreter. With an explicitly configured, validated output artifact from `bridgecaption-train-memory`, the same eight-feature predictor structure is used for inference and training. This learned selector can choose up to two relevant earlier utterances or abstain; it does not independently generate missing facts, and the chosen translation engine still interprets slang.

`bridgecaption-eval-context` evaluates the four arms with identical source utterances and a single explicit provider, and groups test examples by conversation/speaker. It reports per-language chrF, model call latency, harmful-memory selection and helpful-memory misses. Cloud engines can incur rate limits or quota; API calls are off until `--execute` is used. Azure is only a context-free baseline and cannot be substituted into the four prompt-based experimental arms.

The UI has an explicit **Ollama / Groq / Azure / Gemini** provider selector and a safe CHECK ENGINE button. No silent cloud failover is permitted; Azure is marked context-free. Recognition remains local in all modes, though selected transcript text is sent to the explicitly chosen cloud provider.

**Release and research status: not verified.** Local component tests cover the new provider and selector boundaries using mocks, but the GitHub-hosted full test workflow has failed without any reported runner steps, and actual desktop, microphone/system-audio, provider-key, model-availability and held-out spoken-translation tests are outstanding. No real supervised training of context weights has occurred in this project revision.

## 10. Supervised slang-sense classifier (additional opt-in component)

`src/bridgecaption/sense_model.py` implements a **character n-gram Naive Bayes** baseline to classify the meaning of an ambiguous phrase using earlier turns and a grounded situation description. It is not a JEPA, multilingual sentence embedding or foundation model. It can abstain when no valid meaning has sufficiently high dev-calibrated probability. The provider prompt treats its output as uncertain evidence, never an instruction or guaranteed interpretation.

Prepare permissioned `data/private/sense_labels.jsonl` with these required fields: `case_id`, `group_id`, `language`, `expression`, `utterance`, `previous`, `situation`, `meaning`, `provenance`. Each ambiguous expression must have examples of **at least two distinct senses** across independent conversations and speakers; avoid leaking text, targets or speaker groups across splits. `group_id` needs to represent an entire speaker/conversation family, not a random row.

Validation does not train:

~~~bash
bridgecaption-train-sense --data data/private/sense_labels.jsonl
~~~

Actual training is blocked behind `bridgecaption-preflight`, real live-test verification, and the explicit `--train --confirm-live-verified` flags. This is an **operator attestation**, not an automated proof of real-world readiness. Its JSON artifact may contain character fragments of private source data: keep it local and unshared. `BRIDGE_SENSE_MODEL` must be set explicitly to activate it.

The model's held-out coverage and accuracy must be evaluated per language and slang family, including negative transfer between literal and idiomatic meanings. If it fails to improve translation while controlling hallucinations and latency, leave it disabled. No independently validated trained sense artifact is included with this repository.
