# BridgeCaption — Consolidated Spanish/French Model Management (Research, Nov 2026)

**Current status:** Shared opt-in model runtime, **not a new merged neural network**. Production BridgeCaption's existing default translator, microphone/ASR integration, desktop UI and updater have not changed. Do not merge into `main` or switch ordinary users to this adapter registry without independent bilingual quality evidence, complete app QA, and a license/provenance review.

## Why this is a registry, not one weight file

The current ES→EN and FR→EN models use *different Marian pretrained bases*, training corpora and SentencePiece vocabularies. Even when they share LoRA rank and projection shapes, element-wise weight merging would corrupt the associations between each language's token indices and the adapter. This is **not safe** and provides no measured quality benefit. A genuinely single ES/FR/EN model would require a new multilingual base model and controlled distillation/fine-tuning with checks that neither language regresses.

Instead, we now have **one** route and loading implementation: `research/unified_models/registry.py`, `runtime.py` and `registry.json`. It uses explicit ASR/translator language codes (e.g., `es-MX`, `fr-FR`, `en-US`), chooses the correct compatible pretrained model and adapter, and keeps **at most one heavy base model loaded at a time**. It does not guess when source language is `auto`; unknown or unsupported directions are rejected explicitly instead of mistranslated.

| Direction | Preserved model / weights | Rollback | Rights and quality status |
|---|---|---|---|
| `es-en` | v2 accepted after round 2: SHA `3bf849bb...` | Earlier MASSIVE Stage 3: SHA `14dddb50...` | Tatoeba/ManyThings + MASSIVE CC BY training ancestry; **no independent 95% human assessment** |
| `fr-en` | Selected balanced Stage 5: SHA `fba2bd78...`; Stage 6 selected **previous**, preserving stage 5 | French MASSIVE Stage 3: SHA `072f7f5d...` | Earlier Tatoeba/MASSIVE + Global Voices attribution to verify; **no independent 95% assessment** |
| `en-es` | Pinned original `Helsinki-NLP/opus-mt-en-es` revision `5bc4493d...` | Same original base | No fine-tuned English→Spanish adapter |
| Archive only | French WMT24 contextual `c858eb0a...`; Spanish NC subtitle 15-round champion `557838ab...` | **Never routed by the unified runtime** | **NONCOMMERCIAL** training ancestry, must not contaminate allowed production-candidate weights |

This preserves all existing checkpoint files in their original locations, along with their SHA-256 checksum history. The registry is the one discovery point rather than copying many checkpoint bundles into different code paths.

### Opt-in usage, not a user-facing model dropdown

```python
from research.unified_models.runtime import SharedMarianResearchTranslator
from bridgecaption.pipeline import Pipeline

# Optional research invocation only; original BridgeCaption app remains unchanged.
engine = SharedMarianResearchTranslator()
pipe = Pipeline(translator=engine)
es = pipe.translate("Hola, ¿cómo estás?", source="es", target="en")
fr = pipe.translate("Bonjour, comment allez-vous ?", source="fr", target="en")
```

The detected **source language** determines the base/adapter automatically; no user choice of underlying AI is needed. If the source language cannot be confidently identified, do NOT send it to a random Marian model. `en-fr` and `fr-es` are NOT falsely claimed supported. Existing configured cloud provider remains available through the app's existing factory and is not silently invoked by this research runner.

`research/es_en_v2/integration.py` remains a **backward-compatible Spanish research wrapper**, but it now uses the shared loader instead of maintaining a separate French-incompatible copy of LoRA code. The direct `research.unified_models` module is research-only, so no additional >300MB model download is forced on installed users.

## How the 500-row review is being applied

**Original evidence:** `research/es_en_v2/reviewer_feedback_20261120/processed` maps the supplied notes/PDF to the actual 500 model responses: **178 distinct rows flagged** (ES→EN 72, EN→ES 106), with overlapping concerns. `322` unflagged rows are NOT assumed correct, and there are **zero formal per-row pass/fail labels**. Source oddities and some reported row numbers require human adjudication. This is development feedback, NOT the new sealed final holdout.

**Implemented review improvements, without pretending that the weights learned them:**

1. **Better default context-aware provider hints** in `src/bridgecaption/context.py` for ambiguous `guagua`, `I'm down`, `lana`, `ghosted`, `slaps`, `me dejó en visto`, `break a leg`, etc. All hints are uncertain possibilities, not hard-coded answers; they are sent only when context mode is explicitly enabled. Locale prefixes (`es-MX`, `fr-FR`) are normalized to the relevant slang dictionary.
2. **Prevent looping captions** in the shared experimental Marian runtime. When output contains an abnormal dominating repetition sequence, retry **once** with a constrained decoder. Ordinary outputs are not redecoded. A second looping output raises a clear error rather than emitting nonsensical repeated captions. This is a **runtime decoding adjustment**, not fine-tuning or a general 95% translation fix. The three real reported loops in rows 311, 314, 416 are preserved as tests.
3. **Avoid silently omitting the last clause.** The shared research loader rejects input longer than the model's verified 128-token source window instead of silently cutting off later clauses. Chunking/streaming is NOT yet trained or verified. This may cause longer inputs to be rejected; only send a validated shorter speech segment, not arbitrary truncation.
4. **No context-leakage regression.** Neither existing ES nor FR adapter was trained to reliably process appended conversation history. The shared local model explicitly sets `supports_context=False` and refuses supplied past speech when asked to use it. Real context still benefits the default reasoning provider's structured prompt.
5. **No contamination.** Reported errors are used to prioritize future new legally licensed training examples; the already-reviewed 500 are development-only, not new holdout or training targets. The assistant did NOT train new weights or claim the error analysis is a formal adequacy score.

**Remaining model problems:** pronoun flips (row 318), time-of-day shifts (457), lost ordinals (436), omitted clauses, child-directed register, regional names/brands and slang/idiom errors are not yet fixed within the local neural weights. They need dedicated bilingual verified training/validation examples. A decoder retry cannot repair meaning and is not presented as if it can.

## Regression and rights tests

- `PYTHONPATH=.:src python -m unittest discover -s research/unified_models/tests -v`
- `PYTHONPATH=.:src python -m unittest research.es_en_v2.tests.test_inference_safety research.es_en_v2.tests.test_reviewer_feedback_500 -v`
- Actual CPU smoke: `PYTHONPATH=.:src python research/unified_models/smoke.py`, see `research/unified_models/receipts/actual_multilingual_integration.json` if present.

The real smoke verifies each model's original base revision/tokenizer hashes and adapter SHA, runs both languages through `Pipeline`, switches to the correct rollback weights, keeps memory bounded, and compares Spanish/English baseline outputs with previously frozen *development* outputs. This is **technical parity, not meaning adequacy**; a new human evaluation and licensed 2,000-per-direction holdout remain required. A failed quality or parity gate blocks deployment and does not erase the preserved model.

## What is not accomplished yet

- A single weight file supporting all languages.
- A commercially cleared, independent human-rated 95% adequate Spanish or French model.
- A locally trained context-capable French/Spanish Marian model.
- French output quality evaluation against a genuinely unseen human test in this consolidation.
- Windows/macOS microphone-to-caption release QA and a locally installed offline auto-updater check.
- Any activation of the experimental local model as the app's default provider.

Maintaining these honest boundaries is more valuable than turning one mixed model into a language regression or creating false human-accuracy claims.
