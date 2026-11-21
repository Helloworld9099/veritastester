# Single-model research boundary

BridgeCaption's target is **one streaming multilingual speech model**: a shared audio encoder supplies both the source-language transcription and target-language translation. Changing a language control must use that same loaded checkpoint, not choose or reload a language-specific model. No new architecture, foundation, checkpoint or speech backend has been selected or activated here.

The checked-in `registry.json` proposes only scope and direction requirements. It contains a null candidate; `verified_directions` is therefore empty and any attempted translation fails before loading. The `SingleMultilingualResearchTranslator` Python interface is a **transitional text-only diagnostic harness**, not the intended speech interface. It has no default model loader, no downloads, no fallback, no production provider integration, and no path from a direction to a model. Its synthetic tests verify one load across mixed directions and reject incomplete evaluation. They do not establish speech, runtime stability, translation quality, model availability, or deployment readiness.

## Historical specialist evidence, not a product choice

The old independent ES→EN, FR→EN and EN→ES Marian artifacts remain immutable as benchmarks, lineage records, and explicit offline replay/rollback evidence. They are isolated under `historical/`; the single-model registry and runtime do not import or consult that package or the checkpoint inventory. The artifact catalog records archived files but is not a model routing table. See [the historical rehearsal procedure](../../docs/MULTILINGUAL_REGISTRIES.md#offline-historical-replay) for how to replay the pinned M0 DEV suite without enabling another runtime.

## Candidate acceptance is separate

The text harness can validate **one** locally supplied full checkpoint and tokenizer against an independently pinned candidate registry and a matching independently custodied, aggregate-only per-direction result receipt. Each direction needs its own passing evaluation; the scope cannot be rescued by a favorable average. A checksum does not establish who generated the receipt or verify bilingual competence; trust pins must be acquired independently. The adapter cannot choose a model based on the request. A reviewed explicitly injected loader receives only the same whole pinned checkpoint, never a route argument. No bundled loader executes. Do not reinterpret its text evidence as eligibility for the future shared speech system.

```sh
PYTHONPATH=src:. python -m unittest discover -s tests -p test_single_multilingual_research.py
```
