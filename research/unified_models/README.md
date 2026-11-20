# Shared translation runtime (research only)

One explicit EN/ES/FR router for separate pinned Marian bases and adapters, **not one blended model** and not installed/activated by the application. The default provider, context prompt code, UI, ASR and updater are out of scope. There is no silent language guess, fallback provider, pivot through English, training, or automatic download.

| Direction | Research route | Provenance and evaluation status |
|---|---|---|
| ES→EN | v2 selected model and preserved MASSIVE stage-3 rollback | Byte-linked parent is historical Europarl continuation `55d4…` then MASSIVE. The old draft’s Tatoeba shorthand conflicts with this report. Commercial provenance and attribution are not cleared. Existing development metrics and parity are not independent adequacy. |
| FR→EN | balanced stage 5 and preserved MASSIVE stage-3 rollback | Historical Tatoeba/MASSIVE then Global Voices news; article-by-article rights/attribution unresolved. No independently human-rated adequacy. |
| EN→ES | original pinned Marian only | No fine-tuning or adapter rollback. Preserve the baseline identity and original behavior. |
| cmn→en | private noncommercial metadata only | Inactive and not loadable; no private artifacts, identifiers, paths, or eval contents are exposed. |

English→French and direct Spanish↔French are unsupported. Regional hints in the registry describe language metadata, not guaranteed dialect/code-switching coverage, and do not change the tokenizer selected for a route.

## Run the research interface locally

The base cache must contain only the files listed in `registry.json` under `<route>/<base revision>/`. Adapted routes additionally require an explicitly supplied local source with the exact adapter bytes at its pinned repository-relative path. The registry records immutable upstream commits/paths/checksums; it does not fetch or bundle model weights. Never point this at an untrusted or concurrently edited cache. This is a research-only API, intentionally outside the default application provider factory.

```python
from pathlib import Path
from bridgecaption.pipeline import Pipeline
from research.unified_models.runtime import SharedMarianResearchTranslator

engine = SharedMarianResearchTranslator(
    cache_dir=Path("locally-verified-base-cache"),
    artifact_root=Path("locally-verified-external-artifacts"),
)
pipeline = Pipeline(translator=engine)
caption = pipeline.translate("Bonjour !", source="fr", target="en")
```

Before any cold load, a route validates its base/model revision, all base and tokenizer hashes, safe file selection, config-declared Marian architecture and matching adapter schema/checksum, exact tensor keys/F32 shape/layout and finite tensor data. Validation occurs before constructing the model, followed by a defensive loaded-layer check before attachment. Warm instances use the already verified in-memory model. The cache keeps one CPU model; explicit rollback is same-route only and EN→ES has no rollback adapter. Local inference cannot use enabled glossary, sense hints or preceding speech; an explicit direct request supplying them fails closed. The existing Pipeline uses support flags to disable them. Direct requests above the validated token/text window are rejected; no claim of streaming or chunking. The bounded one-retry repetition guard preserves the pinned research decoder but is not an adequacy repair.

Extend the registry by adding canonical language metadata (script, direction, normalization, tokenization, dialect descriptors, code-switch policy) and a pinned route definition. Adapted routes additionally require matching current and rollback identities linked to the base/tokenizer/LoRA compatibility digest; the test-only dummy route shows that a fourth active language can be declared without modifying code and cannot load missing dummy assets. The separate archival checkpoint inventory is documentation, not a routing table. Forbidden noncommercial checkpoints cannot be used by current or rollback routes.

```sh
PYTHONPATH=.:src python -m unittest discover -s tests -p test_registry_m1.py
```
