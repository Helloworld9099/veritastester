# BridgeCaption multilingual research registries (M1)

The long-term product is **one multilingual BridgeCaption bot**, not one file of combined model weights. A shared interface can select eligible specialists without users managing translators. M1 is strictly research-side: it does not route the installed app through these registries or change its provider, prompts, settings, CLI, install dependencies, captions or audio processing.

## Where to add things

| Item | Research boundary |
| --- | --- |
| Language | Declare an independent entry in the language registry: code/aliases, script, writing direction, normalization/tokenization needs, dialects and code-switching. A declaration does not train or enable a translation model. Preserve existing entry and alias behavior. |
| Existing model | Add identity, architecture, tokenizer/base revision, verified directions, adapter/checksum bindings, ancestry, data use and evaluation evidence. Refer to immutable weights in place. Validate byte identity and compatibility before loading. Keep an existing production provider out of the research registry. |
| New adapter | Give each checkpoint a new artifact ID and record its exact base/tokenizer binding, architecture, relationship to its parent, provenance and rights. Never overwrite a previous artifact or treat a weights-only warm start as an exact resume. |
| Historical or private asset | Inventory it without creating a loadable route. Include only public-safe metadata in Git; no private archive URLs, account identifiers, proprietary text or private checkpoints. Missing proof is an explicit unknown, not a substitute or a silent fallback. |
| Dataset | Make a new versioned conversion and checksum receipt. Do not edit the source or infer train/dev/test assignments. Declare provenance, rights, visibility, and which source columns supply original IDs, context and metadata. The converter is not a license check or a sign-off for training. |

## Scope of the registry

- The shared local **research** runtime covers ES→EN and FR→EN using separate existing adapters, and EN→ES with an unadapted base. Do not infer EN→FR or ES↔FR support. An unsupported direction fails clearly. Decoding uses the same 128-token input / 96-token output bounds as M0. Boundary-length omissions still need separate research evaluation.
- The archived noncommercial WMT24 and Spanish data lineages remain preserved, not promoted. CMN→EN is a separate private, noncommercial architecture. Its exact weights and datasets do not belong in this repository. A language entry or checkpoint index record does not enable it; the M0 Mandarin replay gap remains. Auxiliary models are not translation directions.
- The design is adapted from [draft PR #43](https://github.com/Helloworld9099/veritastester/pull/43) at `ca755aec8f7111f03145e18ef9948cd6127aaec0`, without that draft's ancestry, app prompt changes or automation. Keep the older research and app entry points until a separately verified migration.
- Do not use the M0 development outputs as training targets, use any consumed evaluation as a new holdout, publish sealed/private rows, or interpret exact replay as proof of adequacy. Existing cross-platform and independent human-review gaps remain.

## Use the registry without touching the app

Run from a checkout with the existing app package importable (for example, set `PYTHONPATH=src:.`). Listing directions or validating metadata doesn't require Torch. The research translator requires the pinned offline base cache, optional inference dependencies, and an explicit read-only artifact checkout for adapted directions. For this migration the checkout is the #43 commit above; the older app install is still independent. Never put a private repository location or auth token in the registry.

```python
from pathlib import Path
from research.unified_models.registry import ModelRegistry
from research.unified_models.runtime import SharedMarianResearchTranslator

registry = ModelRegistry()
translator = SharedMarianResearchTranslator(
    registry=registry,
    cache_dir=Path("path/to/verified-base-cache"),
    artifact_root=Path("path/to/read-only-pr43-checkout"),
)
print(translator.routes())  # Metadata only; no model is loaded.
```

`registry.json` is the only authority for routes. The checkpoint inventory is an audit index of referenced artifacts, including historical candidates; loading does not consult it and no archived model becomes active because it appears there. Add immutable pins and prove compatibility and direction coverage before proposing a new route. To rehash the preserved ordinary Git objects at the 94 paths in the original main snapshot, run the non-loading audit below in a matching checkout:

```sh
PYTHONPATH=src:. python -m research.unified_models.checkpoint_inventory --verify-dir . --ref main
```

See the [runtime notes](../research/unified_models/README.md) for how to call a route, select its same-direction rollback, and supply the pinned offline data.

## Convert a dataset in a local research workspace

Supply a JSON dataset spec (`dataset_id`, `version`, `source_uri`, `license`, explicit reviewed rights and visibility) and a JSON mapping of canonical field names to source-column names. Assign stable original IDs and explicit splits upstream. Missing license evidence is not permission. Write to a new private directory outside Git, even when the source itself is public:

```sh
PYTHONPATH=src:. python -m research.unified_models.datasets convert \
  local/source.jsonl local/bundles/example-v1 --spec local/spec.json --columns local/columns.json
PYTHONPATH=src:. python -m research.unified_models.datasets verify local/bundles/example-v1
```

There are no downloader or trainer side effects. A bundle contains deterministic records and a manifest with input/output SHA-256 and conversion options. Save the manifest digest separately for later verification; a self-consistent rewritten bundle alone does not prove identity. Use the train-only reader with an explicit purpose, one independently saved manifest digest per bundle, and applicable **unsealed** comparison bundles. An unopened sealed set cannot be certified contamination-free by this converter. Do not give training or tuning jobs the raw sealed final set; keep that independent gate private and under evaluator control. The overlap audit is conservative and cannot detect semantic paraphrases, provenance mislabeling, or leaks through unavailable data.

## Exact M1 replay against M0

Run from either checkout with the same official CPU Python environment in the published M0 fixture manifest. Use absolute paths for the verifier, frozen fixture pair, cache, and receipts. As in M0, the cache layout is `<base-cache>/<direction>/<pinned-base-revision>/<pinned-file>`. Prepare only the manifest-listed files independently before running. No downloads are allowed during replay.

```sh
# Set PR43 and M1 to local checkout roots, AUDIT to a scratch output directory,
# FIXTURES to the published M0 fixture directory, and BASE_CACHE to verified files.
VERIFY="$M1/research/unified_models/verify_baseline.py"
FIXED="$FIXTURES/multilingual_regression_v1"
(cd "$PR43" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -I "$VERIFY" \
  --phase pre --cache "$BASE_CACHE" --fixtures "$FIXED.jsonl" \
  --manifest "$FIXED.manifest.json" --receipt "$AUDIT/before.json")
(cd "$M1" && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -I "$VERIFY" \
  --phase post --artifact-root "$PR43" --cache "$BASE_CACHE" \
  --fixtures "$FIXED.jsonl" --manifest "$FIXED.manifest.json" \
  --expect-registry-path research/unified_models/registry.py \
  --compare-pre-receipt "$AUDIT/before.json" --receipt "$AUDIT/after.json")
```

The pre checkout must be detached at PR #43 commit `ca755aec8f7111f03145e18ef9948cd6127aaec0`. A source and artifact checkout are distinct in the post run. The verifier checks published fixture bytes and manifest, model and tokenizer files, observed route versions, candidate registry identity, exact output hashes for 18 authored DEV examples, rejection of unsupported directions, and actual Spanish and French rollback decoding. It makes no audio, microphone, production, independent translation quality or Mandarin claim. Each run gives six warm requests per direction; two candidate processes let us pool 12 per direction for comparison against published M0. In those runs, **18/18 exact outputs matched in each process** and both same-route rollback hashes and outputs matched the #43 pre-run. Warm p95 was 820→823 ms EN→ES, 857→866 ms ES→EN, and 946→876 ms FR→EN. Peak RSS was 869,736/859,552 KiB at M0 and 827,944/828,540 KiB after M1. These observed deltas stayed within the guardrail on this test; the sample is small and not a production performance or improvement claim.

## Review and rollback

The [M0 frozen fixtures and manifest](https://github.com/Helloworld9099/veritastester/tree/ebcc869c9e3fe7aa42a4d237f30cf44ef6d79028/tests/fixtures) are an independent pinned input. Do not update them to make a candidate pass. Replay only with the exact cache and decode pins, with no automatic downloads in the gate. Keep research-only evaluation separate from foundation tests. Compare the explicit route matrix, exact output bytes and checkpoint hashes first; only compare performance under repeated runs on the same environment. No independent bilingual meaning or performance threshold is implied by a successful smoke replay.

A revert of M1's code/metadata needs no data, weights, workflows or installer rollback. The prior standalone research branch and shipped provider are unchanged. Do not delete historical files. A failed compatibility, rights or parity gate never enables a different route or commercializes a checkpoint.
