# BridgeCaption single multilingual speech-model boundary (M1)

The target is **one streaming multilingual speech model**, with a shared audio encoder feeding the original transcription and translated output. The same loaded checkpoint must handle every independently verified direction without selecting a specialist, swapping an adapter, or reloading weights. This deliverable selects no foundation and implements no audio backend. The installed app, provider, prompts, settings, CLI, install dependencies, captions and audio flow are unchanged.

## What exists now

- The proposed registry records target directions EN→ES, ES→EN and FR→EN, with **zero verified directions** and **no configured candidate or default loader**. These are initial validation requirements, not capability claims. The existing text-only research harness exercises an immutable one-checkpoint boundary with synthetic fixtures; it is not the streaming speech architecture or a replacement product path.
- Future speech validation needs paired source transcripts and target translations, one model identity across directions, chunked-audio/streaming behavior, no full model reload, independent per-direction evidence, and latency/RSS measurements from the actual streaming host. Do not infer that text parity proves ASR, streaming, context, slang, or speaker-level accuracy. Do not select, download, merge or train a model under this scope.
- The historical Marian specialists stay pinned as separate benchmark/rollback artifacts, not selectable runtime alternatives. The archived EN→ES base, ES→EN and FR→EN adapters, noncommercial WMT24 and Spanish lineages, and all opt-in development evidence remain distinct from the proposed architecture. CMN→EN private/noncommercial artifacts stay off repo and unroutable. No unsupported language triggers a fallback.

## Keeping one model

`research/unified_models/registry.json` accepts one complete candidate or null. Every target direction must be bound to independent per-direction results for the **same** checkpoint. The independently obtained pins and redacted review evidence verify the chosen bytes; they do not themselves grant licensing, approval, or evidence of reviewer competency. Further gates are required before real execution or integration. The active research table contains no alternate, default-by-language, or rollback-by-request options.

```python
from research.unified_models.runtime import SingleMultilingualResearchTranslator

status = SingleMultilingualResearchTranslator().status()
assert status["candidate_configured"] is False
assert status["verified_directions"] == []
assert status["production_activated"] is False
```

The checkpoint inventory remains a provenance catalog. It is never consulted by the proposed runtime. Historical specialist testing is intentionally isolated under `research/unified_models/historical/`, requires an explicit replay opt-in, and must not be exposed to app callers, language selectors, or standard translation workflows. To rehash preserved ordinary Git objects from the main snapshot without loading a model:

```sh
PYTHONPATH=src:. python -m research.unified_models.checkpoint_inventory --verify-dir . --ref main
```

## Convert a dataset in a local research workspace

Supply a JSON dataset spec (`dataset_id`, `version`, `source_uri`, `license`, explicit reviewed rights and visibility) and a JSON mapping of canonical field names to source-column names. Assign stable original IDs and explicit splits upstream. Missing license evidence is not permission. Write to a new private directory outside Git, even when the source itself is public:

```sh
PYTHONPATH=src:. python -m research.unified_models.datasets convert \
  local/source.jsonl local/bundles/example-v1 --spec local/spec.json --columns local/columns.json
PYTHONPATH=src:. python -m research.unified_models.datasets verify local/bundles/example-v1
```

There are no downloader or trainer side effects. A bundle contains deterministic records and a manifest with input/output SHA-256 and conversion options. Save the manifest digest separately for later verification; a self-consistent rewritten bundle alone does not prove identity. Use the train-only reader with an explicit purpose, one independently saved manifest digest per bundle, and applicable **unsealed** comparison bundles. An unopened sealed set cannot be certified contamination-free by this converter. Do not give training or tuning jobs the raw sealed final set; keep that independent gate private and under evaluator control. The overlap audit is conservative and cannot detect semantic paraphrases, provenance mislabeling, or leaks through unavailable data.

## Offline historical replay

Use the detached #43 checkout and the unified branch’s explicitly archived specialist baseline package, with the CPU Python environment pinned in the published M0 manifest. This is not an operational translation route, and the proposed single-model speech system is not exercised. Use absolute paths for the verifier, frozen fixture pair, cache, and receipts. As in M0, the cache layout is `<base-cache>/<direction>/<pinned-base-revision>/<pinned-file>`. Prepare only the manifest-listed files independently before running. No downloads are allowed during replay.

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
  --expect-registry-path research/unified_models/historical/registry.py \
  --compare-pre-receipt "$AUDIT/before.json" --receipt "$AUDIT/after.json")
```

The pre checkout must be detached at PR #43 commit `ca755aec8f7111f03145e18ef9948cd6127aaec0`. A source and artifact checkout are distinct in the post run. The verifier checks published fixture bytes and manifest, model and tokenizer files, observed route versions, historical benchmark registry identity, exact output hashes for 18 authored DEV examples, rejection of unsupported directions, and actual Spanish and French rollback decoding. It makes no audio, microphone, production, independent translation quality or Mandarin claim. Each run gives six warm requests per direction; two candidate processes let us pool 12 per direction for comparison against published M0. In those runs, **18/18 exact outputs matched in each process** and both same-route rollback hashes and outputs matched the #43 pre-run. Warm p95 was 820→823 ms EN→ES, 857→866 ms ES→EN, and 946→876 ms FR→EN. Peak RSS was 869,736/859,552 KiB at M0 and 827,944/828,540 KiB in the earlier M1 historical comparison. These past observations stayed within the guardrail on this test; the sample is small and not a production performance or improvement claim.

## Review and rollback

The [M0 frozen fixtures and manifest](https://github.com/Helloworld9099/veritastester/tree/ebcc869c9e3fe7aa42a4d237f30cf44ef6d79028/tests/fixtures) are an independent pinned input. Do not update them to make a candidate pass. Replay only with the exact cache and decode pins, with no automatic downloads in the gate. Keep research-only evaluation separate from foundation tests. Compare the explicit route matrix, exact output bytes and checkpoint hashes first; only compare performance under repeated runs on the same environment. No independent bilingual meaning or performance threshold is implied by a successful smoke replay.

A revert of this research-only code/metadata needs no data, weights, workflows or installer rollback. The prior standalone research branch and shipped provider are unchanged. Do not delete historical files. A failed compatibility, rights or parity gate never enables a different route or commercializes a checkpoint.
