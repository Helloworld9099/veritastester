# M3: research-only training coordination

BridgeCaption keeps its existing path for every training experiment. This milestone adds a typed, offline coordination layer around those paths, not a rewrite of the optimization loop. It does not change the app's provider, public CLI, packaging dependencies, model files, historical scripts, workflows, or the status of any checkpoint. M3 is based on main, not the unmerged M1 and M2 PRs.

## What can be used now

- **Inspect the actual Marian CPU trainer.** The adapter checks the unchanged trainer, route base/tokenizer manifest, LoRA protocol and historical warm adapter SHA in place. It reports the real Spanish→English and French→English argument mapping and fixed recipe. It does not import or invoke the legacy trainer. It cannot run the audited offline recipe because the legacy entry point internally fetches a mutable corpus and cannot take a prepared, independently pinned local split. It also saves adapter weights only. GPU, QLoRA, custom recipe values, unsupported pairs, and exact resume fail explicitly for that backend.
- **Use the coordinator with an explicitly supplied, reviewed backend.** `engine.run_local` requires local authorization and a trusted Python adapter already passed in by the caller. No config value can resolve a Python import, command, URL, or subprocess. The engine checks base/tokenizer files, pinned train/dev data, rights and source checkpoint before passing a structured request to the adapter. It persists a new immutable result at the requested update target, or an explicit, state-complete intermediate checkpoint from a controlled pause. Train and development payloads are separate fields; the reviewed optimizer must consume only the train field. One explicit direction per run keeps language pairs isolated. This API is not an OS sandbox; do not use an unreviewed adapter. No real neural backend is wired for execution here.
- **Preserve checkpoints and rollback.** Use a separate local artifact store outside the checkout and both input roots. Each new output gets a unique ID. The write reserves its directory exclusively, writes nonempty components, and seals the manifest last. Parts of a failed write are not loadable and the ID is not recycled. All referenced ancestor manifests are validated back to an explicit base start; current checkpoint payloads are rehashed without rereading every ancestor payload. No original is overwritten; get an older version by choosing its verified ID and manifest digest. The manifest records source identity, rights, step, and ancestry, not local/private dataset paths or raw rows.

## Warm start is not resume

| Mode | What it means |
|---|---|
| Explicit base start | No parent; only available to a backend that accepts the requested contract. The legacy Marian route rejects it. A failed parent lookup never silently becomes a base start. |
| Weights-only warm start | A compatible, checksum-verified adapter is passed to the backend. Step numbering begins at zero with fresh optimizer/scheduler/sampler/RNG state. Architecture, direction, base, tokenizer, strategy and backend protocol must still match. |
| Exact resume request | The adapter, optimizer, scheduler, sampler and RNG bytes must all be present and independently pinned. The declared recipe, data/order, seed, rights, precision/device and version must match; the backend must explicitly support validating/restoring its exact runtime state and return a restore receipt bound to the source component hashes. Merely possessing byte-complete files does not prove replay or equivalence across hosts. No built-in backend supports this today. |

The noncommercial lane never flows into a commercial candidate, and a commercial-candidate manifest still does not certify commercial rights or promotion. GPU and QLoRA entries are hooks, not active or tested inference/training paths.

## Inspect a local checkout

Use an independently built, versioned JSON configuration (schema defined in `research/unified_training/config.py`). In the repository root run:

```sh
python -m research.unified_training inspect --config "$CONFIG"
```

This command reports recipe compatibility and blockers, not a training result. Exit **0** means the corresponding legacy recipe could be inspected and matched, **3** means a requested capability or pin does not match, and **2** means invalid metadata or unavailable source/adapter. The JSON may say `inspection_only`; always read `execution_supported` separately. There is deliberately no CLI command to run training.

A config references only train/development manifests by SHA-256 and relative path beneath an explicitly selected local data root. A `reviewed_research_training_split` manifest (schema 1) binds `dataset_id`, `version`, `split`, `sealed:false`, `license`, `visibility`, `records_path` relative to the manifest, `records_sha256`, and `row_count`. Use exactly `source`, `target`, `source_language`, `target_language`, and `split` in each JSONL row (optional `original_id` and `group_id`). Do not forward raw mixed-context or holdout records; this first coordinator contract is current-only. A bounded snapshot passes the same verified bytes to the reviewed adapter. Prefer a preparation step using the separately reviewed M1 split and license checks; these type/hash checks are not a provenance audit or a guarantee against paraphrase overlap. Never supply sealed evaluation data.

## Verification, limitations and rollback

The wrapper was verified on the unchanged Spanish and French legacy entry point and its pinned historical inputs. A synthetic in-process adapter tests orchestration, state hand-off, a controlled pause and subsequent resume, output protection and failure paths. No learned weights, held-out evaluation, GPU run, download, or training job was executed. Real offline warm-start execution, exact resume in a neural backend, and output/performance equivalence to the historical trainer remain unverified. Do not replace the legacy route until a reviewed local-input seam and separate, budgeted parity run exist. Do not substitute a fresh model to bypass that gate.

Revert this research-only change to recover the previous tooling. Nothing in production or the original experiments needs undoing; retain every checkpoint and receipt.
