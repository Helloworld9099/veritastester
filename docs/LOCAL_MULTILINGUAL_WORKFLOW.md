# M4: run the research workflow locally

The workflow requires Python and an operator-controlled local input directory. It does not require GitHub Actions, a remote runner, a service account, a model download, or a cloud artifact store. Existing workflows remain unchanged. GitHub is only where the code is reviewed; the workflow below can be used on an offline checkout. All consolidation milestones use one integration branch and PR.

## What it does

1. Read a bounded JSON plan with an independently saved SHA-256. Reserve its exact bytes in a local ledger so a duplicate under the same run ID is reused, not rerun; an identical plan under a different run ID is rejected as already reserved.
2. Optionally prepare and verify **public, reviewed, unsealed train/dev** files. Validate roles, rights declarations, direction metadata, checksums and split overlap without rewriting the inputs or copying text into the ledger.
3. Optionally rehash a local pinned checkpoint. This validates byte identity only, without deserializing, loading, uploading, changing or copying weights. Model/architecture compatibility remains a separate check.
4. Optionally compare two independently pinned redacted **development** receipts. Exact reported checkpoint/output hashes and method/fixture matching can support an infrastructure evidence check only when the reported candidate matches the locally verified checkpoint and direction. Model-development diagnostics do not replace the stronger M2 per-direction gates. Data verification does not prove those rows produced the checkpoint.
5. Seal an immutable aggregate status and per-phase digests in the local ledger. The ledger is not a backup of weights, source corpora, or the phase input files. Keep the original files and pins in their authorized storage.

The plan has exactly these keys (null evidence is allowed and **not verified**):

```json
{
  "schema_version": 1,
  "kind": "local_research_workflow",
  "data": null,
  "checkpoint": null,
  "comparison": null
}
```

If present, `data` follows `research.workflow_data.prepare` and `checkpoint` follows `research.workflow_contract.verify_checkpoint`. A comparison has `baseline` and `candidate` refs of the form `{"path":"receipt.json","sha256":"<independent pin>"}` and optional `lane` of `infrastructure` or `model_dev`. References stay beneath the selected local input root. Use a separate ledger directory **outside** the checkout and input root, and choose opaque run IDs.

```sh
python -m research.local_workflow run --plan "$PLAN" --plan-sha256 "$PLAN_SHA" \\
  --input-root "$INPUT_ROOT" --store "$LEDGER_DIR" --run-id example-001
python -m research.local_workflow audit --store "$LEDGER_DIR" --run-id example-001
```

Run exits **0** for workflow-evidence pass, **3** for failed, incomplete or unverified evidence, and **2** for invalid inputs. Audit only reads; **0** means a finished receipt exists, not that it passed. A validation error after reservation leaves the ID reserved; prepare genuinely corrected inputs with new pins and use a new ID rather than clearing the old record. Deduplication is exact-byte and local to the ledger; use the same ledger for jobs that need to share it. No payload bytes or absolute paths are printed. An accepted checksum proves identity, not that licensing claims or human judgments are true.

## Boundaries

There is no training command, Actions dispatch, scheduled job, artifact upload, or production switch. M3 can inspect established trainers, but their live execution is still blocked by the lack of a verified local-input path. Do not substitute a mutable download or a fresh model. Do not send this tool sealed data; maintain independent custody of final evals. See [the unified speech target](UNIFIED_SPEECH_MODEL.md) for the future model path. Rollback is a code revert; keep ledger evidence and existing checkpoints intact.
