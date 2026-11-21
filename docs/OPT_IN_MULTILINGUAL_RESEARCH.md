# M6: opt-in efficiency and shared-model preparation

This work stays in the multilingual integration branch and PR. Nothing is enabled in the desktop app. Nothing reads a microphone, downloads or loads a model, runs training, selects a real candidate, opens a sealed split, or depends on GitHub Actions. The product goal remains one streaming speech model for both original and translated captions, not a switch between specialist weights.

## Optional tools

| Tool | What was verified | What it does not establish |
| --- | --- | --- |
| Bounded tokenization cache | On a pinned synthetic input sequence, outputs and order are exact with or without cache; calls fall from 32 to 18. Default off, bounded by entries and retained bytes, context and private/default inputs bypass it, no disk. | Wall-clock speedup, field memory savings, or quality on real training. |
| Successive halving and early stopping | An eight-candidate synthetic race allocates 96 updates rather than 128 for eight full-budget runs. Each rung is charged as an independent run from the same pinned warm start; no assumed exact-resume savings. Per-direction failures or unknown evidence cannot be averaged away. | An accuracy win, execution of any allocated update, or a promoted model. |
| Continual learning preparation | Deterministic selections from verified train-only rows with explicit direction and context/slang quotas; no raw text in the plan. | Protection against catastrophic forgetting; only later frozen per-direction regression, specialist baseline comparisons, and rollback checks can measure that. |
| Shared-model preparation | One pinned student identity, rights-cleared training inputs and separate output-use approvals for teacher material; no hard-coded production route. | Rights inferred from a model card alone, permission to train, or compatibility to merge weights. |
| Local speech benchmark evidence | Matches reviewed DEV audio metadata, host/method/budget and supplied incumbent/candidate direction receipts. ASR, speech-translation, timing and resource measurements are compared per direction. | A benchmark run, a model recommendation, verified independent final quality or activation. |

Reuse the existing CPU trainer's opt-in sequence-length bucketing and gradient accumulation work from [PR #42](https://github.com/Helloworld9099/veritastester/pull/42). Do not introduce a second implementation here. Bucketing can change optimizer row order, while caching tokenization alone cannot. Enabling each change requires its own controlled evidence. Active-learning execution, early-stop launch, production optimization and real teacher generation are deliberately absent; no real-data benefit or independent review has been established.

## Run the bounded local demonstrations

```sh
python -m research.unified_training.efficient_batches
python -m research.unified_models.speech_benchmark --help
python -m pytest -q tests/test_unified_efficient_batches.py tests/test_successive_halving.py
```

`successive_halving.plan_next_rung` accepts an explicitly opted-in frozen DEV protocol, ordered completed rungs, and independently trusted hashes of derived candidate-specific receipts. The custodian verifies the underlying M2 reports before pinning those summaries. Each acceptable receipt declares paired row counts, its original method and warm-start pins, the declared candidate recipe and evaluated checkpoint SHA-256, chrF delta, new critical-error count, subgroup status, and comparable performance status for each direction. A result is a recommendation only; there is no training entry point. A derived receipt hash proves matching bytes, not that a run happened.

Use `continual.plan_continual_replay` only with verified source release manifests and explicit train-only quotas. Keep input references, corpus splits and human-review custody intact. Use `shared_plan.prepare_shared_plan` with independently pinned aggregate reviews. Text teachers may contribute auxiliary translation guidance where the applicable rights allow it; text alone cannot supervise a speech encoder. A single shared streaming model would still need paired audio, transcripts and translations, plus direction and licensing evidence. Do not move private or noncommercial ancestry into an incompatible lineage.

## Compare proposed foundations before choosing one

Record the identical licensed DEV clips, source/target references, audio order/boundaries, runtime/machine, decoder policy and budget for the incumbent and candidate. Test both original transcription and translated outputs from the same audio. Record warm original/provisional and stable output latency, missed/revised captions, backlog, RSS, and cold startup separately; compare quality per language direction and annotate meaning-changing errors. A candidate may qualify only for directions actually tested. Keep the original app unchanged until a candidate can be evaluated independently and approved. A small smoke is not a latency promise, and no checkpoint, rights or language pair is promoted here. Use `speech_benchmark preflight --manifest FILE --manifest-sha256 SHA` to validate protocol metadata only; `compare` also takes `--incumbent`, `--candidate` and their respective `-sha256` pins. Too few, noisy, threshold-straddling or unpaired measurements are `not_verified`, not passes. A numeric `reported_within_bounds` is still only a development diagnostic from submitted evidence.

The shipped provider can require a network. Do not substitute cached responses or fake ASR/provider output and call that a live incumbent benchmark. Describe the exact incumbent arrangement and cost, and treat unmatched or unavailable measurements as not verified rather than changing production to manufacture a comparison.
