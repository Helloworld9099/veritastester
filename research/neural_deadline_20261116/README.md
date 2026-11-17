# Deadline-bounded French training experiment

## Outcome: completed, no replacement selected
Registered at **2026-11-16 02:28:15 UTC**, completed all **128 updates** in 73.30 seconds of training compute / 134.37 seconds end-to-end. Finished well before the 02:40 UTC training cutoff. No paid service, app changes, integration, activation, push or deployment.

The hypothesis came only from the prior development curve: its 128-update continuation peak exceeded the later 256-update checkpoint. Test a four-times-lower learning rate (`2.5e-5` versus `1e-4`) from the existing French research adapter on a fresh training sample. Do not use prior test outcomes or the parent's synthetic probes for selection.

Reserved **2,048 train, 128 dev, 256 test human Tatoeba French-English pairs** before fitting. Original source component partitions retained. Excluded all previously sampled components: 5,036 train, 382 dev and 752 test components. No normalized source, reference or component overlap between current partitions or with earlier sampled fitting/dev/test records. The test has 252 components. No DiaBLa, prior held-out cohort, synthetic example or authored semantic probe was used.

### Registered conservative selection
Evaluate warm adapter and updates 64/128 on new dev only. A new candidate must improve corpus chrF by at least **0.25** and must not lower BLEU. Choose the highest qualifying chrF, favor earlier on ties; otherwise retain warm. The 0.25 margin is an operational guard, not a significance test.

| Dev checkpoint | chrF | BLEU | Qualified |
|---|---:|---:|---|
| Existing warm adapter | **76.1606** | **64.4569** | Baseline |
| New update 64 | 75.7017 | 63.9425 | No |
| New update 128 | 75.5338 | 63.7964 | No |

Neither new checkpoint qualified. Selected step is **0 (the unchanged warm adapter)**. Do not claim progress from the lower training loss or deliver either new checkpoint as an upgrade.

After selection was frozen, scored the retained adapter once on the newly reserved 256-row test: **75.3743 chrF / 62.2889 BLEU**. The selected and warm outputs are identical because they are the same model. Paired delta and its degenerate bootstrap interval are zero by identity, **not evidence of model improvement or of general safety**. Rejected candidates were not scored against this test.

## Exact artifacts

- Selected `adapter_selected.safetensors` is unchanged from the earlier delivered French checkpoint:
  `600990e205c1238ea1fba3a8c36e5ef199bdfc10b33e4e0068e5b9695b7e8665`
- Last newly trained, **rejected/non-selected** checkpoint:
  `0f9cc18e72f21d1b926c3ba0b5bf6755ceefbcaaceab02de414d597528c68ba0`
- Exact executed training-source SHA-256:
  `492b7aca5de6c293ac7ca0d5a4b3ed16c47ee04f41d662a2ec1e7ffa0b7566c0`
- All checkpoint files are safe tensor data, about 1.2 MB each. They require the pinned base model and are not standalone translators.

Original/warm checkpoints and base weights are preserved. Peak RSS 952.98 MiB; CPU four threads; no accelerator. Frozen base, 294,912 LoRA parameters, rank 8/alpha 16, microbatch 4 with accumulation 4, AdamW weight decay .01 and grad norm cap 1. This is a parameter-only warm start with a fresh optimizer, not resumed optimizer state.

## Validation

**16 research-integrity checks passed** in `verification.json`, including the frozen code/data hashes, pinned original base files, old/new partition separation, preserved warm adapter, exact registered selection, checkpoint replay, completed update log, exact metric reproduction and freeze-before-output ordering.

The portable prep CLI regenerated identical train/dev/test hashes using `exclusion_groups.json`, without needing previous raw sample folders. All four portable CLIs successfully expose `--help` in fresh processes. The exact original preparation code is retained as `prepare_used.py`; `prepare.py` adds the portable exclusion-ID input while preserving deterministic output. The executed `train.py` remains unchanged and matches its frozen protocol hash.

These checks do not establish live application readiness, speech quality, exact slang interpretation or independent human semantic review. No application test suite is claimed; application source was not changed.

## Rights, delivery and limitations

Tatoeba/ManyThings/TensorFlow 2019-04-07 snapshot carries CC BY 2.0 attribution; acknowledge Tatoeba contributors, ManyThings.org/anki and the TensorFlow mirror. The old snapshot lacks individual contributor IDs; no blanket downstream redistribution clearance is asserted. OPUS Marian French-English model card states Apache-2.0, pinned revision `c4aed37b318c763fd177aa449b44e3b783cc6c02`; `base_manifest.json` preserves upstream file hashes. No synthetic labels were called human labels.

These samples are newly held out from our fitting, not guaranteed unknown to OPUS pretraining. Component partitioning does not prove independent authors or paraphrases. This new test is now consumed. Do not tune further to it or describe later rescoring as untouched validation.

## Delivery scope and review caveats

This addendum includes aggregate evidence only. No rejected checkpoint, raw corpus or new training utility is committed. Original executed sources remain preserved separately; their hashes are recorded above. Review found the new verifier's documented fresh-work invocation requires copying the exact training script into that work directory, and it trusts a caller-supplied base manifest. Its passing checks are experiment integrity observations, not adversarial security verification or a fully verified clean-checkout recipe. The previously delivered hardened `research/replay_adapter.py` is unchanged.
