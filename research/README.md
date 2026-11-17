# Verified training research delivery — 15 November 2026

**Research checkpoints, not a ready or activated live translator.** This delivery preserves the existing application unchanged. The optional translator integration was explicitly set aside and is NOT included. No provider default, API configuration or production model is changed.

## Start here

1. [Continued neural training report](neural_continuation_20261116/REPORT.md): real French→English and Spanish→English Marian LoRA training, preserved selected and final checkpoints, model-selection protocol and uncertainty.
2. [Original neural pilots](translation_training_20261116/NEURAL_TRAINING_REPORT.md): preceding checkpoints and attribution.
3. [Independent neural review](independent_neural_review/NEURAL_REVIEW.md) and [fresh frozen-checkpoint revalidation](independent_revalidation_20261116/REPORT.md).
4. [Reproducible data preparation](data_preparation/README.md) and [dialogue preparation](context_dialogue_prep_20261116/README.md): source-only tooling and reports, no raw corpora.
5. [Auxiliary classifiers](auxiliary/README.md): topic, idiom and synthetic-sense experiments. These are NOT the translation model.

## Current evidence

Continued French training completed 256 new optimizer updates; the development-selected checkpoint was new step 128, not the last checkpoint. On 512 new Tatoeba examples, selected chrF was 75.566 versus base 74.905. On 160 DiaBLa turns, 57.886 versus 57.581. Spanish completed 128 fresh updates; on 256 new WMT examples selected chrF was 59.125 versus base 58.791. These are corpus scores, NOT percentages of correct translations. Reported gain intervals include zero. No broad conversation-quality/readiness claim is supported. A further independent holdout on the ORIGINAL adapters found French chrF delta -0.365 (95% interval [-1.713,+1.034]) and Spanish +0.031 ([-.570,+.613]); neither establishes an upgrade. Those original-adapter findings are not evaluations of the newer continuation checkpoints.

All 144 local DiaBLa dialogue groups have now been consumed across prior evaluations. Another genuinely untouched conversational evaluation needs another source or newly reviewed examples. No missing human memory-usefulness labels were fabricated. Generated fixtures remain labelled synthetic.

## Portable selected-checkpoint replay

The adapters are approximately 1.2 MB each and require the exact original base model. They are not standalone language models. `replay_adapter.py` is a standalone **research** utility, not an application integration. It never downloads, installs, changes settings or calls an external API. It checks the selected adapter against `CHECKSUMS.json`, validates all pinned base-file hashes, uses safe safetensors loading and stays offline.

Use an isolated environment with the versions in `requirements-replay.txt`; training used official CPU PyTorch 2.7.1+cpu. Acquire base files separately from the exact model/revision in the corresponding `model/repository_info.json`, then verify them against `model/download_manifest.json`. Base weights and dependencies are intentionally NOT committed.

From repository root:

```sh
python research/replay_adapter.py \
  --experiment-dir research/neural_continuation_20261116/french \
  --base-model /path/to/pinned/opus-mt-fr-en \
  --base-manifest research/translation_fr_training_20261116/model/download_manifest.json \
  --text 'Je ne veux pas annuler la réservation.'
```

Use `--baseline` for unchanged-base output or `--verify-only` to verify files without importing the neural runtime. Spanish uses the `spanish` experiment directory and `translation_training_20261116/model/download_manifest.json`.

Original executed training/preparation/review scripts are preserved as evidence, including their original directory assumptions. They are NOT advertised as one-command clean-checkout reruns: they require the source datasets, prior experiment layout and pinned base/dependency files. The new replay utility removes that layout dependency for selected-checkpoint inference. Preserve the recorded cohorts and hashes; don't retune on consumed test sets.

## Verification and publication boundaries

- Fresh application snapshot from main `4c45b8d734e52da9d10c567e58b32648b563589f`: 138 tests passed, 1 GUI/xvfb test skipped.
- Neural continuation reports 35 integrity checks, checkpoint replay and exact metric reproduction; data preparation reports 8 checks, dialogue preparation 9 checks.
- These do not establish real microphone capture, live GUI, low latency, robust multilingual translation or human-approved accuracy.
- `CHECKSUMS.json` covers delivered research files; individual experiment manifests describe the original executions.
- No raw third-party datasets, human-review queue text, real-data translation-output files, pretrained base weights, environment directories, credentials or discarded provider integration are included.

Models retain their upstream Apache-2.0 notices/cards. Tatoeba's older CC BY 2.0 mirror lacks contributor IDs; redistribution review remains unresolved. DiaBLa is CC BY-SA 4.0 and evaluation-only. MAGPIE is CC BY 4.0. WMT24 dialogue source is research-only CC BY-NC 4.0; preparation does not imply commercial clearance. Europarl/WMT and SemEval provenance/limitations are recorded in their reports. Publishing scripts/adapters here does not relicense or grant blanket commercial rights to external data.

## Deadline follow-up

[Additional French experiment](neural_deadline_20261116/README.md) completed 128 lower-rate updates on new data. Neither candidate passed the pre-registered development guard; the previously delivered adapter is retained unchanged. This addendum records aggregate evidence, not a model upgrade.

[Independent frozen-continuation audit](continuation_final_audit_20261116/REPORT.md): fresh French and Spanish cohorts of 128 each did not establish benefit over unchanged bases (chrF deltas −0.128 and +0.047; both 95% intervals include zero). Five French components overlap the concurrent experiment’s test set, not its train/dev; the audit reports this and a post-hoc sensitivity check. No candidate is promoted.
