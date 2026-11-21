# BridgeCaption — one genuine multilingual translation model

**Branch:** `train/one-multilingual-student-20261121` · **Draft PR #48** · **Status:** Research experiment, not released.

## What is and is not one model

The previous French/Spanish consolidation shared one loader but retained separate Marian encoder/decoder weight sets and language tokenizers. The new student instead has **one pretrained M2M100 multilingual encoder/decoder** and **one shared learned low-rank adapter** used across all six directions (`es↔en`, `fr↔en`, `es↔fr`). Routing changes token IDs inside the **same** network; it does not select a language-specific model. The `export_standalone.py` experiment **folds** the shared adapter weights into the base matrices and saves an independent `model.safetensors`, plus tokenizer and config, to run without any adapter file.

This is a **shared text translation model, not yet a shared speech/ASR model**. BridgeCaption's microphone, Whisper, captions, update system and production translator remain exactly as before.

The trained result is not a continuation of old Marian tensors: Marian French and Spanish embeddings and tokenizers are incompatible with the new shared base. The old learned checkpoints are preserved for fallback, separate evaluation and possible verified teacher-guided distillation in the future. No legacy optimizer moments were imported.

## Base, data and licenses

| Resource | Pin / license | Use |
|---|---|---|
| Multilingual base | `facebook/m2m100_418M` SHA `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`, MIT | One model for English, Spanish, French |
| New student training | Amazon MASSIVE `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`, CC BY 4.0 | Publisher `train` only, same group ID across three locales |
| Model selection | MASSIVE publisher `validation`, not new independent final test | Repeated development, never called accuracy |
| Reviewed 500 translations | Already consumed development evaluation only | Their source text excluded from new training and dev |
| Previous ES→EN | `3bf849bb...` accepted Spanish v2 | Preserved separate, not merged |
| Previous FR→EN | `fba2bd78...` balanced French Stage 5 | Preserved separate, not merged |
| Noncommercial WMT24/subtitle adapters | CC BY-NC provenance | Excluded entirely from model ancestry |

The first resource budget is **$0** paid compute, existing free `ubuntu-24.04` CPU Actions, maximum 45 minutes. Model-card licence and exact revision are checked before training. Only 120 licensed tri-lingual training groups and 12 developer-only validation groups per route are sampled for the initial reproducible pilot. These groups are predominantly isolated voice commands, not authentic multi-turn slang or conversational context.

## Files

- `config.json`: unambiguous data/model licences, exact upstream revision, six routes, CPU and optimizer bounds.
- `data.py`: pinned publisher train/dev, three-locales-by-original-ID alignment, numeric/entity checks and review leakage exclusions. Does not access MASSIVE test.
- `model.py`: one shared M2M100 tokenizer, q/v LoRA, immutable hash-based adapter loader, route tokens, and in-place mathematical merging of one shared adapter.
- `train.py`: 18 actual optimizer steps, three saved SHA-pinned incremental checkpoints, before/after publisher-development chrF and BLEU per direction. Regression gate cannot authorize production by itself.
- `runtime.py`: opt-in BridgeCaption `TranslationRequest` / `Pipeline` translation with one resident trained student; rejects untrained dialogue-history injection and silent text truncation.
- `export_standalone.py`: convert real trained student to standalone *full* merged multilingual model. Expected download is large (approx 2 GB).
- `standalone_runtime.py` / `smoke_standalone.py`: load **one** standalone weight from local files and check six directions plus the real BridgeCaption text `Pipeline` with no reloads.
- `tests/`: source/license/model contract and deterministic split; numerical LoRA-merge test; all-six-route single-model cache contract.

## Run and inspect

```sh
PYTHONPATH=.:src python -m unittest discover -s research/multilingual_student/tests -p 'test_*.py' -v
PYTHONPATH=.:src python research/multilingual_student/train.py
```

CI: `.github/workflows/one-multilingual-student.yml`. A successful run writes immutable `research/multilingual_student/experiment_001/{report.json,data_manifest.json,shared_lora_step*.safetensors,SHA256SUMS.txt}`. It commits only small learned adapters, **not 2GB pretrained weights**. Re-running against the same output is blocked so successful experiments are not overwritten.

After that succeeds, add `research/multilingual_student/export_request.json` to trigger `.github/workflows/one-multilingual-student-export.yml`: it saves the full merged model as a downloadable 7-day **GitHub Actions artifact**, not as a 2 GB commit. It also records the actual SHA and six-direction BridgeCaption text-Pipeline smoke evidence. No API keys, paid services or private user speech are used.

**No claim of 95% meaning-preservation, dialect/slang competence or live microphone latency is implied by successful execution.** Before product deployment, run independent bilingual meaning scoring (both directions and subgroups), compare with old French and Spanish champions, verify rights/attribution, and test low-memory sustained full audio capture. A regression against old champions blocks any automatic replacement.
