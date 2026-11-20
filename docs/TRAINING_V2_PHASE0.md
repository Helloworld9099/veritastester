# BridgeCaption — Spanish/English training v2: Phase 0 and audit (2026-11-20)

This is the binding execution contract for branch `train/es-en-v2`. Later user direction supersedes pauses at phase gates: continue automatically except when genuine external human participation is unavoidable. Final bilingual holdout scoring is always **human-run, not agent-run**.

## Phase 0 — all fields resolved

| Item | Actual authorized bound |
|---|---|
| Repository / working branch | `Helloworld9099/veritastester` / `train/es-en-v2` |
| Hardware | **CPU only** on existing free GitHub Actions `ubuntu-24.04` runners; no GPU is authorized. Runtime CPU/thread count must be recorded by jobs. |
| Max GPU-hours per round | **0** |
| Max GPU-hours total | **0** |
| Max wall time per training round | **45 minutes** (GitHub timeout; abort rather than spending money) |
| Max paid spend | **$0**; do not use paid APIs, hosted accelerators, or billed machines |
| Compute allowance | Existing free public-repository GitHub Actions only; stop scheduling when not available rather than purchasing capacity. |
| Web access | **Yes**, restricted to public papers, source documentation and licensed corpora |
| Allowed teacher models | Open-weight models with verified permitted redistribution and commercial-use terms, running locally/inside existing free CPU runners. No paid or remote teacher API. Teacher output **not** considered automatically human-verified |
| Dataset licenses | CC0, CC-BY, CC-BY-SA, Apache-2.0, MIT, public domain, subject to per-source provenance, attribution and upstream-rights checks |
| Excluded | CC-BY-NC, research-only, uncertain, unverified upstream, consent-restricted, credentials/private speech |
| Spanish→English pretrained baseline | `Helsinki-NLP/opus-mt-es-en`, pinned revision `c96e2c5399ebfae4fc43d9669556b9afa74bb69d` (Apache-2.0 model card) |
| English→Spanish pretrained baseline | `Helsinki-NLP/opus-mt-en-es` (Apache-2.0 model card). Needs a separately pinned revision/checksum and its own direction-specific evaluation; never swap ES→EN adapters onto this model |
| Rollback and no-production-impact | Preserve immutable historical adapter `experiments/spanish_fifteen_rounds_20261120/saved/champion_adapter.safetensors` and compatible older `experiments/massive_stage3_20261117/saved/spanish/adapter_selected.safetensors`; no changes to default live translator |

## Audit findings: what is VERIFIED and what is NOT

1. The **actual ES→EN base** is Hugging Face Helsinki-NLP/opus-mt-es-en, encoder–decoder MarianMT with SentencePiece (`source.spm`, `target.spm`). The repo pins base files including a ~312 MB `pytorch_model.bin` with manifest hashes. The existing LoRA adds 294,912 trainable parameters across 36 q/v attention projections (rank 8, scale 2).
2. **Rounds 1–15 were REAL but independent**, each with 240 optimizer steps from the same November 20 contextual warm adapter. They are **not** a 3,600-update serial model. Each reset a new AdamW optimizer and step counter; no optimizer moments, scheduler, data-sampler position, or RNG states were serialized for exact resume. Although underlying model weights were resumed, exact interrupted-run continuation is NOT supported without implementing those states.
3. Best **research-only contextual** candidate by the specified shared development chrF is **round 10**, SHA-256 `557838ab7823d3f7a836b5f3ccfe9a2b585b2b5733ec91a219df0e3967a49954`. Shared 128-example dev: 67.6707 chrF / 47.7607 BLEU. The benchmark choice was among 15 trials so it carries model-selection bias. The separate once-only 256-row FLORES test found 58.5058 chrF / 27.2608 BLEU, slightly BELOW the original base 58.8521 / 27.8629. This does **not** show cross-domain generalization or meaning accuracy.
4. The round-10 contextual checkpoint and its parent use the **Samuel & Audrey subtitle corpus licensed CC BY-NC 4.0**. These weights must remain **noncommercial research only**. Training a new adapter on top with allowed data does not prove that the previous restriction disappears. **Never promote this ancestry into the product.**
5. Best verified current **license-compatible Spanish warm path** is the earlier Amazon MASSIVE stage-three selected adapter (`14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b`), itself built through prior Tatoeba (CC-BY 2.0 FR) and MASSIVE (CC-BY 4.0) runs. *This is a provenance assessment, not a legal warranty.* Use this checkpoint for any newly trained commercial-candidate ES→EN model. The original NC research winner remains preserved for side-by-side research and rollback only.
6. Prior training seeds were recorded and CPU PyTorch deterministic seed setup invoked, but true bitwise reproducibility is not established across library/runner versions or multi-thread execution. No scheduler was recorded; LR remained constant in the inspected trainer. New work must record versions, CPU/RAM, seed, parent SHA, total sample count and output SHA.
7. Prior subtitle alignment checks were **temporal overlap**, not bilingual semantic alignment; some alignment errors may remain. **Do not** treat video timestamp matches or automatic chrF as human-verified translation pairs. MASSIVE is predominantly intent/voice-command utterances, not multi-turn spontaneous slang. This domain gap is material.
8. Tatoeba/ManyThings supplies CC BY 2.0 France text, but individual speakers and source document grouping are not generally available at the sentence-pair level. This prevents claiming its random sentence split is a speaker/document-disjoint conversational holdout.
9. Previous `FLORES-200 devtest`, sampled at 256 examples, has been **consumed**. It must never be silently reused for new checkpoint selection. Previous MASSIVE publisher tests and video holdout are also consumed, and the NC corpus is excluded from all **new** training, augmentation and model selection.
10. No independent real-world bilingual adequacy pass rate, critical error rate, robust measured end-to-end audio latency, reliable E2E GPU resource profile, or full bidirectional model quality baseline is established by existing reports. **Unknown is not zero and not 95%.**
11. Reviewer final holdout is **not created yet**. Because this repository is public, confidential 2,000-per-direction source texts and bilingual judgments must be stored outside GitHub, separately from training corpora. The repo will contain only tooling, schemas and public SHA-256 manifests. No agent should access the sealed final holdout while optimizing.

## Quality gates

Meaning preservation is the prespecified primary outcome. Each direction independently requires general conversations >=95%, context/idiom/regional slang >=90%, noisy transcripts >=88%, critical errors <=1%, with Wilson 95% confidence intervals and full error types. These are *observed* pass-rate thresholds, NOT chrF/BLEU values. Both human raters must work independently, blinded to system identity; a third adjudicates differences. Record reviewer agreement before adjudication.

**Promotion safety:** Require dev bilingual adequacy and subgroup scores from real reviewers (initially unavailable), zero extra critical error failures, <=1 point subgroup regression, measured latency within budget and >noise gain across seeds. chrF/BLEU alone **cannot promote** a model to production. Until enough labels exist, mark trained candidates as `experimental_no_human_gate` and preserve unchanged commercial-candidate warm weights. The production provider is untouched.

## Why the two checkpoint lineages matter

| Lineage | Checkpoint | Data rights status | Usage |
|---|---|---|---|
| Research-only contextual | 15-round winner SHA `557838...` | Depends on CC BY-NC corpus | Archive, research comparisons; never commercial |
| Prior MASSIVE/Tatoeba | Stage-3 SHA `14dddb...` | Apache-2.0 base + CC BY / attribution data | Safe **starting candidate** for new allowed-data experiments, pending full legal/provenance audit |
| Untouched OPUS-MT | Pinned Apache-2.0 base | See model card and downstream rights | Necessary external baseline and product fallback, not a substitute for validated contextual translation |

All future training must identify its **actual** parent adapter checksum and exclude NC data by default. A local step counter beginning at 1 is not a reset of model weights if parent SHA is verified.

## Sources audited

- Original model cards: https://huggingface.co/Helsinki-NLP/opus-mt-es-en and https://huggingface.co/Helsinki-NLP/opus-mt-en-es
- Amazon MASSIVE dataset card CC BY 4.0: https://huggingface.co/datasets/AmazonScience/massive
- Tatoeba/ManyThings CC BY 2.0 FR: https://www.manythings.org/corpus/about.html and https://tatoeba.org/en/downloads
- Samuel & Audrey creator transcript corpus CC BY-NC 4.0: https://doi.org/10.5281/zenodo.18665315
- Previously run raw evidence: `experiments/spanish_fifteen_rounds_20261120/saved/summary.json`, `experiments/massive_stage3_20261117/saved/spanish/report.json`, `research/spanish_flores_generalization_20261120/saved/flores_external_evaluation.json`
- Scientific evidence and experiments: `docs/papers.md`

**Outstanding human-only dependencies:** Independent bilingual annotation and sourcing/rights verification for a representative final holdout, category coverage including real slang, and product authorization for any eventual commercial promotion.
