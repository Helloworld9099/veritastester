# BridgeCaption — ES↔EN v2 executed progress (2026-11-20)

**Branch:** `train/es-en-v2`  ·  **PR:** [#42](https://github.com/Helloworld9099/veritastester/pull/42) (draft; no main merge)  ·  **Cost:** $0 paid compute/APIs.

**Core rule:** Human meaning adequacy is **NOT** established. The original NC-trained adapter remains noncommercial research; the new compliant candidate is distinct, incremental and NOT production approved. Live Gemini/ASR/UI/updater defaults remain unchanged.

## What was really executed

| Evidence | Real execution and result | What it does **not** prove |
|---|---|---|
| Historical NC research | 15 separate 240-update ES→EN runs, 3,600 total independently warm-started optimizer updates; selected round 10 research SHA `557838ab...`; shared synthetic subtitling dev 67.6707 chrF | NOT 15 serial rounds, NOT 95% accuracy, NOT commercial-use rights |
| V2 allowed-license continuation | [Run 44806165157](https://github.com/Helloworld9099/veritastester/actions/runs/44806165157): 3 serial 64-step CPU runs (192 *new* optimizer updates); actual adapter parent SHA `14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b`; round 1 and 2 accepted on dev-only automated similarity, round 3 rejected | NOT a proven improvement on independently human-rated slang/context; NOT production promotion |
| Preserved three-round model | [Recovery run 44806915830](https://github.com/Helloworld9099/veritastester/actions/runs/44806915830): verified all 3 original reports and intermediate candidate weights; reconstructed and committed `experiments/es_en_v2_runs`; accepted v2 research candidate SHA `3bf849bb9a4377dcb9f937cdc7750a7f26616fc1c58cf703cef0ba8e46430a71` | NOT a fourth run; does not replace product weights |
| ES→EN CPU benchmark | Original Marian p95 0.339s vs licensed MASSIVE Stage3 adapter p95 0.367s on 16 published synthetic smoke phrases; 5.12s model load, peak 792,504 KiB (approx 774 MiB) | NOT full mic→ASR→caption latency, not long session, not sustained cross-platform |
| EN→ES baseline | [Run 44807166625](https://github.com/Helloworld9099/veritastester/actions/runs/44807166625): original pinned Marian EN→ES, 128 MASSIVE validation examples: 60.2332 chrF / 32.4825 BLEU, synthetic prompt p95 0.291s, 3.30s model load | No trained EN→ES adapter. This is a deliberate original model control, not a training round |
| Dev error analysis | [Run 44807405563](https://github.com/Helloworld9099/veritastester/actions/runs/44807405563): 128 official MASSIVE validation examples; new candidate changed 11 outputs, improved chrF on 8, degraded on 2, and five numeric flags remain | Automatic dev risk tags are **suspicions**, not confirmed semantic correctness |
| Authenticity/slang stress | [Run 44808001708](https://github.com/Helloworld9099/veritastester/actions/runs/44808001708): 40 **agent-authored synthetic development** ES↔EN examples. ES→EN original base 47.37 chrF; v2 isolated 40.54; v2 with raw past text concatenated 28.79. EN→ES base 17.52. Regional `guagua` and contextual `I'm down` were interpreted literally; past speech sometimes appeared in current output | Tiny, artificial, multiple valid phrasings: scores are NOT accuracy and should not be used for final pass |
| Actual BridgeCaption integration | [Run 44808274643](https://github.com/Helloworld9099/veritastester/actions/runs/44808274643): 6 actual research-local text Pipeline calls, cached model, correct source-only behavior, verified rollback loading, wrong-direction and untrained-context rejection, warm p95 0.249s, peak 1,115,812 KiB | NOT product activation, not real spoken input, no full ASR test, and no verified context-aware local model |
| Bilingual BabyLM source audit | [Run 44807245908](https://github.com/Helloworld9099/veritastester/actions/runs/44807245908): pinned real 8MiB prefix; 1,500 MIT synthetic English/Spanish dialogue groups; 1,453 structurally aligned candidates (12,288 turns); 47 rejected; data includes previous speaker turns and code-switching variants | **ZERO** independently human-verified pairs and zero newly trained model updates. Original text was synthetic GPT-4 generated. |
| DRAL original CC0 source audit | [Run 44808690720](https://github.com/Helloworld9099/veritastester/actions/runs/44808690720): HTTP 206, first 16MiB of original 11GB audio tar; 22 members scanned, 18 audio, zero text transcript metadata accessible in this prefix | No transcripts acquired. LDC release terms differ; no source rights or speaker consent assessment from audio. |

### Three *new* v2 real training rounds — shared publisher-development comparison

| Round | Hypothesis | Development chrF | BLEU | Stage result |
|---|---|---:|---:|---|
| Stage3 parent | Original allowed-license trained prior | 64.8557 | 42.6359 | Verified warm-start |
| 1 | Conservative quality screening | 65.2273 | 43.2414 | Dev-only accept |
| 2 | Stricter sentence alignment screen | **65.8532** | **43.9552** | Dev-only accept, current eligible v2 research checkpoint |
| 3 | ASR-style punctuation stripping | 65.7410 at final step | 43.6660 at final step | Dev-only reject, retain round 2 |

The new v2 controller explicitly **blocks product promotion without real blinded bilingual development labels**, no subgroup degradation, critical error control, model/rights audit and measured resource limits. The acceptance of rounds 1–2 means only *further research warm start*, not validated meaning accuracy.

## Code and artifacts

- `docs/TRAINING_V2_PHASE0.md`: resource, legal and target constraints and actual audit.
- `docs/papers.md`: research papers linked to specific experiments/risks.
- `research/es_en_v2/quality.py`: legal/provenance, exact group leakage and PII/numeric heuristics.
- `research/es_en_v2/review_packet.py`: private sealed final-holdout digest, reviewer blinding, explicit human labels, Wilson CI helper. No agent scoring.
- `research/es_en_v2/HUMAN_REVIEW_PROTOCOL.md`: human quality rubric, categories, critical error rules, adjudication and uncertainty.
- `research/es_en_v2/dev_mining.py` and `mine_dev_run.py`: dev-only hard-case ranking, actual aggregate model diagnostics.
- `research/es_en_v2/baby_dialogue.py` and `inspect_babylm_stream.py`: speaker/group-aware candidate parsing with explicit synthetic quarantine and real source receipt.
- `research/es_en_v2/controller.py`: conservative promotion decision and immutable parent-checkpoint ledger.
- `experiments/es_en_v2_runs/`: complete real optimizer reports, source hashes, all candidate checkpoints, selected licensed v2 adapter, rollback history.
- `research/es_en_v2/benchmarks/`: real ES→EN and EN→ES CPU baseline data.
- `research/es_en_v2/dev_stress_slang_context.jsonl`, `stress_eval.py` and `context_stress_receipt/`: actual developer-authored slang and context failure diagnostics.
- `research/es_en_v2/integration.py` and `integration_receipt/`: opt-in local research translator integrated with actual `Pipeline`; correctly advertises **no trained contextual support** and no provider-default changes.
- `research/es_en_v2/tests/` and `.github/workflows/es-en-v2-*.yml`: reproducible isolated checks and $0 CPU jobs.

## Why 95% cannot currently be claimed or automatically reached

1. **Independent bilingual reviewers do not exist in the available tools.** No user-supplied real human labels, calibration panel or third adjudicator has been provided. An AI judge is not equivalent to those humans.
2. **No legally sourced and independently secured final 2,000-per-direction conversational holdout exists.** It must be group/speaker/document-disjoint, kept outside public GitHub, sealed and accessed exactly once by independent custodians after freezing models.
3. **The only readily accessible new context-rich corpus is mostly synthetic and not independently meaning-verified.** The accepted production-candidate source is MASSIVE, principally isolated short intent utterances. Running 12 more nearly identical MASSIVE hyperparameter sweeps would be wasteful and would not directly teach regional slang/context.
4. **Real ablation shows the local model is not context aware.** Blindly injecting history caused it to translate earlier utterances. Research-only integration intentionally rejects it.
5. The expensive DRAL archive is primarily re-enacted bilingual **audio**, with no text transcripts in the inspected 16MiB; publication technical report emphasizes phrase timings/metadata, not a ready parallel text transcript corpus. Downloading 11GB merely to obtain untranscribed audio cannot by itself solve the text translation training gap. Source: https://cs.utep.edu/nigel/dral/ and https://www.cs.utep.edu/nigel/papers/dral-techreport2.pdf .

## Required external human input to proceed safely to the 95% claim

- Obtain licensed, consented, independently verified bilingual *multi-turn* examples with context and regional slang. For a pilot, bilingual reviewers could validate ~200–500 sampled MIT synthetic pairs and annotate authentic dialect/slang cases before use as trustworthy quality controls; synthetic text must remain labeled synthetic.
- Source 2,000 independent authentic conversation examples **per direction** (target), with group/speaker/document IDs, license/consent and private custody. Do not reuse earlier published dev cases, old FLORES subset, or prior evaluated NC video test groups.
- Arrange two independent fluent bilingual raters and a third adjudicator to apply `HUMAN_REVIEW_PROTOCOL.md` once to the final sealed holdout. The agent must **never** perform the final independent human scoring.

**Bottom line:** Three new serial training rounds completed, with meaningful but limited general-MASSIVE development gains, no semantic 95% claim, and confirmed slang/context failure in actual model outputs. Free compute is not the primary bottleneck now: independently verified conversational targets and human assessment are. Do not inflate completed rounds by counting failed attempts, inference benchmarks or source inspections.
