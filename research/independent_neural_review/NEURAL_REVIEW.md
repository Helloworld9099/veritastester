# Independent review of neural translation pilots

2026-11-16. Reviewed the Spanish and French LoRA pilots and delivery archive. Original training files/checkpoints were not changed. No retraining, checkpoint reselection, deployment, push or paid service occurred.

## Conclusion

The artifacts and code support the claim that **real neural translation adapters were trained**, rather than only idiom classifiers. The saved metrics are accurate, and sampled checkpoint inference is reproducible. **Do not promote Spanish. Preserve French as a promising in-domain research checkpoint, not a verified improvement for live conversation.** No major evaluation-integrity defect was found in the completed pilots. Publication/integration readiness is not established.

## Independently verified

- Both pilots contain 1,024 training, 128 development and 256 test rows. NFKC/casefold/whitespace normalization finds zero cross-split source or reference overlap and zero block/component overlap.
- Base model file SHA256 values match their saved manifests. Each selected adapter exactly matches its step-128 checkpoint.
- Inspected training code: backbone parameters are frozen before injecting rank-8 query/value LoRA modules; optimizer receives only the new trainable parameters. No gold test/reference field is fed as an inference input. Targets are appropriately used as supervised training labels and development NLL labels.
- Both logs contain sequential steps 1–128 with finite losses/gradient norms. Both step-128 selections agree with the highest development chrF among baseline, 64 and 128. Selection is saved before test-generation code. I did not rerun full training, and logs/code consistency is not an independent observation of every historical optimizer step.
- Both adapters have 72 tensors, 294,912 parameters and 36 nonzero LoRA-B tensors. Newly created zero-output adapters exactly preserve base logits.
- In separate fresh processes, independently loaded each saved selected adapter and regenerated the FIRST EIGHT examples of each of its two evaluation sets, both baseline and adapter. **64/64 outputs match saved strings exactly**, using the same beam/batch/length settings. This verifies fixed checkpoint replay, not bitwise reproduction of a new training run.
- No training/dev/test source exceeds the 96-token training or 128-token evaluation source cap; no reference exceeds the 96-token training target cap in the sampled partitions.
- Recomputed corpus BLEU and chrF from ALL saved evaluation outputs. Every value matches the training reports within 1e-10.
- Recreated French train/dev/test random samples directly from the parent handoff files: all selected case IDs, source strings and reference strings reproduce. Parent handoff digest and each input-file digest match.
- Recreated the French dialogue guard selection: all 128 case IDs match. Its 64 dialogues are disjoint from the 40 previously examined smoke dialogues. French pilot sources AND references have no exact normalized overlap with the entire supplied DiaBLa French-English inputs/references.
- Spanish WMT guard excludes the earlier independent audit's 200 indices and has zero normalized source or reference overlap with this Europarl pilot. Spanish block grouping is not document/speaker grouping. The nearest selected cross-split source lines are 47 lines apart; no immediately adjacent selected rows cross a split.
- Delivery ZIP/tar SHA256: `b4a52197012e7861c44251495180b9151cee5108a1ff00d30788c92cb279777f`. All 33 delivery-manifest entries match archived bytes; paths are safe; no raw JSONL examples are included.

## Exact metric check

| Evaluation | N | Base chrF | Adapter chrF | Base BLEU | Adapter BLEU |
|---|---:|---:|---:|---:|---:|
| Spanish Europarl | 256 | 66.327950 | 66.204124 | 43.959220 | 44.095750 |
| Spanish WMT13 guard | 128 | 59.744659 | 59.402919 | 36.479811 | 35.735649 |
| French Tatoeba | 256 | 74.244751 | 76.379676 | 61.874715 | 66.455349 |
| French DiaBLa guard | 128 | 54.928233 | 55.428430 | 31.174138 | 30.773612 |

French gains +2.135 corpus chrF / +4.581 BLEU on this Tatoeba sample. Dialogue changes are mixed (+0.500 corpus chrF / -0.401 BLEU). The provided bootstrap is for MEAN SENTENCE chrF, not corpus chrF: its dialogue interval crosses zero and cannot establish a dialogue gain. The report correctly distinguishes these aggregations; do not quote its interval as an interval on corpus chrF. BLEU/chrF are not percentages of correct translations.

## Limitations and useful corrections

1. **No contextual translation was trained or evaluated.** Every inference request contains only the current source sentence. Isolated Tatoeba/Europarl pairs cannot establish helpful-memory selection, exact slang meaning, conversational continuity, speech accents, ASR or live-caption latency.
2. **Pretraining contamination is unknown.** All partitions are held out from OUR LoRA fitting only. OPUS bases may have seen these corpora or benchmarks. Do not call them wholly unseen to the base models. French component splitting prevents exact alternate-translation leakage, not author/topic/template/paraphrase overlap. DiaBLa groups are dialogues, not a verified speaker-disjoint population.
3. **Europarl wording should remain precise.** A 256-line block keeps rows inside that block together, but cannot generally guarantee that adjacent rows at block boundaries stay together. No immediate cross-split neighbor occurred among these actual sampled rows, but it is not a universal property of the grouping algorithm. The official Europarl page recommends Q4 2000 for standardized test reporting; this pilot uses a custom random-block split, not that official test protocol.
4. **Portable reproduction is incomplete.** The archive preserves selected adapters, code, reports and hashes but omits base weights, dependencies, raw/sampled data, evaluation-output JSONL, WMT guard indices and the French data-preparation handoff. Scripts depend on sibling workspace directories. They replay in this workspace, but the tarball is NOT a self-contained install/rebuild package. Add an exact dependency lock, pinned download/preparation instructions, sample-selection manifests (including `wmt_guard_selection.json`) and references to required companion handoffs before describing clean-machine reproducibility. Do not redistribute raw sentences simply to solve this.
5. **No resumable optimizer state.** The selected adapter supports inference and a fresh training restart, but no optimizer/RNG checkpoint was supplied for exact mid-run continuation. Full-from-scratch deterministic training was not independently repeated. Keep this separate from the demonstrated exact inference reload.
6. **Inference speed is not improved by the reported timing.** Baseline and adapted runs shared CPU/cache/order effects, and times are batch-amortized. No controlled latency win was demonstrated. Any conversion/merge/quantization must be re-evaluated.

## Licensing review

This review checks publisher notices and provenance; it is not legal clearance for model redistribution.

- Both [Spanish](https://huggingface.co/Helsinki-NLP/opus-mt-es-en) and [French model cards](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en) identify Apache-2.0. The bundle carries model cards and pinned identifiers. Preserve applicable LICENSE/NOTICE and modification attribution when publishing a derivative; metadata alone does not settle all downstream dataset obligations.
- The [Europarl publisher](https://www.statmt.org/europarl/) says it is unaware of copyright restrictions, asks research users to contact the maintainer, and requests the Koehn 2005 citation. The report's internal-research qualification is appropriate; do not turn this into blanket commercial clearance. No email was sent by this review.
- The old Tatoeba mirror notice says CC BY 2.0 and credits Tatoeba/ManyThings, but lacks individual author identifiers. [Tatoeba's current terms](https://tatoeba.org/en/terms_of_use) emphasize attribution to the sentence author. The report correctly preserves that unresolved provenance caveat. Local-only use is not itself a blanket license exemption, and this audit cannot certify trained-artifact publication rights.
- [DiaBLa's original repository](https://github.com/rbawden/DiaBLa-dataset) identifies CC BY-SA 4.0 and Bawden et al. (2021). It is evaluation-only here, with no raw sentences in the supplied delivery archive. Retain attribution and applicable share-alike terms if distributing dataset-derived inputs/references/outputs; this does not automatically mean an independently trained model has inherited that dataset's license.

## Recommendation

Keep both adapters and guards frozen. Do not retrain or choose a checkpoint based on these consumed test results. For the next study, pre-register a new domain development set and an untouched dialogue holdout, acquire appropriately attributed conversational TRAINING data, and use bilingual human review of errors. Keep existing idiom/Portuguese research hints inactive. Any optional local provider integration should be separately tested without claiming unsupported language or context capabilities.

## Evidence and commands

- `saved_evidence.json`: all split/hash/metric checks.
- `checkpoint_replay_es.json`, `checkpoint_replay_fr.json`: fixed-prefix checkpoint replay and truncation counts.
- `check_saved_evidence.py`, `replay_checkpoints.py`: reproducible audit scripts.

From `/workspace/scratch/2b30092347a7/veritastester-independent-validation`:

```
python neural_review/check_saved_evidence.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python neural_review/replay_checkpoints.py es
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python neural_review/replay_checkpoints.py fr
```

Use separate fresh processes: the original pilot modules initialize PyTorch interop threading once, so importing both after parallel work in the same process raises a thread-initialization exception. This is a replay-harness constraint, not evidence that the trained artifacts fail.
