# Independent frozen continuation audit — 2026-11-16

The delivered French and Spanish continuation adapters do **not show a clear improvement over their unchanged bases**. Both original chrF confidence intervals include zero; the provenance-filtered French sensitivity also includes zero. This is absence of demonstrated benefit, not proof of equivalence or proof of harm. Keep the adapters inactive/research only. No application, repository, weights, training selection, or deployment was changed.

**Final concurrent-experiment check:** The original French 128-component cohort was unused against the preregistered earlier-run inventory, but a separately finished concurrent follow-up test shares **five components** (three exact sources, four exact references). They are in its test/test-output and duplicate portability-test files, with **zero train or dev overlap**. The original delivered adapters remained fixed and did not train on these follow-up examples, so this is overlapping evaluation material, not evidence of training leakage into the evaluated adapters. Still, the full French cohort must **not** be called unused across all now-completed runs. Spanish has zero source/reference/group overlap with the follow-up. No original result or row was replaced. After fixing a provenance-only exclusion list, I calculated a clearly **post-hoc** sensitivity for the remaining 123 French components using existing outputs, without new inference. That subset has zero follow-up overlap. Its registration was saved before computing its subset scores, but after the original inference and aggregate results; it is **not preregistered validation**.

## Frozen design and actual execution

The sample/protocol registration was saved at 02:28:36 UTC, SHA256 `9eadf5a44b16e0c9d84aafa592c3cfb0807cbece8d9589eb9ab92c4906e9c2b0`, before neural imports/inference. Inference started at 02:30:10 UTC and completed at 02:33:54 UTC. No fitting, checkpoint selection, new labels, paid APIs, or network access was used. A separate training experiment's selections/results were not inspected.

- French: 128 distinct, previously unused Tatoeba translation components from the existing French-to-English test partition, one pair per component. Length filters: source 3–32 whitespace words, reference 3–36. There were 13,522 eligible pairs before within-sample deduplication. Selection seed 202611160427.
- Spanish: 128 unused WMT13 Spanish-to-English news rows, 2,156 eligible before within-sample deduplication. Selection seed 202611160428. These are distinct rows, **not verified independent documents or speakers**.
- NFKC/casefold/whitespace normalized exact sources and references are unique within each selected cohort. Prior train/dev/test, pilot outputs, guards, old independent revalidation, initial WMT200 evaluation, synthetic diagnostic probes, complete local DiaBLa and existing context-dialogue train/dev reservations were excluded. Forty registered source artifact hashes were rechecked unchanged. Registration contains the exact inventory.
- All French connected-component IDs and Spanish WMT row IDs were checked against prior consumed groups/rows. Source/source and reference/reference overlaps were zero; the independent post-inference verifier additionally found zero cross-side source/reference exact overlaps. Membership in the original source datasets was verified for every sampled pair. Post-freeze supplemental checks against all string values in auxiliary synthetic/topic/SemEval data and the original smoke artifact found zero overlaps. No post-inference removal or replacement occurred.
- This establishes unused status against the registered earlier-run inventory, excluding the concurrent follow-up addressed above. It does not establish absence from upstream model pretraining, semantic/near-duplicate independence, independence of Tatoeba authors/templates, or an exhaustive history outside the available artifacts.
- The complete 144 local DiaBLa dialogue groups are already consumed. **No fresh dialogue test was claimed or run.** Existing WMT24 conversational train/dev reservations were excluded, not repurposed as test data.

Both arms used the same Marian model/tokenizer, CPU, single intra/inter-op thread, beam 4, batch 8, source cap 128 tokens and maximum 96 newly generated tokens. Rank-8/alpha-16 q/v LoRA was enabled only in the selected arm and disabled for the base arm; all parameters were frozen. Original seven-file base manifests in each language were verified before inference and again afterwards. French used the existing pinned safetensors base; Spanish used the existing pinned PyTorch base with weights-only loading. Offline flags were enabled. Times are sequential wall-clock observations, not a controlled performance benchmark.

## Aggregate results

Delta means selected minus base. chrF is the preregistered primary statistic; BLEU and its intervals are supplementary. Scores are corpus scores, not averages of sentence scores.

| Cohort | Metric | Base | Selected | Delta | Paired 95% percentile interval |
|---|---|---:|---:|---:|---:|
| French, 128 components | chrF | 76.5693 | 76.4416 | -0.1277 | [-1.9045, 1.5167] |
| French, 128 components | BLEU | 64.5407 | 64.8801 | +0.3394 | [-2.4299, 3.2538] |
| Spanish, 128 rows | chrF | 61.1499 | 61.1972 | +0.0473 | [-0.3783, 0.4962] |
| Spanish, 128 rows | BLEU | 35.2984 | 35.2267 | -0.0717 | [-0.8883, 0.6916] |

### Post-hoc French sensitivity excluding the five concurrent-test groups

| Cohort | Metric | Base | Selected | Delta | Paired 95% percentile interval |
|---|---|---:|---:|---:|---:|
| French, 123 retained components | chrF | 75.9864 | 76.1691 | +0.1828 | [-1.5778, 1.9218] |
| French, 123 retained components | BLEU | 63.7143 | 64.6220 | +0.9076 | [-1.8068, 3.9766] |

The exclusion rule used only source/reference/group provenance, not model outputs or per-case scores. `followup_overlap.json` records the combined overlap and follow-up artifact hashes. `sensitivity_registration.json` freezes the 123-row selection hash and rule; `sensitivity_results.json` contains the exact scores. French sensitivity sample SHA256: `5b39e0db344b01f1a2c9dacbdd0e6569734edddd131ed72db66b1141a17fc20d`. This preserves strict cross-run uniqueness for the sensitivity subset while disclosing the post-hoc adjustment. It does not restore the original full cohort's cross-run-unused status or establish dialogue quality. The full original 128-row results above remain available unchanged.

Intervals use 2,000 paired bootstrap replicates, resampling the same row indices for both arms. French has one sentence per connected component, so this is component resampling. Spanish resamples rows; unknown document clustering can make its interval optimistic. Bootstrap seed 202611160427 for each language. Intervals are conditional on this sample/reference/model/decoding protocol, with no multiple-comparison adjustment. No calibrated equivalence margin or power analysis was specified.

SacreBLEU 2.6.0 signatures: chrF `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`; BLEU `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`. Python 3.12.14, Torch 2.7.1+cpu. The verifier recomputed both public metric scores and both bootstrap intervals to 1e-12 from the private output artifacts; cached additive statistics were checked against public corpus scoring.

No sampled source exceeded the source cap; no output reached the generation cap; no output was empty. Selected outputs differed from base on 25/128 French pairs and 33/128 Spanish pairs. This checks only gross generation failures; no independent human adequacy, hallucination, contextual-meaning, ASR, safety, or live-UI assessment was performed. French is in-domain sentence translation; Spanish is external news-domain translation. Neither establishes conversation readiness.

## Immutable models and evidence

- French adapter SHA256: `600990e205c1238ea1fba3a8c36e5ef199bdfc10b33e4e0068e5b9695b7e8665`.
- Spanish adapter SHA256: `55d4b872a6807e0d9bc317b72ef63eddf77d5828c427f91497d5bc23ac2e65f3`.
- French base weight SHA256: `6e3837f34b903802c3d0d670362b997cee6e87584a1108eb3fa89e4625e4424a`.
- Spanish base weight SHA256: `586a4e69e0804459ac7c176a5dc76852652100d9af1ee0db796de719899a5deb`.
- French frozen sample SHA256: `4704c455e773d1c866ad8d829ca7f5fe6360e95408c3a6f754967b7f9e00613e`.
- Spanish frozen sample SHA256: `b5ece888422560bfda43bec2a75d12d4c3ea904f42cfdf429f824f587fd5b3ab`.
- French private output SHA256: `8945ac4446c825736920b87e4d846fae971c51b7c695f79aa48ccf22dfdf33cd`.
- Spanish private output SHA256: `1f5937f1061649b9d43cce98430c07f50d81b7555629a4196141e7b78312fae8`.

`registration.json`, `results.json`, `verification.json`, and `additional_overlap.json` contain inspectable earlier-inventory aggregate/hash evidence; `followup_overlap.json` and the sensitivity files provide the final concurrent-run qualification. The source scripts are included. `prepare_fresh.py` must not be rerun over this frozen directory: it writes new selection and registration artifacts. For historical replay, use a separate empty audit directory with the same local source/base/checkpoint files and dependencies; the recorded hashes are authoritative. `evaluate_frozen.py` refuses to overwrite existing results. `verify_frozen.py` recomputes saved metrics and exclusions without neural inference. No dependency, base weight, adapter, dataset text, selected case list, or model output is in the handoff archive.

Private samples and outputs remain under `private/` (directory 0700, files 0600) solely for local replay and consumed-case tracking. They are now consumed evaluation material: exclude them from future training, development, tuning and any claimed fresh evaluation. Do not expose their memberships or results to the currently bounded training task before it freezes its own experiment. Parent owns any subsequent research-only delivery; no push was performed here.

## Provenance and remaining limits

Tatoeba is the existing 2019 ManyThings/Anki snapshot via the TensorFlow mirror, with a CC BY 2.0 notice crediting Tatoeba contributors and ManyThings. Per-contributor IDs are absent in that mirror. The existing attribution/redistribution caveat remains unresolved; this report neither publishes sentences nor establishes commercial clearance for derived weights. WMT13 source/reference data are the existing official SacreBLEU test adapter files and were used only locally for evaluation; no redistribution rights are inferred. Base model cards state Apache-2.0, but that does not clear every dataset right. This audit introduces no new model publication.

No execution blocker remains for this completed comparison. The original French cohort fails the strict unused-across-all-now-completed-runs requirement for five concurrent-test groups; only its labeled post-hoc 123-group sensitivity satisfies that broader provenance filter. The material blockers to a stronger readiness claim are lack of demonstrated fresh-cohort benefit, no untouched local dialogue cohort, limited sample size/domain coverage, unverified source-document clustering and pretraining exposure, and unresolved historical attribution restrictions. This audit does not alter the prior research-only delivery decision.
