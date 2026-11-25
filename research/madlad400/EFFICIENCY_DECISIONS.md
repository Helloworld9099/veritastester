# MADLAD-400 optimization program — do not reward repetitive optimizer steps

**State:** experimental 425-native-target language research branch; existing BridgeCaption default is unchanged. **No paid compute / API.**

## Measured starting point

- Verified pinned Google `google/madlad400-3b-mt`, revision `fa184c675da0b5c9e1c8694fccd4e12e2d422094`, model card Apache 2.0. [Actual tokenizer audit #44972552608](https://github.com/Helloworld9099/veritastester/actions/runs/44972552608) **passed** and found **425 actual <2language> target tags**. This is tokenizer reach, not independently verified meaning.
- The previous single M2M100 research model is **not discarded**; its immutable 38-update adapter is SHA `a00c7965211853df07ce3fa032a76283dfc984dcdaeac85e03263445029ffb8e`. Original bilingual Marian specialists are untouched.
- The last M2M100 continuation conducted **72 real optimizer updates**, yet **0 of its 24-, 48-, 72-step checkpoints qualified**; 48 and 72 regressed in French-related development routes. [Actual anti-stagnation controller #44972803120](https://github.com/Helloworld9099/veritastester/actions/runs/44972803120) ran and saved exact prior evidence in `optimization_analysis.json`. Do not repeat that dataset/step schedule hoping for free quality.

## Active controlled experiments

| Technique | Real script/run | What it establishes | What it cannot establish |
|---|---|---|---|
| **Pretrained MADLAD actual 3B LoRA** | `train.py` / [training Actions](https://github.com/Helloworld9099/veritastester/actions/workflows/real-madlad400-training.yml) | Whether free BF16 CPU hosted runner can load real 3B translation weights, perform a backward pass on permitted MASSIVE TRAIN, save a genuine shared adapter and measure development-only inference. Training triggered once per explicit run request (ordinary file edits no longer automatically retrain). | Three optimizer updates and one DEV example per route prove **nothing** about semantic adequacy. |
| **INT8 vs full-precision approach** | `quantized_ablation.py` / [INT8 Actions](https://github.com/Helloworld9099/veritastester/actions/workflows/madlad400-quantized-benchmark.yml) | Third-party CT2 INT8 pretrained conversion, pinned HF SHA/license/file hashes, on-CPU one-model load, beam-1 vs beam-2 speed/chrF/BLEU and six main directions plus German/Portuguese/Arabic/Chinese output smoke. | Not training, no independent human labels, third-party conversion parity unverified. |
| **Real data-selection A/B** | `data_ablation.py` / [data actions](https://github.com/Helloworld9099/veritastester/actions/workflows/madlad400-data-ablation.yml) | Compare two reproducible source-selection methods on **three actual licensed Tatoeba archives** (350 pairs each), counting authentic cases involving negation, time/numbers, pronouns, slang, register and missing clauses. | Better source coverage ≠ higher translation accuracy; flagged hard cases need bilingual checks before training. |
| **Early-stop adaptive decision logic** | `optimization_controller.py` / verified [policy run](https://github.com/Helloworld9099/veritastester/actions/runs/44972803120) | No promotion based on n=1, a bad direction, a newly introduced critical error, excessive p95 or RAM; 10-route readiness includes 6 ES/FR/EN + en→de, en→pt, en→ar, en→zh. | The four expansion sentinels still lack proper independent references, so no global model may be released based on the existing development set. |

**Caution:** The first INT8 run failed in the preflight due to an incorrect assumed converted-model filename; the actual `cstr/madlad400-3b-ct2-int8` repo has `spiece.model` and `shared_vocabulary.json`, **not** `sentencepiece.model`. This is corrected, with a separate exact-revision retry. That failed preflight is not counted as a translation-quality experiment. Any other CT2 runtime problems must be diagnosed rather than suppressed.

## Why this direction is efficient

1. **Pretrained baseline before new training:** MADLAD already has translation knowledge in 425 tagged targets, and does not need to relearn every language from scratch. The old M2M100 adapter is architecturally incompatible and is preserved, not misapplied.
2. **CPU INT8 deployment first:** CTranslate2 explicitly supports MADLAD-400 and recommends CPU INT8 with threading, batch bucketing, max batch size and beam-size tuning. It is inference-only, not a faster training algorithm. [OpenNMT docs](https://opennmt.net/CTranslate2/performance.html), [supported model types](https://github.com/OpenNMT/CTranslate2).
3. **LoRA is not magical protection:** Liu & Niehues (MRL 2025), [Conditions for Catastrophic Forgetting in Multilingual Translation](https://aclanthology.org/2025.mrl-main.23/), found PEFT had no consistent advantage against forgetting across their study. Validate unseen language directions as well as trained directions.
4. **Sample informative examples, not more duplicated short commands:** Prior reviewer feedback identified literal slang, incorrectly assigned pronouns, missing first/last clauses, mistranslated time and register inconsistencies. A controlled source-selection experiment now measures how much those cases occur in truly licensed bilingual parallel corpora.
5. **Separate data, weight and release gates:** A better chrF/BLEU or lower p95 is necessary evidence, not sufficient human adequacy. Only after validated bilingual past-turn contexts and independent judgments should we expand contextual adaptation.
6. **Keep the cost budget:** No paid training API, remote paid GPUs, QLoRA with an unsupported CPU runtime, unclear-license commercial model, or uncontrolled huge data dump. Large training should be scheduled only via explicit new `run_request.json` to avoid redundant jobs.

## Future methods *conditional on real evidence*

- **Method A:** If pretrained INT8 meets target p95 and dominates the existing ES/FR development baseline, skip LoRA adaptation altogether until hard-case evidence justifies it.
- **Method B:** If model is fast but slang/meaning weak, collect license-approved, independently verified regional slang/context pairs. Compare current-turn vs past-only 1–4-turn conditioning on conversation-grouped development, not a machine-invented translation target.
- **Method C:** If narrow fine-tuning regresses any of the 10 directions, change sampling/replay ratio or LoRA placement (encoder+decoder vs last-decoder) with a matched compute budget; never repeat the same failing configuration.
- **Method D:** Compare quantized CTranslate2 beam=1 against beam=2 using measured p95 and semantic-critical probes, and keep the less costly decoder when quality holds.
- **Method E:** Consider verified licensed teacher-student distillation into a smaller faster model only when the teacher actually outperforms earlier specialists on human review.
- **Method F:** If free CPU cannot train the 3B backbone within the budget, label it hardware-blocked rather than claiming success or provisioning paid infrastructure.

## Progress is judged on verified deliverables

Every serious change must produce a pinned exact model SHA/revision, real training or inference logs, dataset license/source hashes, split integrity and method comparison, per-route output, load/latency/RAM metrics, no regression across existing specialists, and no final holdout contamination. Native model tag count cannot substitute for independently verified 95% meaning accuracy, and production BridgeCaption continues using its existing fallback until release checks pass.
