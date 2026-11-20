# ES↔EN meaning-preservation evaluation — independent bilingual protocol v2

**Status:** protocol/code implemented; independent reviewer labels **not collected**. No automatic metric represents a 95% pass rate. The host application must not publish a numeric semantic-quality claim before the human handoff.

## Freeze boundaries

- Before freeze: training and development datasets may be opened by approved scripts. Development failure mining, synthetic generation and prompt/model selection are allowed on **dev only**, subject to rights.
- After freeze: select exactly one ES→EN and one EN→ES model identity and hash, quantization/beam policy, context serialization scheme, inference hardware, and latency protocol. Save a timestamped signed model card and exact SHA-256.
- Final holdout: **2,000 or more licensed/permissioned, independently sourced, naturally authored examples PER DIRECTION** if feasible. Group by speaker, dialogue and document; hold back whole groups, not isolated utterances. No training script may load its files. A private custodian (not a tuning agent) controls all access and keeps a timestamped access log. All translations for all frozen systems are generated **once** by the custodian after freeze, before reviewer blinding.
- Stratification: general conversation, context-dependent meaning, idioms, slang and regional variants, technical/domain terminology, noisy ASR input, code-switching and partial speech. Record actual language/dialect, source region, speaker and domain coverage. A single sentence may have multiple category labels, but the analysis plan must predefine denominators to avoid double-counting.
- For test provenance, verify rights and source metadata. Reject synthetic test examples, benchmark examples previously displayed in prompts, training-set near duplicates, consent-restricted sources, and suspected pretraining memorization when evidence exists. High-volume dedup requires indexed retrieval; the included small-corpus heuristic is not an exhaustive global semantic check.
- Public GitHub contains only **code, schema and hashes**, never final test sentences, references, private reviewers or system keys.

## Human meaning-preservation rubric

The two fluent, qualified reviewers **independently** read the source with only the *preceding* dialogue, visible topic and regional metadata; then judge the translated current utterance without knowing system name. Reviewers must ignore literal surface matching when equivalent meaning is conveyed.

**Pass** if intended meaning, pragmatic force, naturalness and crucial factual details are retained **without important unsupported additions or omissions**. Equivalent idiomatic phrasing is allowed. **Fail** if the speaker's intent is changed, a relevant idiom/slang sense is mistaken, an important fact is lost, the wrong person is referenced, or the translation adds false information. Mark **unrateable** if the source is ambiguous without adequate context, the reference is unreliable, or audio/transcription/source rights prevent reliable assessment. Publish unrateables and their counts; do not hide them to inflate pass percentages.

### Worked examples (illustrative only; NOT final holdout)

| Source + context | Passing translation | Failing translation | Why |
|---|---|---|---|
| ES: `Está brutal.` Earlier: speaker excited about a great concert. | EN: `That was amazing.` | EN: `That was violent.` | Regional enthusiastic slang in clear positive context |
| ES: `Ahorita voy.` Friend has asked speaker to hurry up. | EN: `I'll be there shortly.` | EN: `I'm going right now, this instant.` | Regional pragmatic timing; if uncertain, avoid invented exact timing |
| ES: `Me estás tomando el pelo.` Earlier: joke about false admission. | EN: `You're pulling my leg.` | EN: `You're pulling my hair.` | Idiomatic meaning |
| EN: `I'm down.` Earlier: `Want to play after class?` | ES: `Me apunto.` | ES: `Estoy deprimido.` | Context-dependent willingness |
| EN: `I'm down.` Earlier: `How have you been feeling lately?` | ES: `Me siento decaído.` | ES: `Me apunto.` | Same surface text, different intended sense |
| ES: `No fueron 15, fueron 50.` | EN: `It wasn't 15, it was 50.` | EN: `It was 15, not 50.` | Negation/number swap is a critical error |
| EN: `She hasn't submitted it yet.` | ES: `Ella todavía no lo ha entregado.` | ES: `Ella ya lo entregó.` | Temporal negation flip |
| ES: `Le dije a Marta que la llamó Juan.` with disambiguating prior turns | EN preserving context-grounded participants | EN misidentifying who called whom | Incorrect roles may be critical; ambiguous source needs unrateable, not an invented inference |

## Label schema per output

- `adequate`: `pass`, `fail`, `unrateable`
- `critical_error`: `yes` / `no`
- `error_types`: semicolon-delimited `negation;wrong_number;wrong_name;fabricated_claim;wrong_speaker;omission;slang_sense;idiom_sense;pragmatics;register;modality;regional;agreement;incoherent;other`
- `notes`: optional rationale; `reviewer_code`: pseudonymous assigned ID
- Reviewers do **not** edit source, history, output or anonymized system code.

**Critical-error gate:** Any invented claim, wrong person, negation reversal, number distortion or name substitution with consequential semantic impact is a critical error even if the rest reads fluently.

## Adjudication and statistics

1. Assign both blinded packets to two real bilingual raters; neither sees the other's labels.
2. Independently verify label completeness, language fluency and regional familiarity.
3. Calculate raw agreement, Cohen's κ for adequate vs fail where defined, and agreement on critical error. Explicitly disclose unrateables and reviewer disagreement. Do not treat a model's automated scores as a third rater.
4. A third independently qualified bilingual expert adjudicates disagreements after reviewing source, *past* context, translations and the two labels; retain the original ratings and all decisions.
5. For each direction, compute denominator **including unrateable** (conservative) and, separately, pass rate among rateable cases. Report both, with Wilson 95% intervals. Report critical errors / all evaluated cases and subgroup pass counts. Report paired bootstrap confidence intervals by conversation group if comparing systems.
6. **Tiered thresholds per direction:** general >=95% adequacy; context, idiom, regional slang >=90%; noisy >=88%; <=1% critical error. Record confidence intervals rather than interpreting raw 95% as guaranteed 95% population correctness. No production promotion unless the independent human-run gate and latency checks pass.

## Failure and privacy handling

Stop scoring if packets were compromised by reviewer identity exposure or the test was reused for optimization. Do not train on final-holdout errors. If further experiments are required after revealing test results, discard that holdout for the next final claim and source an entirely new one. No commercial human or cloud model use without Phase 0 cost and rights authorization.

## Implemented local tools

- `python -m research.es_en_v2.review_packet seal --private-holdout data/private/unseen.jsonl --private-manifest data/private/seal.json`: hashes local private source and writes no plaintext to the public repository.
- `research/es_en_v2/review_packet.py` exposes `create_blank_review_packet`, `validate_review`, and `wilson_interval` for custodians. The blinding function **requires already-frozen outputs**; it does not call translation APIs or inspect model quality.
- The separate final holdout and trained model outputs must be supplied by independent custodians after freeze. No test holdout or human scoring is claimed here.
