# M2: offline evaluation and promotion gates

This harness evaluates **research development outputs**. It does not run translators, download models, train, open a sealed final set, publish raw text, change the production provider, or promote a model. The app's test/installation surface is unchanged. M2 starts from main and uses explicit, pinned file formats; it does not import the pending M1 branch. A later milestone can furnish the same neutral inputs from the verified M1 registry without needing a second registry.

## Which result are you checking?

| Lane | What it can establish |
| --- | --- |
| Infrastructure-only | Exact checkpoint identities and UTF-8 outputs on the published, frozen development suite; route-by-route, with no new or dropped routes. Identical text is evidence of parity, not proof that the baseline is correct. The performance check requires independently comparable CPU receipts. |
| Model development triage | No direction may lose more than **0.5 absolute chrF points** on the same paired development data, no newly confirmed critical meaning error, and no drop in any assessed subgroup under the conservative default (the suite has no validated subgroup-specific meaning margin). A higher average does not rescue a losing direction. This is a diagnostic lane, not authorization to tune, train or deploy. |
| Independent human evaluation and promotion | Separate, access-controlled work by bilingual evaluators with an untouched holdout. M2 cannot make it pass. Independent final evidence and explicit approval are required before any production decision; no status printed by this tool changes model routing. |

Use the published M0 fixtures only for replay or development; their illustrative translations are not human gold. Only allowlisted M0 bytes or independently entrusted, explicitly unsealed DEV inputs belong here; a checksum cannot authenticate a misdeclared data split. Do not turn an exposed evaluation set into a final test, convert frozen outputs into references, or give a tuning job access to sealed examples. The public suite covers EN→ES, ES→EN, FR→EN and does not measure Mandarin, genuine prior-turn behavior, microphone capture or independent human adequacy.

### Interpretation

- chrF and BLEU use the same paired reference rows and are corpus diagnostics, not accuracy percentages. Group counts and missing annotations matter. Lexical checks for named entities, figures, negation and terms rely on explicit annotations; unlabeled data are unavailable, not zero errors. Literal/token checks cannot decide meaning, correct code-switching, context resolution, or natural slang. Report available subgroup measures for those cases, not semantic pass claims.
- A blinded bilingual development review may identify new critical changes in negation, quantities, named entities, omissions or context leakage. The trusted custodian randomizes and unblinds pairs and supplies independently pinned labels with full case coverage. The hook verifies the matched suite and output bytes; it cannot verify reviewer competence or independence. Do not store review identities, raw text, or private labels in Git or CI logs. No independent final review was run here.
- Compare warm CPU p95 and peak process RSS only on the same declared host, workload, runner/method, thread settings and decoding conditions. Require verified current/checkpoint pins in infrastructure mode, and explicitly pin both expected versions in model-change mode. A small development smoke sample, unstable repeated runs, or missing evidence is not a ±10% pass. Cold-start and end-to-end audio timings are reported separately and never substituted for warmed text translation.

## Replay a pinned development suite

From a repository checkout, use the existing evaluation extra (`python -m pip install -e '.[eval]'`) in your local environment. The tool itself runs offline and expects explicit, separately trusted SHA-256s. The published fixture and manifest are in [the M0 audit commit](https://github.com/Helloworld9099/veritastester/tree/ebcc869c9e3fe7aa42a4d237f30cf44ef6d79028/tests/fixtures). Point `DEV` to that local fixture directory and `CANDIDATE` to a local prediction receipt.

```sh
FIXTURE_SHA=d5fce03b8c49d9dd071da0619661ea57106c6847b750101cb4d0693b3ae51a0c
MANIFEST_SHA=7dc7be2a0df65f8507a67f9cf1e61b558fefdecad3bb81d8552cd4b62b5d36f4
python -m research.unified_eval preflight --suite-format m0 \
  --suite "$DEV/multilingual_regression_v1.jsonl" --suite-sha256 "$FIXTURE_SHA" \
  --manifest "$DEV/multilingual_regression_v1.manifest.json" --manifest-sha256 "$MANIFEST_SHA"
python -m research.unified_eval compare --suite-format m0 --m0-baseline --lane infrastructure \
  --suite "$DEV/multilingual_regression_v1.jsonl" --suite-sha256 "$FIXTURE_SHA" \
  --manifest "$DEV/multilingual_regression_v1.manifest.json" --manifest-sha256 "$MANIFEST_SHA" \
  --candidate "$CANDIDATE" --candidate-sha256 "$CANDIDATE_SHA"
```

Preflight verifies inputs only. `compare` reports only aggregate direction/subgroup evidence, reason codes and digests, not example IDs or raw text. A valid comparison exits **0** only when its technical lane passes; **3** means failed or not verified, with a report; **2** means invalid input or evaluation unavailable. No exit code authorizes promotion.

### Neutral data and evidence format

For new, **unsealed DEV** suites use `--suite-format generic` with a detached JSON manifest (`kind: dev_suite_manifest`, schema version 1, `scope: development`, `split: dev`, `sealed: false`, fixture SHA, row count, and explicit expected direction counts). Keep this manifest pinned outside the candidate output. Rows carry opaque IDs and source/target language, source and reference text, optional diagnostics categories, and optional explicit checks (see `metrics.score_direction`). A check such as `{"id":"number_1","category":"numerals","kind":"required_numeric","expected":["16"]}` tests surface preservation only. The installed metric signatures are recorded; BLEU tokenization and surface diagnostics are not a general cross-script or semantic performance claim.

A prediction JSONL starts with a `dev_prediction_receipt` heading (schema v1, same development/split/unsealed fields, `variant` of baseline or candidate, exact fixture SHA and direction entries), followed by complete `{id,direction,output}` records. Each direction can provide immutable checkpoint pins and a performance receipt. The separately pinned suite manifest can provide expected baseline-output hashes by direction, independently verified checkpoint pins for both arms, and the expected warm request count per direction. Never set expected pins by copying an uncrosschecked candidate receipt. Use `--baseline` and `--baseline-sha256` for generic comparisons.

An optional DEV review receipt (`--development-review` and `--development-review-sha256`) is JSON with `version: 1`, `kind: paired_blinded_development_review`, `fixture_sha256`, canonical `baseline_outputs_sha256` and `candidate_outputs_sha256`, and `judgments` keyed by opaque case ID. Each pair has `baseline` and `candidate` lists of non-text issue tokens such as `quantity:slot_2`. The same error uses the same token across arms; different errors of the same category need different IDs. An empty list means reviewed, not unknown. Those hashes bind the reviewed content; they do not certify how the review was done.

For performance, record at least three independent alternating AB/BA pairs with at least 40 warm requests per repetition. The request count is pinned; extra samples, omitted samples, different hashes, noisy/straddling runs and unmatched environments do not qualify. See `performance.compare_performance` for the receipt contract. Each supplied measurement must have been collected as described; a syntactically valid receipt is not proof that a benchmark ran.

### Verified integration smoke

On the 18 published M0 cases, the harness paired the frozen outputs with M1's genuine replay digests; all three directions matched **6/6**. This tool correctly reported **not verified** for the overall infrastructure lane: the published M0 manifest does not itself supply all trusted rollback/performance inputs for this gate. This does not undo the earlier M1 checksum/replay evidence. As a negative control, deliberately replacing only ES→EN outputs caused that direction to fail (chrF 65.32→7.34, while the other directions stayed unchanged); the full suite failed. Neither exercise ran a new model or evaluated a sealed set.

### Preservation and rollback

No checkpoints, data, original evaluators, prompts or provider defaults are removed. Revert this research-only change to restore the old tools. Keep frozen suite pins, raw reviewer data, and legacy eval scripts in their established locations. An unsupported pair is reported as unavailable; never silently switch to another direction, provider, model, or an easier fixture.
