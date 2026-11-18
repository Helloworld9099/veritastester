# French dialogue research — Multi3WOZ, official heldout isolated

**Scope:** Train another research-only French→English LoRA continuation using genuine conversational data in the public Multi3WOZ corpus. Keep all older weights; do not activate the candidate in the BridgeCaption app.

## Provenance and rights

**Source:** Cambridge Language Technology Lab's [Multi3WOZ repository](https://github.com/cambridgeltl/multi3woz), commit `c65e80fe120704125255de5cf582a51ebaa285cd`. Source `data.zip` SHA-256: `da238995eed56ed76551c19d0d3fc58e039b62197514ca0d7b705880664ca65e`.

**Attribution:** Songbo Hu, Han Zhou, Mete Hergul, Milan Gritta, Guchun Zhang, Ignacio Iacobacci, Ivan Vulić, Anna Korhonen, *Multi3WOZ: A Multilingual, Multi-Domain, Multi-Parallel Dataset for Training and Evaluating Culturally Adapted Task-Oriented Dialog Systems*, TACL 2023. The repository's root LICENSE is MIT; the publication is CC BY 4.0. Retain dataset and publication credits.

## Crucial translation-alignment caveat

Multi3WOZ is **culturally adapted multilingual dialogue**, not a literal human translation of every original English turn.

Verified from the authors' actual source files:

| French turn | Attached English `en_text` |
|---|---|
| Bonjour, Je cherche un restaurant indien. | I'd really like to take my client out to a nice restaurant that serves indian food. |

These differ materially: the original English includes a client and an outing that the French speaker did not mention. **Unfiltered training would teach mistranslation**. This experiment rejects obvious mismatches with word-length and character-length ratio bounds, numeric token preservation, and source annotation-error guards. These checks are conservative screening, **not a semantic equivalence guarantee**. Some misleading pairs may remain. Do not claim this is a verified alignment dataset or use it for human accuracy certification.

## Separate, frozen partitions

- Read the authors' explicit `valListFile.txt` and `testListFile.txt` for the French corpus. They are excluded from the entire fitting partition by *dialogue ID*, before selecting any rows.
- Draw **1,536 training utterances** only from non-validation, non-test dialogue IDs.
- Draw **128 development utterances** only from official validation dialogue IDs to select a checkpoint.
- Draw **256 test utterances** only from official test dialogue IDs; score this reserved test **once**, after the development checkpoint has been selected, with no access from the optimizer or checkpoint selection.
- Deduplicate source and English reference texts globally across every partition. Select by stable hash, independent of model predictions or scores. Persist only dialogue-ID and sample hashes, aggregate metrics, and candidate LoRA weights—not copyrighted raw dialogues.
- Ensure the original pretrained Marian might still have seen some source content; we can exclude only **our additional training**, not unknown upstream pretraining.

## Training setup

Start from the saved, SHA-verified **MASSIVE stage-three French adapter** (`072f7f5d2d90b26991c5c62bf8ddb40f686f65e66be52717e7b121d507008ad9`). Use the same frozen Marian base, rank-8/alpha-16 LoRA with a fresh AdamW optimizer, learning rate 2.5e-5, effective batch size 16, and **96 optimizer updates**. Evaluate candidates at steps 24, 48, 72, 96 on the development partition only. Preserve the old adapter unless the new candidate improves dev chrF by at least 0.25 without lowering BLEU.

The heldout test will compare pretrained base, previous MASSIVE stage-three adapter, and selected candidate **only after selection**. The outcome is automatic chrF/BLEU, **not 90% human-rated correctness**.

## Free CPU reproducibility

The workflow `.github/workflows/train-multi3woz.yml` checks out exactly the author's public commit, verifies the archive SHA-256, validates source quality gates, trains on fresh dialogue IDs, and preserves only aggregate reports and LoRA weights as a research artifact. The source zip is processed on the transient runner and never copied into this repo.

```bash
python -m unittest discover -s experiments/multi3woz_20261118/tests -p 'test_*.py' -v
python -u experiments/multi3woz_20261118/train_dialogue.py \
  --source-zip /tmp/multi3woz/data.zip --output-dir /tmp/french-dialogue-research
```

**Success criteria:** research-only reproducible model selection and non-overlapping train/dev/test groups with independently measured diagnostic scores. **The user's 90% goal is not certified** by this run; a separate, genuinely unseen conversation corpus and independent bilingual assessment remain necessary.
