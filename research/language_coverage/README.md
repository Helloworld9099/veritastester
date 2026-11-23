# BridgeCaption language expansion policy

The unified research model stays **facebook/m2m100_418M**, pinned at revision
`55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`. No MADLAD migration.
The currently trained languages are English, Spanish and French. Mandarin
Chinese (`zh` in the M2M100 tokenizer) is the next language to add; it has
**not** yet received BridgeCaption custom training or proficiency validation.

Read `language_policy.json` for the explicit twelve-route target, including
English as a bridge and regression language. No future language is silently
marked trained simply because M2M100 can tokenize it.

## Training sequence

1. Preserve the SHA-pinned previous champion and evaluate the rank-8
   continuation. Its existence in GitHub does **not** mean it passed.
2. Audit and obtain commercial-use-compatible authentic parallel Mandarin
   conversational data, with independent splits and source provenance.
3. Train Chinese-to-English and English-to-Chinese in the same shared model,
   with balanced replay of genuine Spanish/French/English parallel data.
   Expand to Chinese-to/from Spanish and French only when aligned data and
   suitable evaluation exist.
4. Independently evaluate simplified Mandarin and regional slang, context,
   idioms, names, numbers, negation and code-switching. Do not equate automatic
   chrF or BLEU with 95% meaning accuracy.
5. Accept a candidate only if old six-route performance is preserved and
   the Chinese route improves. Freeze accepted checkpoints and run a new
   independent test before claiming proficiency.
6. For later languages, extend the same language registry and route ledger;
   do not create a new model per language.

## Status

This document and its JSON policy prepare the multilingual pipeline; they
do not constitute new Mandarin optimizer updates. Training and release must
be evidenced by actual checkpoint receipts and bilingual evaluation.
