# Worldwide language support — architecture and honest capability model

**Branch:** `feat/global-language-architecture-and-conversation-training-20261121` · **PR #49, draft** · **No production activation.**

## What the verified catalog contains

The real Glottolog 5.4 and official IANA downloads completed in [Actions run 44965841974](https://github.com/Helloworld9099/veritastester/actions/runs/44965841974). The on-branch `generated/manifest.json` records **8,623 language-level languoids, 13,773 dialect-level languoids (22,396 total)**, and 8,275 IANA language subtags. Glottolog is **CC BY 4.0**. Source archive SHA, compressed catalog SHA, and IANA SHA are pinned; `DocumentedCatalog.load()` refuses altered files.

These numbers are **taxonomy entries, not necessarily extant spoken L1 languages or active translation coverage**. Glottolog includes documented varieties, historical languages, bookkeeping entries and dialects. Its own site distinguishes spoken-first-language counts from broader catalog entries. Never put “8,623 languages supported by AI” in BridgeCaption's marketing.

## Capability model — six separate stages

1. **Cataloged:** Glottolog / IANA know the identifier and sometimes the language family and ISO 639-3 code.
2. **Native tokenizer:** A model actually has an ID for that language. Read `M2M100Tokenizer.lang_code_to_id` from the exact pinned model; never create a guessed fallback such as substituting Chinese for an unrelated language.
3. **Candidate translation:** The tokenizer-backed pretrained model can generate. This is **unvalidated** for a new locale and may be poor.
4. **Trained:** The specific pair and dialect participated in documented fine-tuning. Currently only English/Spanish/French have BridgeCaption-specific training.
5. **Human validated:** Qualified independent bilingual reviewers verify meaning, slang, register, names, numbers and context on a grouped unseen holdout. This has **not** happened for the global expansion.
6. **Production approved:** Translation + ASR + latency + rights + rollback and regression checks pass. No global routes have been activated.

The repository's opt-in `ExpandableResearchTranslator` reads the last selected one-network M2M100 checkpoint, serves only real tokenizer-native language pairs, preserves regional/script tags as **unvalidated variants**, and rejects any request that lacks a native language token. Unknown or dialect-only Glottocodes cannot be auto-routed into a semantically different model. Ordinary users continue using the original BridgeCaption default translator; no model dropdown is introduced.

**Important limitation:** the 418M M2M100 model has native codes for roughly 100 languages. The ability to catalog all known languages is NOT a guarantee one 418M checkpoint can support them all. Future language-families may need re-tokenization, trained language-code embeddings, multiple legally cleared base models, or community-built specialists. The goal is a unified application and training infrastructure, not an impossible promise that 100 tokens are the same as 8,623 trained languages. Codes are versioned independently of pretrained weights; migration must be controlled.

## Growth protocol beyond the first 100

For each new language or dialect, preserve its Glottocode + ISO 639-3 + BCP 47 script/region without language conflation. Require a sourced *license-approved* corpus, native-speaker review and speaker/conversation-grouped splits. Map the language to an existing tokenizer only when validated; otherwise choose a legal pretrained base with that token or perform explicitly evaluated tokenizer/vocabulary expansion. Train with parent-checkpoint provenance, low-resource sampling and replay for existing supported languages. Block checkpoint promotion if any previously supported language regresses; compare latency and RAM. Do not silently train on unknown-provenance, NC, API-limited or unconsented speech.

For sign languages, written/typed language identification is insufficient: future visual sign-language recognition and nontext annotation must be separate capabilities. For endangered languages, no data/permissions may exist; the correct state is `catalog_only` and engagement with language communities, not manufactured translations.

## Relevant source licenses

- Glottolog 5.4 CC BY 4.0: https://glottolog.org/download
- IANA language subtags, RFC 5646: https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry
- M2M100 native language support, MIT model: https://huggingface.co/facebook/m2m100_418M
- Tatoeba CC BY 2.0 FR (attribution required): https://tatoeba.org/en/downloads
- OpenAssistant OASST1 Apache 2.0, **monolingual conversation trees, not parallel translation references**: https://huggingface.co/datasets/OpenAssistant/oasst1
- UTEP DRAL CC0 Spanish–English re-enacted speech (2,893 pairs, 104 conversations; 11GB full archive has not been ingested): https://www.cs.utep.edu/nigel/dral/

## Actual commands

```sh
PYTHONPATH=.:src python -m unittest discover -s research/global_languages/tests -p 'test_*.py' -v
PYTHONPATH=.:src python -m research.global_languages.sync_catalog
PYTHONPATH=.:src python research/global_languages/real_tokenizer_audit.py
```

The sync is idempotent for identical source bytes and fails if a source changes without a versioned catalog migration. All global-language research code lives under `research/`, not in the installed translation provider factory. No app deployment or `main` merge is performed by these commands.
