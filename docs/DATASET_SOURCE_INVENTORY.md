# Dataset acquisition receipt — 2026-11-15

This run used public releases for private research. No source rows, references,
reviewer data or learned weights are included in the code PR. The pinned URLs,
exact raw byte sizes and SHA-256 values are in [dataset_downloads.json](dataset_downloads.json).
The acquisition command verifies existing bytes and defaults to a dry run.

| Source | Revision and access | Rights decision | Actual acquired/prepared data | Use and limits |
| --- | --- | --- | --- | --- |
| [MAGPIE](https://github.com/hslh/magpie-corpus) | `7fa677b82b9a772dfa54bbdd0fb414412d73db3b`; 2026-11-15 | Root [CC BY 4.0](https://github.com/hslh/magpie-corpus/blob/master/LICENSE), retained with attribution. BNC-derived context; keep source text private in this project. | Official type-based file, 46,913,846 bytes, 48,395 rows. 26,519 confidently aligned rows imported; 21,876 ambiguous/nonverbatim target locations quarantined. Strict document/text screen leaves 8,769 rows: 5,404 train / 1,064 dev / 2,301 test. English. | Actual human binary idiomaticity labels used for auxiliary training. They are not exact meanings. Filtering creates selection bias toward verbatim targets. |
| [Multi3WOZ](https://github.com/cambridgeltl/multi3woz) | `c65e80fe120704125255de5cf582a51ebaa285cd`; 2026-11-15 | Repository-root [MIT](https://github.com/cambridgeltl/multi3woz/blob/main/LICENSE), notice retained; no additional licence file in data archive. Private dialogue review only in this run. | 90,631,848-byte ZIP. Selected 180 source dialogues per language, with the same IDs across EN/FR/AR/TR: 120 train / 30 dev / 30 test. Total 10,300 turns: 6,740 train / 1,804 dev / 1,756 test. | No memory-utility labels or literal cross-language translation references asserted. 140 dev-list and 140 test-list IDs absent from each released language were excluded. 28 repeated long-text warnings; these conversations were not used for model training. |
| [SemEval-2022 Task 2](https://sites.google.com/view/semeval2022task2-idiomaticity) | [Official repository](https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity), `872625c8f9c5d656b7866ff063a2ab0238ac4c80`; 2026-11-15 | Root GPL-3.0 retained. Private research imports; no corpus/model redistribution in PR. | Downloaded zero-shot train, dev, dev gold and test: 3,666,040 bytes. Joined dev labels by exact ID, checked language. 4,491 train / 739 dev / 2,342 unlabelled test, EN/PT/GL. | No official test gold in this pinned release, so no three-way scored experiment. One-shot files were not mixed into zero-shot data. |
| [DiaBLa](https://huggingface.co/datasets/rbawden/DiaBLa) | `5345895c56a601afe1a98519ce3199be60a27dba`; 2026-11-15 | [CC BY-SA 4.0](https://huggingface.co/datasets/rbawden/DiaBLa/blob/main/LICENSE), notice retained. Test-only throughout. | 9,124,531-byte official JSON; 5,748 references in 144 real source dialogues. EN 2,865 / FR 2,883. Parsed JSON directly, never executed dataset loader code. | All held out. Actual dictionary dialogue keys retained as groups. References target the opposite language; French-source rows target English. No inference, tuning or training on these rows. |
| [SlangInterp](https://github.com/zhewei-sun/slanginterp) | [Data terms](https://github.com/zhewei-sun/slanginterp/blob/master/Data/README.txt) read 2026-11-15 | Urban Dictionary copyrights; research-only restriction confirmed. | Terms inspected; corpus not downloaded. | Deferred. Dictionary meanings would remain unverified review suggestions and cannot replace human contextual labels. No artifact redistribution rights assumed. |
| [NaijaSenti](https://github.com/hausanlp/NaijaSenti) | README inspected 2026-11-15 | Platform/data rights not approved for collection in this run. | No tweet data acquired. Advertised HA/IG/YO/PCM coverage. | Deferred: sentiment labels do not supervise either requested product model. No platform permissions assumed. |
| [Okwu](https://github.com/Okwu-AI/corpus) | README inspected 2026-11-15, advertises v1.9 | Data advertises CDLA-Permissive-2.0, code Apache-2.0. Exact release and upstream provenance not audited for acquisition. | No corpus acquired. Advertised YO/SW/PCM parallel data. | Deferred: parallel pairs are not conversation-utility labels; corpus quality/provenance would need review first. |
| [Bangor Miami](https://talkbank.org/biling/access/Bangor/Miami.html) | Official corpus/access page read 2026-11-15 | TalkBank access, participant/privacy and reuse requirements not established for this run. | No transcripts/audio acquired. Advertised Spanish–English code switching. | Skipped. Source describes recordings with special consent/transcription constraints. No raw CHAT parsing or permitted collection asserted. |

Licence decisions describe this bounded research workflow, not a warranty of
all upstream third-party rights or permission to publish datasets. Retain the
original notices and citations in any permitted derivative work.

MAGPIE attribution: Haagsma, H., Bos, J., and Nissim, M. (2020), *MAGPIE: A Large
Corpus of Potentially Idiomatic Expressions*. Multi3WOZ attribution: Hu et al.
(2023), *Multi3WOZ: A Multilingual, Multi-Domain, Multi-Parallel Dataset for
Training and Evaluating Culturally Adapted Task-Oriented Dialog Systems*.
SemEval attribution: Tayyar Madabushi et al. (2022), *SemEval-2022 Task 2:
Multilingual Idiomaticity Detection and Sentence Embedding*. DiaBLa attribution:
Bawden et al. (2021), *DiaBLa: A Corpus of Bilingual Spontaneous Written
Dialogues for Machine Translation*; see the retained source card for full citation.
