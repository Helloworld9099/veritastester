# M11: Streaming foundation and adaptation inputs

**Development choice:** evaluate NVIDIA Nemotron 3.5 ASR Streaming 0.6B as a warm start, not as a translator or production selection. Prepared separate, local FLEURS TRAIN and DEV label manifests from six exact publisher TSVs. No full weights or training audio were downloaded, no fitting or deployment ran, and no new final holdout was opened.

## Why this candidate

The [pinned NVIDIA model release](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/ea30d66debe3740a08b573244286791d423d6b3e) documents cache-aware FastConformer attention and convolution, separate source-language conditioning after the encoder, and Spanish and French transcription. Its reported performance is publisher evidence on its own setup, not matched BridgeCaption CPU evidence. The [OpenMDW 1.1 terms](https://openmdw.ai/license/1-1/) provide adaptation and use rights subject to their terms; redistribution of covered materials requires retention of the agreement and applicable copyright/origin notices. Third-party rights remain separate. This is a documented licensing path, not a legal clearance for every downstream use or a change to the app's own code license.

| Pinned artifact | Provenance / SHA-256 |
| --- | --- |
| Native `.nemo` (~2.37 GB) | Publisher LFS metadata, bytes not checked: `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74` |
| Transformers `model.safetensors` (~2.55 GB) | Publisher LFS metadata, bytes not checked: `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174` |
| Transformers `config.json` | Publisher Git blob matched locally; SHA-256 `62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845` |
| Transformers `tokenizer.json` | Publisher Git blob matched locally; SHA-256 `3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246` |
| Transformers `tokenizer_config.json` | Publisher Git blob matched locally; SHA-256 `5c641c5b3f50702a60082690d27c1ce7fcb5a92c4a624793bcae0f21eda3d6e0` |
| Transformers `processor_config.json` | Publisher Git blob matched locally; SHA-256 `ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0` |

These are distinct packaging options: do not use the Transformers metadata as attestation of files embedded in `.nemo`. Freeze the existing audio processor, encoder and original ASR path for the first experiment. Tap the shared acoustic state before ASR language conditioning, and add one direct English translation head for both source languages, with its own decoder state and bounded wait/commit policy. [The integration API](https://huggingface.co/docs/transformers/v5.21.0/en/model_doc/nemotron3_5_asr) exposes encoder cache and separately supplied encoder outputs; [NVIDIA's fine-tuning example](https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr) is transcription-only. Neither provides a fitted translation head. First verify ASR token parity and one actual encoder call per new chunk after attaching an inactive second head, then collect CPU memory/timing before authorizing fitting. Keep both heads and the encoder in one artifact; no language-specific checkpoint or AV provider switch. No new paid compute is authorized by this note.

## What was prepared

[Pinned FLEURS](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd) is published under CC BY 4.0. For each source WAV member named in the `es_419` or `fr_fr` TSV, retain its own unmodified source transcript. Get its English reference only from `en_us` at the same publisher split and parallel sentence ID, as supported by the [authors' parallel-speech design](https://simran-khanuja.github.io/assets/fleurs_slt.pdf). Multiple audio takes may share a sentence and reference. Drop unmatched rows rather than inventing a label. The six TSVs were matched to publisher Git blob IDs and independently SHA-256-pinned.

| Route | TRAIN paired rows | DEV paired rows | Missing English label (TRAIN) |
| --- | ---: | ---: | ---: |
| Spanish→English (Latin American Spanish) | 2,732 | 408 | 64 |
| French→English (France French) | 3,119 | 289 | 74 |
| Total | **5,851** | **697** | **138** |

The [report](../research/unified_models/fleurs_adaptation_inputs_report.json) confirms no cross-split overlap in exact parallel IDs, audio filenames or punctuation/space-insensitive normalized source/English text across all locales. Those checks cannot find re-encoded audio, paraphrases or speaker identity. The public TSVs do not supply independent speaker IDs. Eight already-used M7 DEV WAVs matched their prior normalized SHA-256s and source/English label pairs in this join; that is not full-roster audio verification. Before any optimizer step, verify the exact TRAIN audio archive/member bytes, format, media join, decoded PCM deduplication and validation media independently. The aggregate receipt references metadata rows, not physically staged media or verified training hours.

[CoVoST](https://github.com/facebookresearch/covost) has useful paired material but is excluded here: the [current Spanish](https://mozilladatacollective.com/datasets/cmp704ive02qymp075orb8ok4) and [French](https://mozilladatacollective.com/datasets/cmpbfyxlc002nmj07k67e3ok2) distributions carry research-only restrictions and report withdrawal concerns. Meta's original CC0 description is not used to override those specific package terms. No CoVoST data were accessed.

### Reproduce locally

Put the six pinned TSVs in a dedicated local directory (fetch only `data/{en_us,es_419,fr_fr}/{train,dev}.tsv` from the pinned revision and locally name them `<locale>-<split>.tsv`). This tool has no network or audio path and never requests TEST.

```sh
PYTHONPATH=.:src python -m research.unified_models.fleurs_adaptation_inputs \
  --root "$LOCAL_FLEURS_TABLES" --export-dir "$NEW_LOCAL_DIRECTORY"
```

Output manifests contain text and WAV member names and belong outside Git. Omit `--export-dir` for redacted counts and manifest digests only. Preserve attribution and license with any permitted redistribution. Rollback is to stop using the local preparation tool or revert this research-only commit; there is no app change or external dataset state.

## Evaluation boundary

1. Use FLEURS DEV only for diagnosis/tuning. Keep publisher TEST out of loaders and exports, but do **not** describe it as independent: the NVIDIA card lists FLEURS in its training mixture and evaluation and does not provide lineage sufficient to rule out sample overlap. The prior M7/M8 corpus remains development evidence.
2. Before a generalization or meaning-accuracy claim, freeze a separate held-out conversational set with new, rights-cleared speakers, sessions, prompts and bilingual labels. An independent steward should keep media/labels sealed, audit speaker/session/parallel-sentence and decoded-audio overlap across all training, DEV and reused sets, and freeze the scoring rubric and digests before fitting. Do not reuse the same participants across roles. No collection or sharing was authorized or performed by this phase.
3. Evaluate source WER, translation chrF2 and independent bilingual meaning judgments per direction. Report slang, context, noisy speech, numbers/names and negation separately, plus latency and resource use under a common paced protocol. An automatic text score is not a percentage of meanings kept, and pooled improvement cannot hide a failed language. A failed locked holdout must not become training or selection data.

Next: run a bounded local **ASR-only** cache and token-parity smoke with the pinned real candidate and the eight existing DEV WAVs, and stage a checksum-verified small TRAIN audio slice. That would test whether the warm start is practical before implementing a learnable second head.
