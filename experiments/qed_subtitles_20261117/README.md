# BridgeCaption: educational lecture/subtitle adaptation (QED v2.0a)

**Research only.** This follows the French/Spanish checkpoint-replay and OPUS Books experiments, but begins from the **unchanged selected historical adapters** so data-domain effects can be compared. It does not replace the live Gemini provider or any existing model.

## Why this corpus?

QED/AMARA from Qatar Computing Research Institute contains multilingual educational-video and lecture subtitles. Its transcription/translation style is closer to BridgeCaption's target use than literary books. QED is explicitly **RESEARCH-ONLY** with Qatar Computing Research Institute copyright. A future commercial distribution or deployment requires separate licensing review; no blanket reuse permission is implied.

Original source: https://opus.nlpl.eu/datasets/QED. Aligned Moses files are downloaded from OPUS QED **v2.0a** for the French-English and Spanish-English pairs, with OPUS video/document identifiers for separating fitting and evaluation groups. Cite Abdelali, Guzman, Sajjad and Vogel (LREC 2014) and OPUS (Tiedemann, LREC 2012).

The script records the full SHA-256 digest of the downloaded archive in aggregate reports. **The first experimental run is discovery, not pre-pinned provenance:** after verification, pin the observed digest in `train_qed.py` and rerun as a new experiment if source integrity needs to be independently reproduced.

## Controls

- Existing pinned Marian base weights and unchanged historical LoRA adapter verified by existing checksum records.
- 5,120 filtered aligned training pairs; **128 development** and **256 never-trained-on heldout** subtitle pairs, with independent video/document identifier groups.
- Frozen candidate protocol: 320 optimization steps with checkpoints 64/128/192/256/320. Only development scores select the best candidate.
- Required development gate: at least +0.25 chrF vs prior adapter and no BLEU decrease; old adapter stays preferred otherwise.
- After selection is frozen, score base, previous adapter and selected adapter (if any) on the reserved video-group-heldout sample **once**.
- Outputs contain only aggregate chrF/BLEU, exact-reference fraction (explicitly *not accuracy*), group hashes, and candidate adapter weights. No source sentences, generated translations, source base weights, usernames or credentials are uploaded.
- OPUS pretrained models may already contain data from QED; video-group separation controls this experiment only, not unknown pretraining overlap. Subtitle alignment quality may vary.
- A separate independent bilingual human adequacy review is required to establish the requested 90% correctness milestone; code and scores do not imply that it has happened.

## Reproduce on a free Linux CPU runner

```bash
python -m unittest discover -s experiments/qed_subtitles_20261117/tests -p 'test_*.py' -v
python experiments/qed_subtitles_20261117/train_qed.py --language french --output-dir /tmp/qed-french-1
python experiments/qed_subtitles_20261117/train_qed.py --language spanish --output-dir /tmp/qed-spanish-1
```

Requires CPU PyTorch, sacrebleu, Transformers and huggingface_hub (see existing pinned research requirements). The GitHub Actions workflow `.github/workflows/qed-subtitles-research.yml` runs both languages and preserves results as research artifacts. **No production model is activated.**
