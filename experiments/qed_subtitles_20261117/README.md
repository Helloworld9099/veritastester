# BridgeCaption: QED educational subtitle adaptation (research only)

## Why a different corpus?

BridgeCaption's existing French->English and Spanish->English models were adapted on sentence corpora (Tatoeba, Europarl) and book-aligned text (OPUS Books). QED/AMARA is a multilingual collection of subtitles translated for educational videos and lectures, closer to BridgeCaption's intended lecture/meeting context.

**Rights:** QED v2.0a is made available for RESEARCH purposes ONLY by Qatar Computing Research Institute; no production/commercial use or redistribution is assumed. Attribution: Abdelali, Guzman, Sajjad and Vogel (2014), OPUS/Tiedemann. Original source: https://opus.nlpl.eu/datasets/QED.

## Data integrity update

The first attempt to use OPUS Moses bitext files was aborted. Although the archive contains French/Spanish, English, and XCES alignment XML, their plain-text files contained different row counts. Silently matching those files by line number could create *incorrect bilingual training pairs*.

This experiment therefore uses OPUS's **compressed TMX translation memory**, containing both language segments in one `<tu>` unit. Each entry is parsed with the standard library XML parser. Source/English order, lengths, exact-text duplicates and language identifiers are checked before training.

TMX does NOT reliably preserve source-video/speaker IDs. We reserve **128 development and 256 final evaluation translation units** disjoint by normalized exact source and English reference from 5,120 new training pairs. **Do not claim independent video/document evaluation**: near duplicates, author/topic overlap and pretraining contamination remain possible. The final test is scored only after development checkpoint selection is frozen.

## Model protocol

Original pinned Marian base and prior saved selected adapter remain immutable and are verified by original research checksums. Each research run uses LoRA rank8/alpha16, a fresh AdamW optimizer with learning rate 2.5e-5, 320 additional updates (effective batch16), and checkpoints at 64/128/192/256/320 updates. The accepted model must improve dev chrF by at least 0.25 versus the previous adapter **and not lower dev BLEU**; otherwise the previous adapter remains preferred. The selected candidate is never activated in the BridgeCaption live app.

After selecting only from development scores, compare frozen base, previous research adapter, and new candidate (if any) on the 256 held-out translation units. Report *corpus chrF and BLEU*, not percent-correct translations. Save only aggregate reports, archive SHA-256, and candidate adapter weights in GitHub Actions artifacts; raw subtitle sentences, personal data and full base weights remain excluded.

This is a domain-adaptation experiment, **not evidence that the model has reached 90% semantic translation adequacy**. The separate independently rated quality rubric in `experiments/translation_quality_20261117/` defines the user-requested 90% threshold.

## Reproducibility

The first correctly parsed TMX archive hash is logged as `QED_TMX_SHA256`; pin its exact digest in `train_qed.py` after inspection to prevent silent source changes. The QED source URLs are fixed to `v2.0a`, but that alone is not a cryptographic pin.

```bash
python -m unittest discover -s experiments/qed_subtitles_20261117/tests -p 'test_*.py' -v
python experiments/qed_subtitles_20261117/train_qed.py --language french --output-dir /tmp/qed-fr-1
python experiments/qed_subtitles_20261117/train_qed.py --language spanish --output-dir /tmp/qed-es-1
```

Use existing CPU PyTorch and pinned research requirements, plus `huggingface-hub>=0.34,<1.0`. The separate workflow `.github/workflows/qed-subtitles-research.yml` runs both languages on free GitHub Actions CPU runners.
