# GlobalVoices French/Spanish → English research adaptation

**Research only.** This experiment adapts the existing checksum-verified Marian + LoRA research checkpoints from BridgeCaption on a *different parallel-text domain* — citizen-media articles from Global Voices, packaged by OPUS.

## Source and attribution

- OPUS-GlobalVoices **v2018q4**, bilingual TMX archives: `en-fr.tmx.gz` and `en-es.tmx.gz`.
- Original Global Voices contributors should receive article-level attribution; OPUS (Tiedemann) provides distribution/alignments. Global Voices content is generally published under Creative Commons Attribution, but check individual article/license metadata before releasing derived model weights commercially.
- The workflow downloads the archive from OPUS object storage, verifies that each `<tu>` contains both languages, records its SHA-256 and **never uploads raw sentences, output translations or pretrained base weights**.
- We do not use TED content for AI training. TED's current terms prohibit AI training without a separate written license, even for noncommercial research.

## Scientific controls

- Both languages initialize from the **original, unchanged selected historical adapters**. This is deliberate: QED and OPUS Books experiments had only small/inconsistent domain-specific gains.
- 4,096 source–English pairs for training, 128 for development, 256 source/reference-disjoint held-out pairs for a **single** frozen-test evaluation; the exact same text is not allowed in multiple partitions.
- The TMX archive has no trustworthy article IDs, so the cohorts are **not guaranteed article-independent**. Pretrained OPUS Marian may also have seen this material.
- LoRA rank8/alpha16, frozen Marian base, fresh AdamW state, 256 additional optimizer steps, effective batch 16, learning rate 2.5e-5. Candidate evaluations at steps 64, 128, 192 and 256.
- Model selection uses *only* development chrF (must improve by ≥0.25 compared with historical adapter) and requires development BLEU not decline. Rejected candidates are still retained as separate research checkpoints; they are not automatically promoted.
- After model selection is frozen, evaluate the original Marian base, historical adapter and selected candidate (if distinct) on 256 untouched examples **once**. These are diagnostic corpus scores (chrF/BLEU), not percent of semantically correct translations.
- No candidate is deployed or automatically changes BridgeCaption's live translation provider. Human evaluation with the independently rated gate in `experiments/translation_quality_20261117` is required for a defensible **90% correctness** claim.

## Run

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/requirements-replay.txt 'huggingface-hub>=0.34,<1.0'
python -m unittest discover -s experiments/globalvoices_20261117/tests -p 'test_*.py' -v
python -u experiments/globalvoices_20261117/train_globalvoices.py --language french --output-dir /tmp/globalvoices-fr-1
python -u experiments/globalvoices_20261117/train_globalvoices.py --language spanish --output-dir /tmp/globalvoices-es-1
```

The output folders must not exist before running. GitHub workflow `.github/workflows/globalvoices-research.yml` uses free Linux CPU runners to execute both languages, subject to GitHub policy/availability. Source archive checksums should be pinned after the first verified run if long-term reproducibility is required. It retains research adapters and aggregate metrics as Actions artifacts with a finite expiration, then any worthwhile candidate can be preserved under a versioned checksum without overwriting older research.

## Goal

The user wants **at least 90% correct translation for one language**. This cannot be inferred from chrF, BLEU, corpus loss, reference exact-match, or a repeated test set. Even an automatic metric score ≥90 would not establish 90% semantic correctness. The independent bilingual human-rating protocol must be met before claiming the milestone.
