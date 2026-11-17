# BridgeCaption: MASSIVE localized voice-assistant translation adaptation

This is a controlled French→English / Spanish→English LoRA research experiment on **MASSIVE 1.1** from Amazon Science, with natural human-localized voice-assistant commands. It is much closer to short spoken commands than the previous Tatoeba, translated news, or books experiments.

## Source, permissions and language alignment

- Dataset: [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive), pinned repository revision `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`.
- Dataset license: **CC BY 4.0**. Preserve attribution to FitzGerald et al. (ACL 2023), Amazon Science, and the original SLURP work by Bastianelli et al. (EMNLP 2020).
- The dataset has human-translated localized `fr-FR`, `es-ES`, and `en-US` utterances, aligned using the **original SLURP `id`**. It also contains user intent and slot labels; the training script only reads language, ID, partition, utterance, and intent consistency. It does not collect user data.
- Download via Hugging Face with the pinned official dataset builder and `trust_remote_code=True` (required by the 2022 dataset script). This executes a pinned third-party loader; review the upstream code and source before trusting it. It downloads the author's 1.1 archive from its distribution host. No credentials or paid APIs are required.

## Train / validation / test integrity

- Keep the author's **original train/dev/test partitions**; never move a test utterance into training.
- Pair by equal SLURP ID, check that English and target-language intent labels match, and remove normalized exact source/reference duplicates across partitions, prioritizing untouched test and development sets.
- Train on **4,096** pairs, select on **128** development pairs and evaluate **256** untouched test pairs once after model selection.
- Preserve original selected Marian LoRA adapter and pinned base weights. Fresh AdamW, rank-8 / alpha-16, effective batch 16, learning rate 2.5e-5 and 256 updates with checkpoints 64,128,192,256. Candidate must improve chrF by ≥0.25 without reducing development BLEU.
- No raw utterances, references or generated translations are uploaded or committed. Only hashes, aggregate metrics, and candidate adapter weights are retained in research artifacts.
- Valid test results do **not** imply that the original pretrained Marian model never saw these utterances; and Amazon localizations may intentionally differ from English in names/times/slots. This is *text-only*, no live speech or multi-turn context.

## Reproduce with free CPU compute

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r research/requirements-replay.txt
python -m pip install 'huggingface-hub>=0.34,<1.0' 'datasets==2.21.0'
python -m unittest discover -s experiments/massive_20261117/tests -p 'test_*.py' -v
python experiments/massive_20261117/train_massive.py --language french --output-dir /tmp/massive-fr-1
python experiments/massive_20261117/train_massive.py --language spanish --output-dir /tmp/massive-es-1
```

`.github/workflows/massive-research.yml` uses free GitHub Actions CPU runners when permitted by the repo and availability. Training artifacts remain research-only; no app activation or provider changes.

## 90% target

The requested `90% baseline` is **90% independently human-rated semantic translation adequacy** for at least one language. Corpus chrF, BLEU, exact-match rate or NLU intent consistency is a different quantity and cannot be reported as translation accuracy. A separate, newly reserved bilingual human evaluation is required before claiming the goal; use `experiments/translation_quality_20261117/score_human.py`. Even if all 256 MASSIVE test references match, that is not the same as independent human assessment of real conversational speech.
