# BridgeCaption — MADLAD-400-3B-MT (free, permissive, 400+ languages)

**Draft research branch:** `train/madlad400-3b-shared-20261121` · **PR #50** · **Paid spend: $0** · **Production status: unchanged**

## Why switch the research backbone

Google's [MADLAD-400-3B-MT](https://huggingface.co/google/madlad400-3b-mt) is an open-weight, commercially permissive **Apache-2.0** 3B-parameter translation-pretrained T5 model. Its publisher describes over 400 languages and the model repository labels **419 languages**. A single encoder/decoder and target-language prefix (`<2en>`, `<2es>`, `<2fr>`) handle many source/target pairs. The larger MADLAD-7B/10B variants do not add a comparable language-coverage improvement, but need substantially more RAM. M2M100 has 100 native codes; Meta's NLLB-200 is non-commercially licensed and excluded from our future product training.

These numbers are not a verified guarantee of good meaning transfer in every dialect, slang style, noisy voice transcription, endangered language or documented Glottolog language. `research/global_languages/generated/manifest.json` indexes 8,623 documented languages and 13,773 dialects **without claiming machine translation for all of them**. `research/madlad400/model.py` counts only actual `<2...>` tags in the pinned model vocabulary. `research/madlad400/backend.py` rejects unknown, unverified regional variants and unavailable trained weights.

## Version and license

- HF model: `google/madlad400-3b-mt`
- **Exact pinned model commit:** `fa184c675da0b5c9e1c8694fccd4e12e2d422094` (verified source: [HF commit](https://huggingface.co/google/madlad400-3b-mt/commit/fa184c675da0b5c9e1c8694fccd4e12e2d422094)).
- License: `Apache-2.0`; checked against live model-card metadata before downloading weights. Model parameters and training data have different licenses; our first fine-tuning corpus is CC BY 4.0 MASSIVE publisher TRAIN.
- Original full checkpoint `model.safetensors` approximately **11.8 GB**; BF16 CPU inference/training attempts to hold approximately 6 GB of weights, plus gradients/activations/tokenizer and I/O cache. There is a hard RAM/disk preflight.
- Smaller third-party GGUF/CTranslate2 quantizations exist **for inference**, but cannot be assumed to support training or exact model outputs. No license-unverified or model-incompatible quantized checkpoint is used for the first training test.
- No API key, paid GPU, proprietary model or paid cloud endpoint is required.

## Real-model training — what it does

`research/madlad400/train.py` executes a bounded **3-real-optimizer-update free CPU feasibility pilot**:

1. Verify the original trained M2M100 selected adapter SHA `a00c7965211853df07ce3fa032a76283dfc984dcdaeac85e03263445029ffb8e`, plus historical trained Spanish/French specialists, are unchanged. **Do not copy incompatible M2M100 tensors into MADLAD.**
2. Verify pinned MADLAD upstream MIT? **No: MADLAD is Apache-2.0** and the exact 40-hex commit. Load real pretrained T5 weights at CPU BF16 (never a miniature mock for the training run).
3. Retrieve only MASSIVE `AmazonScience/massive` permitted publisher TRAIN and VALIDATION splits, license CC BY 4.0. The 500 previous reviewer examples and all publisher validation sentences are excluded from training; publisher TEST/final holdout remains untouched.
4. Train one **shared rank-4 LoRA** across the final decoder block's four query/value self/cross-attention linear modules, leaving every pretrained tensor frozen. Three steps sample Spanish→English, French→English, English→Spanish. All three directions share the same adapter and the same multilingual base.
5. Save SHA-pinned learned LoRA tensors at steps 1, 2, 3. Measure one pub-VALIDATION example per direction before and after; report per-request time, process RAM, steps and gradients. These tiny development comparisons **cannot establish quality**; never promote from a single chrF/BLEU score.
6. Save the model receipt and small adapters on the research branch; **never push the 11.8 GB pretrained base or change the installed BridgeCaption UI, Whisper ASR or updater**.

### Check/tests

```sh
PYTHONPATH=.:src python -m unittest discover -s research/madlad400/tests -p 'test_*.py' -v
# Only if hardware/network support a large model; requires dependencies from CI:
PYTHONPATH=.:src BRIDGE_NO_PAID_APIS=true python -u research/madlad400/train.py
```

[Real GitHub Actions execution](https://github.com/Helloworld9099/veritastester/actions/workflows/real-madlad400-training.yml). The contract-test job needs only CPU PyTorch; the heavy training job runs **only if these pass** and if hosted runner RAM/disk checks permit downloading/training the actual 3B model. If the job fails or hits resource limits, that is **not a completed training experiment**—inspect logs and preserve prior best model. It is not acceptable to claim that a 3B backbone was trained when only a mock or quantized inference engine ran.

## After the first real checkpoint

Expand carefully: add more **permitted, aligned** bilingual conversational and dialect pairs; restore general training via replay; compare MADLAD vs preserved Spanish/French specialists on a much larger source-conversation-grouped development sample; run qualified blinded bilingual semantic adequacy judging on a separate final holdout, with slang, register, negation/numbers/names/ASR noise subgroups. Train past-only multi-turn context only from human verified bilingual sources (e.g. CC0 UTEP DRAL after proper ingestion). Consider teacher-student distillation from old permitted translators **after** collecting appropriate data and evaluation. Prototype quantized CPU inference for caption latency before shipping.

**No 95% meaning-preservation accuracy or production readiness is claimed by an optimizer step.**
