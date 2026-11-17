# Research-only French LoRA continuation on a free CPU runner

Run a bounded continuation from the **existing, checksum-verified** BridgeCaption French->English adapter. Uses the model's original Hugging Face revision, a source-only French/English Tatoeba text archive distributed by ManyThings, and fresh optimizer state. No application files, provider selection, historical checkpoints, or deployment configuration are changed.

The run makes **96 CPU AdamW optimizer updates** (LoRA rank 8, alpha 16), scoring the prior checkpoint and training steps 48 and 96 against the same internal development sample. A candidate is *selected for further research only* when its dev chrF exceeds the prior checkpoint by at least 0.25 points **and** BLEU does not decrease. Selection does **not** constitute independent evidence of better translation. The archive is downloaded at run time, its digest recorded, and no raw third-party text is uploaded as an artifact. No earlier held-out tests are retuned or reopened.

## Data and limitations

- Source: Tatoeba French-English sentences, distributed by ManyThings.org (`fra-eng.zip`). Text is under CC BY 2.0 France. Credit Tatoeba contributors and ManyThings.org. The archive version is not controlled by us; its SHA-256 is recorded for later provenance checks.
- Source/reference duplicate screening applies to the new train/development partition. **Overlap with prior experiments and the original pretrained model cannot be excluded**, so don't describe the development scores as independent external test performance.
- No manually verified conversational evaluation, no contextual multi-turn training, and no new production model activation. Human and context quality requires newly reserved independent data.
- The previous adapter remains the default choice if the pre-registered development rule fails. Even if the rule passes, this step only creates a research candidate, not a deployable model.

## Run

The branch includes a workflow at .github/workflows/research-cpu-continuation.yml. A push to its named branch should trigger a Linux ubuntu-24.04 CPU runner. The workflow installs pinned Python dependencies, runs four fast logic tests, downloads the official pinned base model, trains and records a small artifact containing **only** candidate LoRA weights and aggregate metrics. No API keys or paid inference are required.

Manual reproduction in a sufficiently provisioned environment from the repository root:

```sh
python3 -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r research/requirements-replay.txt
python3 -m pip install huggingface_hub
python3 -m unittest discover -s research/cpu_continuation_20261117/tests -p 'test_*.py'
python3 research/cpu_continuation_20261117/train.py --output-dir /tmp/bridgecaption-cpu-run-1
```

The output path must not exist before training. Keep model base weights and raw text out of Git. Model weights are under their original Apache-2.0 model terms; this code does not grant new licenses to third-party data.
