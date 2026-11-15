# Reproduce or resume the offline work

Use the `research/dataset-training-20261115` branch and Python >=3.10. Source
rows and models stay under ignored `data/private/` and `results/`. Download
scripts only retrieve pinned data; no source-supplied scripts are executed.

## Fresh reproduction

Run in a fresh checkout so existing exports or annotations cannot be overwritten.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,research,eval]'
python -m compileall -q src tests scripts
python -m pytest -q
bridgecaption-preflight

# Dry runs print what would happen without downloading, preparing or training.
python scripts/acquire_datasets.py
python scripts/run_offline_research.py

# Read DATASET_SOURCE_INVENTORY.md and retain the original licence notices.
python scripts/acquire_datasets.py --download
python scripts/run_offline_research.py --prepare
python scripts/prepare_diabla.py
python scripts/run_offline_research.py --train
```

`--train` runs the source-labelled auxiliary idiomaticity experiment only. It
does not invoke memory/sense training, assert `--confirm-live-verified`, change
environment settings or activate learned artifacts. Checksum mismatches,
source schema errors and split/group conflicts fail loudly. Seed 2026 fixes
subsets, source split selection and models. The ZIP is read by exact member
name and only sampled dialogues are written; no unrestricted archive extract.

## Resume human annotation

Use the existing JSONL review tool to fill the missing scene/reference fields
and label; the ready CSV files are intentionally empty until fields are grounded.

```bash
bridgecaption-corpus-review --task memory \
  --candidates data/private/memory_candidates.jsonl \
  --out data/private/reviewed_memory.jsonl --append

bridgecaption-corpus-review --task sense \
  --candidates data/private/sense_candidates.jsonl \
  --out data/private/reviewed_senses.jsonl --append
```

`q` quits safely, and `--append` skips already-reviewed IDs. Metadata CSVs are
companions for provenance, original idiom groups and missing fields; do not
insert metadata columns or invented values into the exact annotation schemas.
If supplying fully grounded CSVs instead, `bridgecaption-annotate --task memory
--queue data/private/memory_queue.csv --out data/private/reviewed_memory.jsonl
--append` (or sense equivalent) is supported. Complete each required field
independently before using that route.

## Validate and train product predictors later

These commands are **not receipts of execution**. They require genuine labels.

```bash
bridgecaption-train-memory --data data/private/reviewed_memory.jsonl
bridgecaption-train-sense --data data/private/reviewed_senses.jsonl
```

Memory needs >=30 independent cases per tested language, >=6 independent
conversation/speaker groups, helpful and harmful training cases and each
language in independent train/dev/test splits. Sense needs >=60 reviewed
cases, >=12 independent groups, >=10 per language/expression/meaning and two
or more senses of an expression in the actual training split. Enough total
queue rows does not establish these conditions. Rare expressions need more
consented/permissioned examples, and literal senses must be reviewed as exact
meanings too. Do not use held-out DiaBLa rows for training or threshold tuning.

The current product trainers retain their real live-validation gate. Only
after actual Ubuntu audio/provider/privacy/latency checks may an operator run:

```bash
bridgecaption-preflight
bridgecaption-train-memory --data data/private/reviewed_memory.jsonl \
  --train --confirm-live-verified --out results/memory_research.json
bridgecaption-train-sense --data data/private/reviewed_senses.jsonl \
  --train --confirm-live-verified --out results/sense_model.json

# Dry-run controlled four-arm evaluation; same configured engine for every arm.
bridgecaption-eval-context --data data/private/reviewed_memory.jsonl \
  --artifact results/memory_research.json --engine ollama
```

Do not execute cloud inference without its required account/consent. A local
Ollama model must already be available and resource-feasible. Azure is excluded
from B/C/D because its conventional endpoint does not accept contextual prompts.
After independent held-out benefit and live verification, opt in explicitly:

```bash
export BRIDGE_CONTEXT_MODEL="$PWD/results/memory_research.json"
export BRIDGE_SENSE_MODEL="$PWD/results/sense_model.json"
```

Neither setting was enabled in this run. The auxiliary idiomaticity artifact
cannot be used as either product predictor. Translation chrF/meaning and
end-to-end p50/p95 speech latency remain separate future measurements.
