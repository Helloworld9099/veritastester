# BridgeCaption execution receipt

**STATUS: partial — all feasible offline stages completed.**

| Item | Receipt |
| --- | --- |
| Repo/base | `Helloworld9099/veritastester`; `ff895c353ac8bb2eb1dfbb4e41f1a7ba7f86cfef` |
| Branch | `research/dataset-training-20261115` |
| Datasets | Acquired MAGPIE (CC BY 4.0), Multi3WOZ (repo MIT), SemEval (repo GPL-3.0), DiaBLa (CC BY-SA 4.0). Pinned hashes and full decisions in `docs/DATASET_SOURCE_INVENTORY.md`. |
| CSVs | `data/private/memory_needs_review.csv` 1,600 rows; `sense_needs_review.csv` 1,000 rows. Independently verified product labels: **0**. Exact-schema ready queues have zero rows; public templates included. |
| Trained models | `results/idiomaticity_models.json`: actual English auxiliary idiomaticity, single macro-F1 .6938, two-view .6878 on 2,301 held-out cases. No demonstrated two-view benefit. Repeated weights/predictions identical. |
| Untrained models | Memory usefulness and exact slang meaning need independent human labels. Binary idiomaticity and dialogue adjacency are not replacements. |
| Tests | Python 3.12.14/Linux venv: `python -m compileall -q src tests scripts`, `python -m pytest -q`: **138 passed, 1 GUI skip**. `bridgecaption-preflight`: offline pass; live checks unverified. |
| Security/licences | Source text, private queues, models and references are gitignored; licences retained; no provider credentials or raw source rows in PR. No paid inference, activation, merge or deployment. |
| Blocked evaluation | Real speech, accent/code-switching, latency and A/B/C/D translation require actual workstation/provider access and qualified reviewed models. |
| Next action | Have independent fluent reviewers run the resumable commands in `docs/REPRODUCE_OFFLINE_RESEARCH.md`, then validate group/language/sense coverage before product training. |

Full measurements, exclusions, uncertainty and receipts:
[OFFLINE_RESEARCH_RESULTS.md](docs/OFFLINE_RESEARCH_RESULTS.md).
