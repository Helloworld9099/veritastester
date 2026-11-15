# BridgeCaption — 2026-11-15

- Base main: ff895c353ac8bb2eb1dfbb4e41f1a7ba7f86cfef; branch research/dataset-training-20261115.
- Acquired four source corpora; private raw/normalized text and artifacts remain ignored.
- Retained PR #3 idiom-review fix; fixed real MAGPIE split alias; retained speaker/turn IDs.
- Added pinned download/import and review export commands with actual private queues.
- Auxiliary CPU idiomaticity: single macro-F1 .6938, two-view .6878; no established benefit.
- 1,600 memory and 1,000 sense candidates have zero verified human labels; product training blocked.
- 5,748 DiaBLa rows held out; no translation/audio inference or live activation.
- Final verification: 138 passed, one GUI skip; compile and offline preflight pass.
- See docs/OFFLINE_RESEARCH_RESULTS.md for actual receipts and limits.
- Next: independent fluent review, then coverage/holdout validation and real workstation checks.
