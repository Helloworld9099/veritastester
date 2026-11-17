# Auxiliary classifiers: research only

These artifacts are NOT the translation adapters, not activated in the app, and not evidence of live translation readiness.

- Synthetic English topic hints: original 8/24 correct, selected character TF-IDF candidate 10/24. Hyperparameters selected with 4-fold CV inside 96 training rows. Tiny synthetic test; no general accuracy claim.
- SemEval idiomaticity: 4,491 EN/PT training examples; selection uses group-held-out internal validation. Current-only 62.79% accuracy/.6260 macro-F1 on 739 official development examples, previous-context 62.11%/.6209. Context F1 difference bootstrap CI [-.0437,.0339]. This is official DEV, not official test, and partly overlaps a previous project pilot.
- Frozen additional 140 source one-shot examples: current-only 51.43%, context 62.14%; F1 difference CI [.0335,.1787]. No fitting or selection on this set, but evidence is mixed across cohorts. These examples are now consumed for future development.
- Synthetic English sense data: 80 model-authored examples, 48/16/16 by scenario family. Context gets 11/16 versus current-only 8/16. NOT human-reviewed validation; five wrong suggestions prohibit readiness claims.

SemEval source numeric labels: 0 idiomatic, 1 literal. This differs from some other corpora. No following sentence, gold label or reference is a feature. Model weights are safe JSON, not pickle. Real-model JSON reload inference reproduced all 739 predictions. Probabilities are uncalibrated.

Run from this directory in an environment with numpy and scikit-learn 1.8.0:

```sh
PYTHONPATH=../../src python scripts/train_topic_candidate.py
python scripts/train_synthetic_senses.py
python scripts/predict_idiom_research.py --model results/real_idioms/current_only_weights.json --text 'They broke the ice.'
```

Real-data reproduction additionally requires the pinned CSVs in `results/real_idioms/source_manifest.json` at `data/external/semeval/`. Raw source corpus is deliberately excluded. Then run `python scripts/train_real_idioms.py`, followed by `python scripts/validate_frozen_idioms.py`. Source repository GPL-3.0 license is retained separately, but dataset rights/product suitability should be reviewed before redistribution or deployment. New experiment code follows the repository MIT license; no third-party source text is relicensed.

Scripts overwrite their local experimental result filenames; preserve existing artifacts before starting a new experiment. Do not retune against any reported evaluation cohort and call it untouched validation.
