# TICO-19 external benchmark — IMPORTANT quality correction

**Status: historical diagnostic only. Do not use the PR #24 chrF/BLEU numbers to claim generalization or choose a model.** The earlier automated benchmark was computed and persisted reproducibly, but qualitative inspection after the first frozen run exposed incorrect French source/reference pairings in the selected 256-item test cohort.

## Verified examples of defective reference alignments

The French source:

> Le principal objectif médical du lavage des mains est d’éliminer les agents pathogènes ... et les produits chimiques ...

is about removing pathogens and chemicals through **handwashing**. The published English reference paired to that row was:

> MERS-CoV and its closest relative bat CoV-HKU25 share only 87% nucleotide sequence homology.

That is unrelated. The base and stage-three models translated the *actual French sentence* substantially correctly. The reference would incorrectly count those translations as low-scoring.

Two more examples in the same 12-case deterministic audit paired French statements about (1) bats as genetic reservoirs of HCoV and (2) intermediate mammal hosts with unrelated English references about viruses/receptors. These are defects in the loaded source/reference pair mapping *as evaluated*, regardless of whether they originate in the dataset or its distribution pipeline. The original reference files and exact Git revisions are retained for investigation. **Do not replace references with model predictions or rewrite historical logs.**

## Translation errors uncovered by bilingual AI-assisted inspection

These are **AI-assisted qualitative observations, not independent fluent-human ratings**:

- French: `69 décès (0,004 % ...)` was rendered with **0.04%** by both base and latest stage-three adapter—a **tenfold numeric error**. The added `semantic_checks.numeric_fidelity` utility now flags that kind of error.
- French: `respirateurs destinés à l’Espagne` (in the context of seized medical ventilators) became **breathers** in the stage-three model; it did not reliably preserve the medical equipment meaning.
- Spanish: `me dio fiebre ayer` (I got a fever yesterday) became **gave me a fever yesterday** in the stage-three model, omitting the person affected. The unmodified base translated this correctly.
- Spanish: The stage-three model rendered retail store `horarios de atención` as **hours of care**, altering the meaning.

These constitute actual, observed failures. A dozen cases per language cannot give a defensible population-wide accuracy rate, and the TICO-19 references are demonstrably unreliable in the checked French subset.

## Scientific consequence

The frozen TICO sample is **consumed**. Keep its code, models, hashes and historical scores as an audit trail, but **do not** use its scores for model selection, continued training, or the final >=90% accuracy milestone.

For subsequent, independently meaningful evaluation:

1. Choose a genuinely separate high-quality human-translated dataset, e.g., the original FLORES+ `devtest` **only once, at the end of training**, and reserve it in a versioned dataset manifest. FLORES+ has 1,012 human translations per language in `devtest` under CC BY-SA 4.0.
2. Check source/reference alignment **before scoring**, using independent bilingual or calibrated cross-lingual review and source corpus metadata. Do not infer correctness solely from chrF/BLEU.
3. Train only on appropriately licensed **train** partitions. Keep final test examples and their paraphrases out of further training, checkpoint selection, prompt/rule tuning and demonstration examples.
4. For the genuine >=90% translation-correctness target, use the separately merged `experiments/translation_quality_20261117/collect_reviews.py` to obtain real, independent bilingual judgments on genuinely unseen conversations. AI-only judgments cannot be represented as human-reviewed adequacy.

Original qualitative error audit: GitHub Actions run `44238812101`. Original frozen benchmark run: `44237442364`. Both are preserved to avoid silently hiding erroneous evidence.
