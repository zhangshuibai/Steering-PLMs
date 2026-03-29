# Lysozyme Preprocess Pipeline

This pipeline prepares a controlled lysozyme family benchmark that is separate from the editing/eval scripts.

It writes:

- `lysozyme_pool.csv`
- `lysozyme_scored.csv`
- `lysozyme_train.csv`
- `lysozyme_test.csv`
- `P_sol_train.csv`
- `N_sol_train.csv`
- `P_therm_train.csv`
- `N_therm_train.csv`
- `N_sol_test.csv`
- `N_therm_test.csv`
- Optional: `reference_generation_pool.csv`
- `preprocess_manifest.json`

## Run

If you do not already have a lysozyme family CSV, fetch one from UniProt first:

```bash
python fetch_uniprot_family_pool.py
```

That default fetch writes:

- `data/lysozyme/lysozyme_family_input.csv`
- `data/lysozyme/lysozyme_family_input.metadata.json`

The default UniProt query is:

```text
(reviewed:true) AND (protein_name:"lysozyme C") AND (length:[120 TO 170]) AND (fragment:false)
```

This is intentionally narrower than searching plain `lysozyme`, so the pool is more controlled and does not mix as many distinct lysozyme subtypes.

From [`Steering-PLMs`](/Users/chloe/Desktop/project/protein/Steering-PLMs):

```bash
python prepare_family_benchmark.py \
  --config configs/lysozyme_preprocess.example.json
```

If you want to override the output folder without editing the JSON:

```bash
python prepare_family_benchmark.py \
  --config configs/lysozyme_preprocess.example.json \
  --output_dir data/lysozyme/preprocessed_run2
```

## Config Notes

- `input.format` supports `csv`, `tsv`, and `fasta`.
- `scoring.enabled=true` runs both predictors on the cleaned family pool.
- `scoring.enabled=false` is allowed if your input CSV already has `sol_prob` and `therm_score` columns.
- The example config expects `data/lysozyme/lysozyme_family_input.csv`. Running `python fetch_uniprot_family_pool.py` creates it.
- `selection.positive_fraction` and `selection.negative_fraction` control the top/bottom quantile cuts on the train split.
- `selection.test_target_size` controls how many low-score test sequences are exported for each optimization benchmark.
- `reference_pool.enabled=true` exports a separate generation reference pool.

## Output Semantics

- `P_*_train` and `N_*_train` are for steering-vector construction only.
- `N_*_test` are low-scoring benchmark inputs for downstream optimization/evaluation.
- Train/test split happens on the controlled family pool, not on the original predictor datasets.
- `preprocess_manifest.json` records counts, thresholds, split seed, effective test-set fractions, and resolved output paths.
