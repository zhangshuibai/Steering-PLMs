# Eval Metrics Framework

This note summarizes what `mvp_eval_pipeline.py` measures after the current metric expansion, how to read those metrics, and what still requires extra assets.

## What is covered now

### Fitness improvement

The pipeline reports both absolute and source-relative task fitness:

- final edited fitness mean
- source fitness mean
- mean / median `delta_fitness`
- `success_rate`
- `edit_success_rate`
- edited top-k fitness
- source top-k fitness
- top-k fitness deltas
- top-k success rate

This makes it possible to answer both:

- "does editing help on average?"
- "does the method produce a strong best-of-batch tail?"

### Edit size and drift

The pipeline now tracks sequence drift at three levels:

1. Sequence space
- Hamming distance to source
- percent identity to source
- fitness gain per mutation

2. Representation space
- `representation_l2`
- `representation_cosine`

3. Edit pattern geometry
- mutation count / fraction
- contiguous run count
- mean and max run length
- mutation span fraction
- mutation position entropy
- global mutation position entropy

These metrics help separate:

- local editing with clustered substitutions
- global rewriting with dispersed mutations

### Per-round accumulation

Each mode now stores `round_metrics`, including:

- per-round fitness and `delta_fitness`
- per-round representation drift from the source
- stepwise representation drift from the previous round
- per-round Hamming distance from the source
- stepwise Hamming distance from the previous round
- per-round mutation clustering metrics

The summary also adds:

- `delta_<primary>_auc`
- `representation_l2_auc`
- `hamming_auc`
- `best_round_by_<primary>`

This is useful when later rounds keep changing sequences even after the main fitness gain has already saturated.

### Protein-likeness and novelty

Available now:

- diversity via `unique_fraction`
- sampled pairwise Hamming distance
- sampled pairwise edit similarity

Optional when enabled:

- nearest-natural identity
- pseudo-perplexity (`ppl`)
- pLDDT via either ESMFold or ColabFold

For nearest-natural identity and `ppl`, the pipeline also reports source values and edited-source deltas.

## Current interpretation guidance

### Strong fitness with low drift

Usually the most attractive regime:

- positive `delta_fitness`
- low Hamming
- high identity
- small representation drift
- low mutation entropy

This looks like sparse, conservative editing.

### Strong fitness with high drift

This can still be useful, but it is closer to sequence rewriting than editing:

- positive `delta_fitness`
- high Hamming
- low identity
- large representation drift
- high mutation span / entropy

This should be interpreted cautiously, especially in controlled-family benchmarks.

### Good fitness but worse protein-likeness

Watch for:

- `delta_fitness > 0`
- `delta_ppl > 0` when lower `ppl` is better
- negative `delta_nearest_natural_identity`
- lower `mean_plddt`

That combination can indicate oracle hacking or off-manifold edits.

## What still needs external assets

### Natural database for nearest-neighbor identity

You still need to provide a natural sequence database through:

```bash
--natural_db_path ...
```

Without that file, nearest-natural metrics are skipped.

### Structure backend for pLDDT

The pipeline now supports two optional pLDDT backends:

- `--plddt_backend esmfold`
- `--plddt_backend colabfold`

For `colabfold`, the recommended default is now the official AlphaFold2 + MMseqs2-style flow:

- `--colabfold_msa_mode mmseqs2_uniref_env`
- `--colabfold_model_type alphafold2_ptm`
- `--colabfold_rank plddt`
- `--colabfold_num_models 1`
- `--colabfold_num_seeds 1`

If the selected backend is unavailable, the run degrades gracefully instead of failing. For example:

- `plddt_status: "unavailable: No module named 'omegaconf'"`

For ColabFold runs, a shared cache file is strongly recommended:

```bash
--plddt_cache_csv results/shared_plddt_cache.csv
```

That keeps source-sequence structure predictions reusable across multiple experiment runs.

## Recommended next experiments

If we want the most informative benchmark tables next, the best additions are:

1. rerun the full sol / therm experiments with a natural DB path
2. decide whether we want full `ppl` on every run or only on shortlisted modes, since it materially increases runtime
3. prefer the ColabFold backend when ESMFold environment issues are the main blocker for pLDDT
