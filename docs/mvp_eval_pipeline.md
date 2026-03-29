# MVP Eval Pipeline

This repo now has an ESM2-only MVP editing/evaluation driver:

`mvp_eval_pipeline.py`

## What it does

For a controlled family test set, it:

1. Builds steering vectors from the top 20% and bottom 20% of a family pool.
2. Edits input sequences with either:
   - random multi-round 10% masking, or
   - ASPO-style multi-round masking via token-relatedness scores from ESM2 representations.
3. Compares:
   - `no_steering`
   - `naive_steering`
   - `alignment_steering`
4. Evaluates:
   - primary fitness improvement (`sol` or `therm`)
   - auxiliary projector score
   - success rate
   - top-k fitness
   - Hamming distance / percent identity / gain per mutation
   - representation drift
   - round-by-round edit trajectories
   - mutation clustering / entropy
   - diversity (`unique_fraction`, sampled pairwise Hamming, sampled pairwise edit similarity)
   - optional protein-likeness metrics (`pPPL`, nearest-natural identity, pLDDT)

## Important MVP choices

- Backbone is ESM2 only.
- Alignment steering is a projection hook with an identity projector for now.
  This keeps the experiment shape stable so a future diffusion prior can drop in.
- Random task:
  all modes are allowed, each round masks `mask_ratio` of the remaining sites, and original tokens are allowed by default.
- ASPO task:
  only steering modes are allowed, token-relatedness scores are computed from cosine similarity to steering vectors at the chosen layers, and the lowest-score sites are edited each round.
- Optional metrics like pLDDT and nearest-neighbor identity are available behind flags and external assets, but are not required for the core editing benchmark.

## Metric groups

### 1. Fitness improvement

Always reported in `summary.json`:

- `sol_mean` / `therm_mean`
- `source_sol_mean` / `source_therm_mean`
- `delta_*_mean`, `delta_*_median`
- `success_rate`
- `edit_success_rate`
- `top_1_*_mean`, `top_5_*_mean`, `top_10_*_mean`
- `source_top_1_*_mean`, `source_top_5_*_mean`, `source_top_10_*_mean`
- `delta_top_1_*_mean`, `delta_top_5_*_mean`, `delta_top_10_*_mean`
- `top_1_success_rate`, `top_5_success_rate`, `top_10_success_rate`

### 2. Edit size and sequence drift

Per-sequence CSV:

- `hamming_distance`
- `percent_identity_to_source`
- `representation_l2`
- `representation_cosine`
- `mutation_count`
- `mutation_fraction`
- `mutation_run_count`
- `mutation_run_mean_length`
- `mutation_run_max_length`
- `mutation_span_fraction`
- `mutation_position_entropy`

Run-level summary:

- `hamming_mean`
- `percent_identity_mean`
- `fitness_gain_per_mutation_mean`
- `representation_l2_mean`, `representation_cosine_mean`
- `mutation_*_mean`
- `global_mutation_position_entropy`

Round-level trajectory in `summary.json -> results -> <mode> -> round_metrics`:

- `representation_l2_mean`, `representation_cosine_mean`
- `step_representation_l2_mean`, `step_representation_cosine_mean`
- `hamming_mean`, `step_hamming_mean`
- `percent_identity_mean`
- `mutation_*_mean`
- `sol_mean` / `therm_mean`
- `delta_sol_mean` / `delta_therm_mean`

Round-trajectory summary:

- `round_count`
- `delta_<primary>_auc`
- `representation_l2_auc`
- `hamming_auc`
- `best_round_by_<primary>`
- `best_round_<primary>_mean`
- `best_round_delta_<primary>_mean`

### 3. Protein-likeness and novelty

Always available:

- `unique_fraction`
- `pairwise_hamming_mean`
- `pairwise_edit_similarity_mean`

Optional:

- `ppl`, `source_ppl`, `delta_ppl`
- `nearest_natural_identity`, `source_nearest_natural_identity`, `delta_nearest_natural_identity`
- `mean_plddt`, `source_mean_plddt`, `delta_mean_plddt`

## Optional inputs and dependencies

### Natural sequence database

To enable nearest-neighbor identity, provide:

```bash
--natural_db_path path/to/natural_sequences.csv
```

Supported formats:

- CSV / TSV with a sequence column
- FASTA
- plain text, one sequence per line

Relevant flags:

- `--natural_db_sequence_col`
- `--natural_db_max_seqs`

### pPPL

Enabled by default unless you pass `--skip_ppl`.

Useful flags:

- `--ppl_model 150M|650M|3B`
- `--ppl_gpu_ids ...`
- `--ppl_batch_masks`

Lower pPPL is generally more protein-like.

### pLDDT

Enable with:

```bash
--compute_plddt
```

Backend selection:

```bash
--plddt_backend esmfold
```

or

```bash
--plddt_backend colabfold
```

Useful ColabFold flags:

- `--colabfold_batch_cmd`
- `--colabfold_msa_mode`
- `--colabfold_model_type`
- `--colabfold_rank`
- `--colabfold_num_models`
- `--colabfold_num_seeds`
- `--colabfold_data_dir`
- `--colabfold_host_url`
- `--colabfold_templates`
- `--colabfold_overwrite_existing_results`
- `--colabfold_batch_args`
- `--colabfold_output_dir`
- `--plddt_cache_csv`

Practical note:

- the built-in ColabFold backend now defaults to an AlphaFold2 + MMseqs2-style invocation
- specifically, the default `--colabfold_msa_mode` is `mmseqs2_uniref_env`
- this matches the official ColabFold notebook/server-backed workflow more closely than the earlier single-sequence fallback
- the recommended default on a GPU server is direct `colabfold_batch`
- point `--colabfold_data_dir` at a persistent weights directory so the AlphaFold parameters are downloaded once
- use a shared `--plddt_cache_csv` across runs so source sequences are folded once and reused
- the local `colabfold_search` + `colabfold_batch` wrapper is still available as an advanced path when you already maintain a local MMseqs database
- that advanced wrapper lives at [`run_colabfold_local.sh`](/Users/chloe/Desktop/project/protein/Steering-PLMs/scripts/run_colabfold_local.sh)

Recommended pattern for multi-run suites:

- point all four runs at the same `--plddt_cache_csv`
- this avoids recomputing source-sequence structures across random / ASPO and sol / therm runs

If the selected backend is unavailable, the run does not fail. Instead, the summary records:

- `plddt_status: "unavailable: ..."`
- `plddt_backend: "esmfold"` or `"colabfold"`

## Example commands

Solubility steering on a controlled family test set:

```bash
python mvp_eval_pipeline.py \
  --property sol \
  --input_csv data/N_sol_test.csv \
  --family_csv data/sol_filtered.csv \
  --output_dir results/mvp_sol_random \
  --mask_strategy random \
  --num_rounds 10 \
  --device cuda:0 \
  --ppl_gpu_ids 0 1 2 3
```

Thermostability steering with ASPO masking:

```bash
python mvp_eval_pipeline.py \
  --property therm \
  --input_csv data/N_therm_test.csv \
  --family_csv data/therm_filtered.csv \
  --output_dir results/mvp_therm_targeted \
  --mask_strategy aspo \
  --modes naive_steering alignment_steering \
  --num_rounds 10 \
  --device cuda:0 \
  --ppl_gpu_ids 0 1 2 3
```

Using ColabFold for pLDDT with a shared cache:

```bash
python mvp_eval_pipeline.py \
  --property sol \
  --input_csv data/N_sol_test.csv \
  --family_csv data/sol_filtered.csv \
  --output_dir results/mvp_sol_random_colabfold \
  --mask_strategy random \
  --device cuda:0 \
  --compute_plddt \
  --plddt_backend colabfold \
  --colabfold_batch_cmd colabfold_batch \
  --colabfold_msa_mode mmseqs2_uniref_env \
  --colabfold_model_type alphafold2_ptm \
  --colabfold_rank plddt \
  --colabfold_num_models 1 \
  --colabfold_num_seeds 1 \
  --colabfold_data_dir /scratch/$USER/colabfold \
  --plddt_cache_csv results/shared_plddt_cache.csv
```

Small UniRef50 smoke test with the recommended direct ColabFold + MMseqs2 backend:

```bash
export COLABFOLD_DATA_DIR=/scratch/$USER/colabfold
export DEVICE=cuda
./scripts/run_uniref50_colabfold_af2_smoke.sh sol random
```

Single full run with direct ColabFold + MMseqs2:

```bash
export COLABFOLD_DATA_DIR=/scratch/$USER/colabfold
export DEVICE=cuda
export OUTPUT_ROOT=results/mvp_colabfold_af2
./scripts/run_uniref50_colabfold_af2_eval.sh sol random
```

Whole direct ColabFold suite across `sol/therm` and `random/aspo`:

```bash
export COLABFOLD_DATA_DIR=/scratch/$USER/colabfold
export DEVICE=cuda
export OUTPUT_ROOT=results/mvp_colabfold_af2_suite
./scripts/run_uniref50_colabfold_af2_eval.sh all all
```

Helper scripts:

- [`run_uniref50_colabfold_af2_smoke.sh`](/Users/chloe/Desktop/project/protein/Steering-PLMs/scripts/run_uniref50_colabfold_af2_smoke.sh)
- [`run_uniref50_colabfold_af2_eval.sh`](/Users/chloe/Desktop/project/protein/Steering-PLMs/scripts/run_uniref50_colabfold_af2_eval.sh)
- [`run_uniref50_colabfold_smoke.sh`](/Users/chloe/Desktop/project/protein/Steering-PLMs/scripts/run_uniref50_colabfold_smoke.sh)

## Expected input columns

- `sequence`: required in both `input_csv` and `family_csv`
- `score`: required in `family_csv`
- `score` in `input_csv` is optional; if present it is copied into the output

## Outputs

For each mode:

- `results/<run>/<mode>/per_sequence_results.csv`

And a run-level summary:

- `results/<run>/summary.json`

The summary now also includes:

- run-level aggregate metrics for fitness, drift, diversity, and optional protein-likeness
- `round_metrics` for per-round accumulation analysis
