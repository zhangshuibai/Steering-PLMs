# MVP Eval Pipeline

This repo now has an ESM2-only MVP editing/evaluation driver:

`mvp_eval_pipeline.py`

## What it does

For a controlled family test set, it:

1. Builds steering vectors from the top 20% and bottom 20% of a family pool.
2. Edits input sequences with either:
   - random 10% masking, or
   - targeted ASPO-style masking via cosine-ranked ESM2 token representations.
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
   - diversity (`unique_fraction`, sampled pairwise Hamming)
   - pPPL

## Important MVP choices

- Backbone is ESM2 only.
- Alignment steering is a projection hook with an identity projector for now.
  This keeps the experiment shape stable so a future diffusion prior can drop in.
- Targeted masking is ASPO-style rather than paper-exact:
  token positions are ranked by cosine similarity to steering vectors at the chosen layers.
- Optional metrics like pLDDT and nearest-neighbor identity are intentionally left out of the MVP summary.

## Example commands

Solubility steering on a controlled family test set:

```bash
python mvp_eval_pipeline.py \
  --property sol \
  --input_csv data/N_sol_test.csv \
  --family_csv data/sol_filtered.csv \
  --output_dir results/mvp_sol_random \
  --mask_strategy random \
  --device cuda:0 \
  --ppl_gpu_ids 0 1 2 3
```

Thermostability steering with targeted masking:

```bash
python mvp_eval_pipeline.py \
  --property therm \
  --input_csv data/N_therm_test.csv \
  --family_csv data/therm_filtered.csv \
  --output_dir results/mvp_therm_targeted \
  --mask_strategy targeted \
  --device cuda:0 \
  --ppl_gpu_ids 0 1 2 3
```

## Expected input columns

- `sequence`: required in both `input_csv` and `family_csv`
- `score`: required in `family_csv`
- `score` in `input_csv` is optional; if present it is copied into the output

## Outputs

For each mode:

- `results/<run>/<mode>/per_sequence_results.csv`

And a run-level summary:

- `results/<run>/summary.json`
