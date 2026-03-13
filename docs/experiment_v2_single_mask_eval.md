# Experiment V2: Single-Mask Evaluation — Isolating Single-Step GLP Error

## Date: 2026-03-12

## Motivation

V1 (`docs/experiment_v1_comprehensive_eval.md`) found that GLP projection degrades pPPL across all (u, steps) configurations. However, V1 uses 10 rounds of iterative mask-predict (mask_ratio=0.1), applying steering+GLP once per round — 10 total applications. The pPPL degradation could stem from error accumulation across rounds rather than single-step GLP error.

This experiment isolates single-step GLP error by masking **one position at a time** with a **single forward pass**, removing the iterative accumulation factor.

## Experimental Setup

### Generation Pipeline
- **Base model**: ESM2-650M (`esm2_t33_650M_UR50D`)
- **Steering**: Single-layer steering at Layer 17, norm-preserving rescale
- **GLP model**: `generative_latent_prior/runs/glp-esm2-650m-layer17-d6/`, 334M params
- **GLP projection**: SDEdit — normalize -> add noise to level u -> denoise -> denormalize
- **Single-mask protocol**: For each reference sequence, randomly sample 10 positions. For each position: mask it -> one forward pass -> nucleus sampling (temperature=1.0, top_p=0.9) -> fill back. Each masked position produces one output sequence.
- **Seed**: All groups use the same `np.random.RandomState(42)` so mask positions are identical across groups.
- **Reference data**: `data/sol_easy.csv` (162 sequences, avg length 180.4, range 71-256)

### Evaluation Metrics
- **Sol Mean Prob**: Mean predicted solubility probability from oracle predictor (PropertyPredictor on ESM2-650M features). Continuous score, not thresholded.
- **Sol Ratio**: Fraction with prob >= 0.5 (for reference, but not informative in this experiment since single-token changes are too small to affect global solubility).
- **pPPL**: ESM2-3B pseudo-perplexity (lower = more natural). **Primary metric** for this experiment.

### Experimental Groups (33 total)

#### Baselines (3 groups)

| # | Method | steering_vectors | glp_project_fn | N seqs |
|---|--------|------------------|----------------|--------|
| 0 | Reference | — | — | 162 (original, no generation) |
| 1 | No Steering | None | None | 1620 |
| 2 | L17 no GLP | sv_single (zeros except L17) | None | 1620 |

#### GLP Groups (30 groups: 6 u × 5 steps)

| u \ steps | 25 | 50 | 100 | 200 | 400 |
|-----------|----|----|-----|-----|-----|
| 0.1 | 1620 | 1620 | 1620 | 1620 | 1620 |
| 0.3 | 1620 | 1620 | 1620 | 1620 | 1620 |
| 0.5 | 1620 | 1620 | 1620 | 1620 | 1620 |
| 0.7 | 1620 | 1620 | 1620 | 1620 | 1620 |
| 0.9 | 1620 | 1620 | 1620 | 1620 | 1620 |
| 1.0 | 1620 | 1620 | 1620 | 1620 | 1620 |

Total: 162 + 32 × 1620 = **52,002 sequences**

### Forward Path Selection (inside `generate_single_mask()`)

- **No Steering**: `model(tokens)` — vanilla ESM2 forward
- **L17 no GLP**: `model.steering_forward(tokens, sv_single)` — `steering_forward` from `module/steerable_esm2.py`, applied at all layers but `sv_single` has zeros at non-17 layers (mathematically equivalent to steering only at L17)
- **GLP groups**: `model.steering_forward_glp(tokens, sv_all, glp_fn, layer=17)` — `steering_forward_with_glp` from `steering_with_glp.py`, applies steering only at `layer_idx == 17`, then GLP projection

### GPU Setup
- `CUDA_VISIBLE_DEVICES=4,5,6,7`
- Phase 1 (generation + sol eval): `cuda:0` (physical GPU 4)
- Phase 2 (pPPL eval): `cuda:0-3` (physical GPUs 4-7), 4-way parallel ESM2-3B

## Repo Changes

### New files
- `run_single_mask_eval.py` — main experiment script

### Existing files used (no modifications)
- `steering_with_glp.py` — `build_glp_projection_fn()`, `steering_forward_with_glp()`, `evaluate_sol()`, `PropertyPredictor`, `extract_features_650m()`
- `module/steerable_esm2.py` — `steering_forward()`
- `evaluate_ppl.py` — `compute_pseudo_perplexity_multi_gpu()`
- `utils/esm2_utils.py` — `load_esm2_model()`, `decode()`
- `utils/gen_utils.py` — `sample_top_p()`
- `generative_latent_prior/glp/denoiser.py` — `GLP` model class

### Data files used
- `data/sol_easy.csv` — 162 reference sequences
- `saved_steering_vectors/650M_sol_steering_vectors.pt` — steering vectors
- `saved_predictors/sol_predictor_final.pt` — solubility oracle predictor
- `generative_latent_prior/runs/glp-esm2-650m-layer17-d6/` — trained GLP checkpoint

### Generated data
- `results/single_mask_eval/reference.csv` — 162 sequences
- `results/single_mask_eval/no_steering.csv` — 1620 sequences
- `results/single_mask_eval/l17_no_glp.csv` — 1620 sequences
- `results/single_mask_eval/l17_glp_u{u}_s{steps}.csv` — 1620 sequences each, 30 files
- `results/single_mask_eval/summary.json` — all results with metrics

## Results

### Baselines

| Method | N | Sol % | Sol Mean Prob | pPPL Mean | pPPL Median | pPPL Std |
|--------|---|-------|---------------|-----------|-------------|----------|
| Reference | 162 | 17.90% | 0.1987 | 5.4659 | 4.9269 | 2.4669 |
| No Steering | 1620 | 17.84% | 0.1995 | 5.4806 | 4.9391 | 2.4680 |
| L17 no GLP | 1620 | 17.84% | 0.1992 | 5.4776 | 4.9501 | 2.4447 |

### GLP pPPL Mean Matrix (u × steps)

| u \ steps | 25 | 50 | 100 | 200 | 400 | Row Mean |
|-----------|-------|-------|-------|-------|-------|----------|
| **0.1** | 5.595 | 5.594 | 5.600 | 5.583 | 5.587 | **5.592** |
| **0.3** | 5.576 | 5.568 | 5.559 | 5.569 | 5.574 | **5.569** |
| **0.5** | 5.512 | 5.509 | 5.516 | 5.514 | 5.515 | **5.513** |
| **0.7** | 5.486 | 5.492 | 5.496 | 5.496 | 5.492 | **5.493** |
| **0.9** | 5.486 | 5.490 | 5.480 | 5.482 | 5.484 | **5.484** |
| **1.0** | 5.481 | 5.484 | 5.483 | 5.486 | 5.488 | **5.484** |
| **Col Mean** | **5.523** | **5.523** | **5.522** | **5.522** | **5.523** | |

### GLP ΔpPPL vs Reference Matrix

| u \ steps | 25 | 50 | 100 | 200 | 400 |
|-----------|--------|--------|--------|--------|--------|
| **0.1** | +0.129 | +0.128 | +0.135 | +0.117 | +0.121 |
| **0.3** | +0.110 | +0.102 | +0.094 | +0.103 | +0.108 |
| **0.5** | +0.046 | +0.043 | +0.050 | +0.048 | +0.049 |
| **0.7** | +0.020 | +0.027 | +0.030 | +0.030 | +0.026 |
| **0.9** | +0.020 | +0.024 | +0.014 | +0.016 | +0.018 |
| **1.0** | +0.016 | +0.018 | +0.017 | +0.020 | +0.022 |

Reference: No Steering ΔpPPL = +0.015, L17 no GLP ΔpPPL = +0.012

### GLP Sol Mean Prob Matrix

| u \ steps | 25 | 50 | 100 | 200 | 400 |
|-----------|--------|--------|--------|--------|--------|
| **0.1** | 0.1976 | 0.1978 | 0.1986 | 0.1978 | 0.1989 |
| **0.3** | 0.1993 | 0.1992 | 0.1988 | 0.1992 | 0.1989 |
| **0.5** | 0.1997 | 0.1992 | 0.2003 | 0.1997 | 0.1993 |
| **0.7** | 0.1999 | 0.2008 | 0.2002 | 0.2002 | 0.1995 |
| **0.9** | 0.2000 | 0.1994 | 0.1998 | 0.2003 | 0.2005 |
| **1.0** | 0.1997 | 0.1998 | 0.1992 | 0.2002 | 0.1999 |

All values within 0.1976–0.2008, indistinguishable from Reference (0.1987).

### GLP Sol Ratio (%) Matrix

| u \ steps | 25 | 50 | 100 | 200 | 400 |
|-----------|-------|-------|-------|-------|-------|
| **0.1** | 17.65 | 17.53 | 17.78 | 17.78 | 17.96 |
| **0.3** | 18.21 | 17.90 | 18.15 | 17.90 | 18.02 |
| **0.5** | 18.02 | 17.96 | 18.02 | 18.09 | 17.96 |
| **0.7** | 18.09 | 18.21 | 18.02 | 18.09 | 18.09 |
| **0.9** | 18.15 | 17.96 | 18.27 | 18.15 | 18.02 |
| **1.0** | 18.09 | 18.27 | 17.96 | 18.15 | 18.09 |

All values within 17.5–18.3%, no meaningful variation.

### Detailed Per-Configuration Data

#### u=0.1
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.1765 | 0.1976 | 5.5951 | 5.0708 | 2.4646 |
| 50 | 0.1753 | 0.1978 | 5.5942 | 5.0461 | 2.4595 |
| 100 | 0.1778 | 0.1986 | 5.6005 | 5.0605 | 2.4845 |
| 200 | 0.1778 | 0.1978 | 5.5825 | 5.0710 | 2.4490 |
| 400 | 0.1796 | 0.1989 | 5.5872 | 5.0660 | 2.4632 |

#### u=0.3
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.1821 | 0.1993 | 5.5760 | 5.0529 | 2.4744 |
| 50 | 0.1790 | 0.1992 | 5.5682 | 5.0390 | 2.4690 |
| 100 | 0.1815 | 0.1988 | 5.5594 | 5.0476 | 2.4426 |
| 200 | 0.1790 | 0.1992 | 5.5691 | 5.0413 | 2.4543 |
| 400 | 0.1802 | 0.1989 | 5.5742 | 5.0515 | 2.4667 |

#### u=0.5
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.1802 | 0.1997 | 5.5122 | 4.9642 | 2.4701 |
| 50 | 0.1796 | 0.1992 | 5.5091 | 4.9646 | 2.4643 |
| 100 | 0.1802 | 0.2003 | 5.5162 | 4.9567 | 2.4828 |
| 200 | 0.1809 | 0.1997 | 5.5136 | 4.9515 | 2.4755 |
| 400 | 0.1796 | 0.1993 | 5.5146 | 4.9670 | 2.4712 |

#### u=0.7
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.1809 | 0.1999 | 5.4861 | 4.9391 | 2.4565 |
| 50 | 0.1821 | 0.2008 | 5.4924 | 4.9470 | 2.4605 |
| 100 | 0.1802 | 0.2002 | 5.4961 | 4.9453 | 2.4738 |
| 200 | 0.1809 | 0.2002 | 5.4963 | 4.9391 | 2.4856 |
| 400 | 0.1809 | 0.1995 | 5.4921 | 4.9475 | 2.4676 |

#### u=0.9
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.1815 | 0.2000 | 5.4861 | 4.9430 | 2.4705 |
| 50 | 0.1796 | 0.1994 | 5.4897 | 4.9397 | 2.4848 |
| 100 | 0.1827 | 0.1998 | 5.4796 | 4.9229 | 2.4523 |
| 200 | 0.1815 | 0.2003 | 5.4815 | 4.9501 | 2.4633 |
| 400 | 0.1802 | 0.2005 | 5.4836 | 4.9470 | 2.4627 |

#### u=1.0
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.1809 | 0.1997 | 5.4814 | 4.9391 | 2.4674 |
| 50 | 0.1827 | 0.1998 | 5.4840 | 4.9518 | 2.4678 |
| 100 | 0.1796 | 0.1992 | 5.4830 | 4.9391 | 2.4638 |
| 200 | 0.1815 | 0.2002 | 5.4858 | 4.9391 | 2.4749 |
| 400 | 0.1809 | 0.1999 | 5.4881 | 4.9401 | 2.4744 |

## Key Findings

### 1. Single-step GLP error is negligible
- Maximum ΔpPPL = +0.135 (u=0.1, s=100), vs Reference 5.466
- For u >= 0.7, ΔpPPL < +0.03, within natural sampling noise (No Steering itself is +0.015)
- **Confirms: V1's pPPL degradation (5.47 -> 7.2-16.4) is NOT from single-step GLP error, but from error accumulation across 10 iterative mask-predict rounds**

### 2. Steps have zero effect on single-step error
- Column means are identical (5.522-5.523) across all 5 step values
- Same-u variation across steps is < 0.02, well within noise
- Flow matching ODE quality converges even at 25 steps for single-step application

### 3. u monotonically controls single-step error magnitude
- u=0.1: ΔpPPL ≈ +0.12 (add little noise, denoise little -> minimal correction, but introduces reconstruction error)
- u=0.5: ΔpPPL ≈ +0.05
- u=0.7-1.0: ΔpPPL ≈ +0.02 (nearly invisible)
- Larger u means more denoising, which paradoxically produces more natural outputs because GLP is better at recovering from high-noise states

### 4. Solubility is unaffected by single-token changes
- All groups: Sol Mean Prob ∈ [0.1976, 0.2008], Sol Ratio ∈ [17.5%, 18.3%]
- Changing 1 out of ~180 amino acids has negligible effect on a global sequence property
- Sol metric is uninformative for single-mask experiments; only meaningful for full iterative generation

### 5. Implication for iterative generation
- Even u=0.1 only introduces ΔpPPL ≈ +0.13 per application
- But V1 shows u=0.1 with 10 rounds produces ΔpPPL ≈ +10.9 (5.47 -> 16.35)
- This is ~84× amplification, not 10×, indicating **superlinear error accumulation**
- The GLP projection error at each step shifts activations slightly off the true ESM2 manifold, and subsequent layers/rounds compound this drift

## Comparison with V1

| Metric | V1 (10 rounds) | V2 (single mask) | Amplification |
|--------|----------------|-------------------|---------------|
| u=0.1 ΔpPPL | +10.9 | +0.13 | ~84× |
| u=0.5 ΔpPPL | +6.2 | +0.05 | ~124× |
| u=1.0 ΔpPPL | +1.7 | +0.02 | ~85× |
| L17 no GLP ΔpPPL | +1.5 | +0.01 | ~150× |

The amplification factor is roughly constant (~85-150×) across u values, suggesting error accumulation is a multiplicative process inherent to the iterative mask-predict scheme, not specific to GLP.

## Next Steps
- **V3 (Step-wise tracking)**: Record pPPL and sol at each of the 10 decoding rounds to visualize error accumulation trajectory
- Investigate error correction strategies (e.g., apply GLP only at final round, reduce mask_ratio)
