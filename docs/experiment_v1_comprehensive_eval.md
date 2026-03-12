# Experiment V1: Comprehensive GLP Steering Evaluation

## Date: 2026-03-12

## Motivation

GLP on-manifold projection after L17 steering was observed to increase pPPL (7.01 -> 10-16),
contradicting the expectation that projecting back onto the protein activation manifold would
preserve sequence naturalness. This experiment systematically explores the (u, steps) parameter
space to find an optimal configuration or confirm the GLP model needs further training.

## Experimental Setup

### Generation Pipeline
- **Base model**: ESM2-650M (`esm2_t33_650M_UR50D`)
- **Steering**: Single-layer steering at Layer 17, norm-preserving rescale
- **GLP model**: `generative_latent_prior/runs/glp-esm2-650m-layer17-d6/`, 334M params
- **GLP projection**: SDEdit — normalize -> add noise to level u -> denoise for `steps` steps -> denormalize
- **Sequence generation**: Iterative mask-predict, 10 rounds, mask_ratio=0.1, temperature=1.0, top_p=0.9
- **N generated**: 100 sequences per configuration
- **Reference data**: `data/sol_easy.csv` (162 sequences)

### Evaluation Metrics
- **Sol Ratio**: Fraction of sequences predicted soluble (>0.5) by oracle predictor (PropertyPredictor on ESM2-650M features)
- **pPPL**: ESM2-3B pseudo-perplexity (lower = more natural)

### Parameter Grid
- u (noise level): {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}
- steps (denoising steps): {25, 50, 100, 200, 400}
- Total GLP configurations: 30

### Baselines (4 groups, from existing CSVs)
1. **Reference**: Original sequences from `data/sol_easy.csv`
2. **No Steering**: `results/ESM2_gen_no_steering_sol_easy.csv`
3. **L17 Single (no GLP)**: `results/single_layer_steering/layer_17.csv`
4. **All-Layer Steering**: `results/ESM2_gen_steering_sol_easy.csv`

### GPU Setup
- CUDA_VISIBLE_DEVICES=4,5,6,7
- Phase 1 (generation + sol): cuda:0 (physical GPU 4)
- Phase 2 (pPPL): cuda:0-3 (physical GPUs 4-7), 4-way parallel ESM2-3B

## Results

### Baselines

| Method | N | Sol % | Sol Mean Prob | pPPL Mean | pPPL Median | pPPL Std |
|--------|---|-------|---------------|-----------|-------------|----------|
| Reference | 162 | 17.9% | 0.199 | 5.47 | 4.93 | 2.47 |
| No Steering | 100 | 24.0% | 0.281 | 7.19 | 6.22 | 3.55 |
| L17 Single (no GLP) | 100 | 32.0% | 0.330 | 7.01 | 6.25 | 3.11 |
| All-Layer Steering | 100 | 34.0% | 0.398 | 15.23 | 15.38 | 1.56 |

### GLP Results: Sol Ratio (%)

| | u=0.1 | u=0.3 | u=0.5 | u=0.7 | u=0.9 | u=1.0 |
|---|---|---|---|---|---|---|
| s=25 | 41 | 38 | 24 | 15 | 33 | 26 |
| s=50 | 47 | 21 | 22 | 23 | 24 | 22 |
| s=100 | **51** | 31 | 20 | 18 | 26 | 28 |
| s=200 | 40 | 27 | 18 | 23 | 26 | 25 |
| s=400 | 46 | 34 | 14 | 18 | 24 | 27 |

### GLP Results: pPPL Mean

| | u=0.1 | u=0.3 | u=0.5 | u=0.7 | u=0.9 | u=1.0 |
|---|---|---|---|---|---|---|
| s=25 | 16.53 | 15.73 | 11.83 | 8.54 | 7.24 | 7.44 |
| s=50 | 16.45 | 15.56 | 11.65 | 8.13 | 7.43 | 7.22 |
| s=100 | 16.35 | 15.57 | 11.66 | 8.38 | 7.58 | 7.21 |
| s=200 | 16.38 | 15.54 | 11.52 | 8.13 | 7.26 | 7.32 |
| s=400 | 16.40 | 15.45 | 11.77 | 8.58 | 7.29 | 7.30 |

### Combined View (Sol% / pPPL)

| | u=0.1 | u=0.3 | u=0.5 | u=0.7 | u=0.9 | u=1.0 |
|---|---|---|---|---|---|---|
| s=25 | 41/16.5 | 38/15.7 | 24/11.8 | 15/8.5 | 33/7.2 | 26/7.4 |
| s=50 | 47/16.5 | 21/15.6 | 22/11.6 | 23/8.1 | 24/7.4 | 22/7.2 |
| s=100 | **51/16.3** | 31/15.6 | 20/11.7 | 18/8.4 | 26/7.6 | 28/7.2 |
| s=200 | 40/16.4 | 27/15.5 | 18/11.5 | 23/8.1 | 26/7.3 | 25/7.3 |
| s=400 | 46/16.4 | 34/15.4 | 14/11.8 | 18/8.6 | 24/7.3 | 27/7.3 |

Baseline comparison: L17 no GLP = **32 / 7.01**

## Key Findings

### 1. Sol vs pPPL tradeoff is severe
- u=0.1 (strongest projection): Sol up to 51% but pPPL ~16.4 (worse than All-Layer steering)
- u=0.9-1.0 (weakest/no projection): pPPL ~7.2-7.4 (close to L17 no GLP) but Sol drops to ~24-28%
- **No GLP configuration simultaneously improves both Sol and pPPL over L17 no GLP (32% / 7.01)**

### 2. Steps have minimal effect
- For any fixed u, varying steps from 25 to 400 changes pPPL by < 0.3
- Flow Matching Euler ODE converges in ~25 steps; more steps trace the same trajectory more finely

### 3. u=1.0 confirms GLP reconstruction quality issue
- u=1.0 = pure noise input -> GLP generates from scratch -> output is "most on-manifold" according to GLP
- Result: Sol ~25% (≈ No Steering 24%), pPPL ~7.3 (slightly worse than No Steering 7.19)
- GLP's learned manifold doesn't perfectly match ESM2's internal representation space

### 4. Error accumulation from iterative mask-predict
- Each sequence requires 10 rounds of mask-predict (mask_ratio=0.1)
- Each round applies steering + GLP projection once -> 10 total applications
- GLP reconstruction error may accumulate across rounds, amplifying pPPL degradation
- **This motivates the single-mask experiment (V2) to isolate per-application error**

## Files
- Script: `run_comprehensive_eval.py` (main 25-group experiment)
- Script: `run_u1_eval.py` (u=1.0 supplement)
- Results: `results/comprehensive_eval/summary.json` (29 groups)
- Results: `results/comprehensive_eval/u1_summary.json` (5 groups, u=1.0)
- Generated CSVs: `results/comprehensive_eval/glp/u{u}_steps{steps}.csv`
- Bug fix: `generative_latent_prior/glp/flow_matching.py` — added `.clamp(max=len-1)` for u=1.0 support

## Detailed Per-Configuration Data

### u=0.1
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.41 | 0.480 | 16.527 | 16.652 | 0.966 |
| 50 | 0.47 | 0.493 | 16.451 | 16.355 | 0.725 |
| 100 | 0.51 | 0.512 | 16.346 | 16.377 | 0.929 |
| 200 | 0.40 | 0.474 | 16.379 | 16.300 | 0.927 |
| 400 | 0.46 | 0.486 | 16.403 | 16.408 | 0.876 |

### u=0.3
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.38 | 0.405 | 15.727 | 15.988 | 1.548 |
| 50 | 0.21 | 0.322 | 15.561 | 15.655 | 1.604 |
| 100 | 0.31 | 0.367 | 15.569 | 15.865 | 1.816 |
| 200 | 0.27 | 0.341 | 15.541 | 15.730 | 1.690 |
| 400 | 0.34 | 0.370 | 15.445 | 15.829 | 1.684 |

### u=0.5
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.24 | 0.266 | 11.829 | 11.799 | 3.331 |
| 50 | 0.22 | 0.268 | 11.646 | 11.320 | 3.242 |
| 100 | 0.20 | 0.240 | 11.656 | 11.454 | 3.147 |
| 200 | 0.18 | 0.215 | 11.520 | 11.663 | 3.033 |
| 400 | 0.14 | 0.193 | 11.772 | 11.609 | 3.155 |

### u=0.7
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.15 | 0.204 | 8.543 | 7.336 | 3.523 |
| 50 | 0.23 | 0.261 | 8.126 | 7.440 | 3.244 |
| 100 | 0.18 | 0.262 | 8.376 | 7.336 | 3.447 |
| 200 | 0.23 | 0.261 | 8.131 | 7.144 | 3.057 |
| 400 | 0.18 | 0.245 | 8.579 | 7.587 | 3.537 |

### u=0.9
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.33 | 0.314 | 7.239 | 6.470 | 3.498 |
| 50 | 0.24 | 0.270 | 7.427 | 6.516 | 3.366 |
| 100 | 0.26 | 0.292 | 7.585 | 6.564 | 3.573 |
| 200 | 0.26 | 0.288 | 7.264 | 6.586 | 3.216 |
| 400 | 0.24 | 0.276 | 7.293 | 6.277 | 3.356 |

### u=1.0
| steps | sol_ratio | sol_mean_prob | ppl_mean | ppl_median | ppl_std |
|-------|-----------|---------------|----------|------------|---------|
| 25 | 0.26 | 0.304 | 7.435 | 6.400 | 3.676 |
| 50 | 0.22 | 0.273 | 7.223 | 6.341 | 3.418 |
| 100 | 0.28 | 0.282 | 7.214 | 6.211 | 3.351 |
| 200 | 0.25 | 0.275 | 7.316 | 6.401 | 3.363 |
| 400 | 0.27 | 0.296 | 7.305 | 6.145 | 3.472 |
