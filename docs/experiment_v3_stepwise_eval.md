# Experiment V3: Step-wise Evaluation — Tracking Error Accumulation Across Decoding Rounds

## Date: 2026-03-12 ~ 2026-03-13

## Motivation

V2 (`docs/experiment_v2_single_mask_eval.md`) demonstrated that single-step GLP error is small (max ΔpPPL = +0.135 for u=0.1, negligible for u >= 0.7). V1 (`docs/experiment_v1_comprehensive_eval.md`) showed that 10-round iterative mask-predict produces dramatically larger pPPL degradation (ΔpPPL up to +10.9), with amplification factors of ~85-150x relative to single-step error.

V3 bridges the gap between V1 and V2 by taking **snapshots after each decoding round** (rounds 1-10) and evaluating both pPPL and solubility at each step. This reveals the shape of the error accumulation trajectory — whether it is linear, superlinear, or saturating — and how different sampling strategies (nucleus vs greedy) affect the accumulation dynamics.

## Experimental Setup

### Generation Pipeline
- **Base model**: ESM2-650M (`esm2_t33_650M_UR50D`)
- **Steering**: Single-layer steering at Layer 17, norm-preserving rescale
- **GLP model**: `generative_latent_prior/runs/glp-esm2-650m-layer17-d6/`, 334M params
- **GLP projection**: SDEdit — normalize -> add noise to level u -> denoise -> denormalize
- **Iterative mask-predict**: Same as V1 — mask_ratio=0.1, 10 rounds (ceil(1.0/0.1)), each round masks ~10% of remaining candidate sites
- **Snapshots**: After each round, record the current sequence state -> evaluate sol + pPPL
- **Reference data**: `data/sol_easy.csv` (162 sequences, avg length 180.4, range 71-256)

### Two Sampling Modes

| Parameter | Nucleus (Run 1) | Greedy (Run 2) |
|-----------|-----------------|----------------|
| Temperature | 1.0 | 0.0 |
| top_p | 0.9 | N/A (argmax) |
| Output directory | `results/stepwise_eval/` | `results/stepwise_eval_greedy/` |

### Experimental Groups (32 per run)

#### Baselines (2 groups)

| # | Method | steering_vectors | glp_project_fn | N seqs |
|---|--------|------------------|----------------|--------|
| 1 | No Steering | None | None | 100 |
| 2 | L17 no GLP | sv_single (zeros except L17) | None | 100 |

#### GLP Groups (30 groups: 6 u x 5 steps)

| u \ steps | 25 | 50 | 100 | 200 | 400 |
|-----------|-----|-----|------|------|------|
| 0.1 | 100 | 100 | 100 | 100 | 100 |
| 0.3 | 100 | 100 | 100 | 100 | 100 |
| 0.5 | 100 | 100 | 100 | 100 | 100 |
| 0.7 | 100 | 100 | 100 | 100 | 100 |
| 0.9 | 100 | 100 | 100 | 100 | 100 |
| 1.0 | 100 | 100 | 100 | 100 | 100 |

Total per run: 32 groups x 100 sequences x 10 rounds = **32,000 sequence snapshots** evaluated.

### Evaluation Metrics
- **Sol Mean Prob**: Mean predicted solubility probability from oracle predictor (PropertyPredictor on ESM2-650M features)
- **Sol Ratio**: Fraction with prob >= 0.5
- **pPPL**: ESM2-3B pseudo-perplexity (lower = more natural). Primary metric.

### GPU Setup
- `CUDA_VISIBLE_DEVICES=4,5,6,7`
- Phase 1 (generation + sol eval): `cuda:0` (physical GPU 4)
- Phase 2 (pPPL eval): `cuda:0-3` (physical GPUs 4-7), 4-way parallel ESM2-3B

## Code Files Used

### New files
- `run_stepwise_eval.py` — main experiment script, implements `generate_iterative_with_snapshots()` for step-wise generation with intermediate recording, `plot_trajectories()` for per-steps line plots and final-round heatmaps

### Existing files used (no modifications)
- `steering_with_glp.py` — `build_glp_projection_fn()`, `steering_forward_with_glp()`, `evaluate_sol()`, `PropertyPredictor`
- `module/steerable_esm2.py` — `steering_forward()`
- `evaluate_ppl.py` — `compute_pseudo_perplexity_multi_gpu()`
- `utils/esm2_utils.py` — `load_esm2_model()`, `decode()`
- `utils/gen_utils.py` — `sample_top_p()`
- `generative_latent_prior/glp/denoiser.py` — `GLP` model class

## Data Files Used

- `data/sol_easy.csv` — 162 reference sequences
- `saved_steering_vectors/650M_sol_steering_vectors.pt` — steering vectors (pos_sv, neg_sv)
- `saved_predictors/sol_predictor_final.pt` — solubility oracle predictor
- `generative_latent_prior/runs/glp-esm2-650m-layer17-d6/` — trained GLP checkpoint (`final`)
- `data/uniref50_random1000.csv` — 1000 random UniRef50 proteins sampled for baseline comparison

## Generated Data Files

### `results/stepwise_eval/` (Nucleus sampling, temperature=1.0, top_p=0.9)
- `trajectories.json` — all methods' trajectories with per-round pPPL and sol metrics
- Per-method directories (e.g., `No_Steering/`, `L17_no_GLP/`, `L17_GLP_u0.1_s25/`, ...):
  - `snapshots.json` — cached generated sequences for each round
  - `round_0.csv` through `round_9.csv` — sequences at each round
- `trajectory_steps25.png/pdf` through `trajectory_steps400.png/pdf` — per-steps line plots (pPPL + sol vs round)
- `final_round_heatmap.png/pdf` — u x steps heatmap of final round 10 metrics

### `results/stepwise_eval_greedy/` (Greedy sampling, temperature=0.0)
- Same structure as above

### External baseline
- `data/uniref50_random1000.csv` — 1000 random UniRef50 proteins: pPPL mean=6.0174, median=4.6809, std=3.8624, sol_mean_prob=0.4221, sol_ratio=38.8%

## Repo Changes

### Added to `run_stepwise_eval.py`
- `argparse` support: `--temperature` (default 1.0), `--top_p` (default 0.9), `--output_dir` (default auto), `--n_gen` (default 100)
- Auto output directory selection: temperature=0.0 -> `results/stepwise_eval_greedy`, else `results/stepwise_eval`

## Results

### Reference Baseline

| Metric | Value |
|--------|-------|
| sol_mean_prob | 0.1987 |
| sol_ratio | 17.9% |
| pPPL mean | 5.5246 |
| pPPL median | 5.1263 |

### UniRef50 Random 1000 Baseline

| Metric | Value |
|--------|-------|
| pPPL mean | 6.0174 |
| pPPL median | 4.6809 |
| pPPL std | 3.8624 |
| sol_mean_prob | 0.4221 |
| sol_ratio | 38.8% |

---

### Nucleus Sampling Trajectories (temperature=1.0, top_p=0.9)

#### No Steering — Nucleus

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 5.9405 | 5.3757 | 2.7313 | 0.2211 | 20% |
| 2 | 6.2756 | 5.5567 | 3.1031 | 0.2198 | 20% |
| 3 | 6.5949 | 5.7741 | 3.2907 | 0.2326 | 20% |
| 4 | 6.7169 | 6.0815 | 3.2172 | 0.2360 | 18% |
| 5 | 6.8514 | 6.0991 | 3.2253 | 0.2254 | 19% |
| 6 | 6.8666 | 6.0333 | 3.1698 | 0.2367 | 23% |
| 7 | 6.9717 | 6.2176 | 3.1313 | 0.2273 | 20% |
| 8 | 6.9489 | 6.0707 | 3.1072 | 0.2517 | 24% |
| 9 | 6.9359 | 5.8894 | 3.1295 | 0.2521 | 21% |
| 10 | 6.8897 | 5.9514 | 3.0666 | 0.2434 | 22% |

#### L17 no GLP — Nucleus

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 5.9218 | 5.4468 | 2.3596 | 0.2282 | 20% |
| 2 | 6.5192 | 5.7033 | 3.2151 | 0.2182 | 18% |
| 3 | 6.8200 | 5.9498 | 3.4320 | 0.2321 | 20% |
| 4 | 7.1043 | 6.0424 | 3.6646 | 0.2394 | 22% |
| 5 | 7.1774 | 6.1698 | 3.4935 | 0.2426 | 20% |
| 6 | 7.2621 | 6.2463 | 3.4350 | 0.2520 | 22% |
| 7 | 7.3128 | 6.3929 | 3.4600 | 0.2515 | 21% |
| 8 | 7.3022 | 6.3379 | 3.4321 | 0.2458 | 19% |
| 9 | 7.2168 | 6.1743 | 3.2924 | 0.2552 | 19% |
| 10 | 7.1817 | 6.2589 | 3.2887 | 0.2778 | 28% |

#### L17+GLP u=0.1 s=100 — Nucleus

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 7.8903 | 7.6294 | 2.2908 | 0.1776 | 14% |
| 2 | 10.5361 | 10.1553 | 2.6136 | 0.1521 | 13% |
| 3 | 13.3818 | 12.9840 | 2.8290 | 0.1379 | 7% |
| 4 | 15.9168 | 16.1390 | 2.5975 | 0.2114 | 15% |
| 5 | 16.9297 | 17.1864 | 1.4912 | 0.2359 | 17% |
| 6 | 17.2071 | 17.3347 | 0.9535 | 0.3146 | 21% |
| 7 | 17.0062 | 16.9883 | 0.9690 | 0.3866 | 29% |
| 8 | 16.8096 | 16.6660 | 0.9465 | 0.3970 | 32% |
| 9 | 16.4793 | 16.4817 | 0.9003 | 0.4695 | 43% |
| 10 | **16.3137** | 16.2768 | 0.9123 | **0.5077** | **52%** |

#### L17+GLP u=0.5 s=100 — Nucleus

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 6.7054 | 6.1748 | 2.7724 | 0.1972 | 13% |
| 2 | 7.5784 | 6.8934 | 2.9860 | 0.2198 | 18% |
| 3 | 8.5861 | 7.6140 | 3.2755 | 0.2322 | 20% |
| 4 | 9.2809 | 8.0856 | 3.4409 | 0.2289 | 18% |
| 5 | 9.7591 | 8.9111 | 3.2833 | 0.2427 | 19% |
| 6 | 10.2869 | 9.8604 | 3.2852 | 0.2438 | 21% |
| 7 | 10.6279 | 10.3981 | 3.2579 | 0.2401 | 18% |
| 8 | 10.9517 | 10.7892 | 3.1207 | 0.2431 | 17% |
| 9 | 11.2606 | 11.2124 | 3.1411 | 0.2481 | 19% |
| 10 | **11.4061** | 11.4315 | 3.0937 | 0.2316 | 16% |

#### L17+GLP u=1.0 s=100 — Nucleus

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 6.0994 | 5.4833 | 2.7573 | 0.2285 | 22% |
| 2 | 6.3987 | 5.6997 | 2.8149 | 0.2357 | 21% |
| 3 | 6.7643 | 5.8939 | 3.3063 | 0.2403 | 19% |
| 4 | 6.8911 | 6.1155 | 3.2043 | 0.2503 | 19% |
| 5 | 7.1010 | 6.2666 | 3.3598 | 0.2733 | 22% |
| 6 | 7.2830 | 6.3801 | 3.5607 | 0.2811 | 25% |
| 7 | 7.3113 | 6.2399 | 3.4701 | 0.2983 | 27% |
| 8 | 7.2693 | 6.2708 | 3.4471 | 0.3061 | 31% |
| 9 | 7.2079 | 6.1840 | 3.2992 | 0.3065 | 30% |
| 10 | **7.1587** | 6.3057 | 3.2688 | 0.3229 | 28% |

---

### Greedy Sampling Trajectories (temperature=0.0)

#### No Steering — Greedy

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 5.6721 | 5.1756 | 2.5249 | 0.2339 | 22% |
| 2 | 5.7310 | 4.8816 | 2.9970 | 0.2715 | 21% |
| 3 | 5.3219 | 4.7249 | 2.4470 | 0.3100 | 32% |
| 4 | 4.8852 | 4.3478 | 1.8708 | 0.3400 | 32% |
| 5 | 4.5121 | 4.1480 | 1.4918 | 0.3727 | 34% |
| 6 | 4.1594 | 3.8564 | 1.2594 | 0.4059 | 40% |
| 7 | 3.7498 | 3.4527 | 1.0505 | 0.4352 | 43% |
| 8 | 3.4065 | 3.1275 | 0.9333 | 0.4793 | 47% |
| 9 | 2.9815 | 2.8025 | 0.7847 | 0.4993 | 50% |
| 10 | **2.7011** | 2.5605 | 0.7748 | **0.5178** | **49%** |

#### L17 no GLP — Greedy

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 5.6360 | 5.1286 | 2.4459 | 0.2518 | 24% |
| 2 | 5.4308 | 5.1633 | 2.2362 | 0.2631 | 24% |
| 3 | 5.1709 | 4.8470 | 1.9821 | 0.2966 | 28% |
| 4 | 4.8690 | 4.5081 | 1.6346 | 0.3298 | 32% |
| 5 | 4.4740 | 4.2894 | 1.3759 | 0.3740 | 36% |
| 6 | 4.0863 | 3.9530 | 1.1253 | 0.3997 | 39% |
| 7 | 3.7250 | 3.5127 | 1.0139 | 0.4427 | 43% |
| 8 | 3.3737 | 3.1649 | 0.9309 | 0.4644 | 47% |
| 9 | 3.0329 | 2.8282 | 0.9350 | 0.4778 | 49% |
| 10 | **2.7579** | 2.6041 | 0.8884 | **0.4968** | **53%** |

#### L17+GLP u=0.1 s=100 — Greedy

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 8.1066 | 7.6254 | 2.4674 | 0.1980 | 17% |
| 2 | 11.0232 | 10.5590 | 2.5354 | 0.1779 | 13% |
| 3 | 14.1213 | 13.4439 | 2.8177 | 0.1330 | 6% |
| 4 | 16.5048 | 16.7738 | 2.2751 | 0.1630 | 12% |
| 5 | 17.9507 | 18.0535 | 1.3927 | 0.1878 | 8% |
| 6 | 18.2460 | 18.3444 | 0.9681 | 0.1941 | 7% |
| 7 | 18.3679 | 18.5664 | 0.8694 | 0.2503 | 15% |
| 8 | 18.4851 | 18.6008 | 0.8714 | 0.2515 | 17% |
| 9 | 18.3700 | 18.4868 | 0.9412 | 0.3090 | 23% |
| 10 | **18.4267** | 18.4756 | 1.0486 | 0.3061 | 23% |

#### L17+GLP u=0.5 s=100 — Greedy

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 6.1720 | 5.6215 | 2.5867 | 0.2279 | 22% |
| 2 | 6.4912 | 6.0218 | 2.2241 | 0.2350 | 20% |
| 3 | 6.8490 | 6.5338 | 2.1358 | 0.2586 | 22% |
| 4 | 6.9398 | 6.5614 | 2.0862 | 0.2739 | 24% |
| 5 | 6.9755 | 6.7687 | 2.0903 | 0.2884 | 26% |
| 6 | 6.9687 | 6.8575 | 2.0625 | 0.2826 | 23% |
| 7 | 6.7976 | 6.7241 | 2.2165 | 0.3007 | 26% |
| 8 | 6.4489 | 6.2172 | 1.9908 | 0.3159 | 30% |
| 9 | 5.9848 | 5.7039 | 2.0128 | 0.3189 | 28% |
| 10 | **5.6203** | 5.3129 | 2.0802 | 0.3266 | 25% |

#### L17+GLP u=1.0 s=100 — Greedy

| Round | pPPL Mean | pPPL Median | pPPL Std | Sol Mean Prob | Sol % |
|-------|-----------|-------------|----------|---------------|-------|
| 1 | 5.7766 | 5.0931 | 2.7603 | 0.2290 | 20% |
| 2 | 5.6054 | 4.9415 | 2.6630 | 0.2585 | 22% |
| 3 | 5.3148 | 4.8343 | 2.2223 | 0.2784 | 25% |
| 4 | 5.0157 | 4.5886 | 1.9099 | 0.3034 | 29% |
| 5 | 4.6505 | 4.4027 | 1.5370 | 0.3369 | 33% |
| 6 | 4.2571 | 4.0170 | 1.2884 | 0.3646 | 34% |
| 7 | 3.8859 | 3.8049 | 1.1022 | 0.4046 | 42% |
| 8 | 3.5235 | 3.3899 | 1.1141 | 0.4375 | 42% |
| 9 | 3.1069 | 2.8731 | 0.9415 | 0.4897 | 52% |
| 10 | **2.8146** | 2.5557 | 1.0094 | **0.5048** | **51%** |

---

### Final Round (Round 10) pPPL Heatmap — Nucleus Sampling

| u \ steps | 25 | 50 | 100 | 200 | 400 | Row Mean |
|-----------|--------|--------|--------|--------|--------|----------|
| **0.1** | 16.43 | 16.36 | 16.31 | 16.33 | 16.37 | **16.36** |
| **0.3** | 15.65 | 15.38 | 15.84 | 15.42 | 15.70 | **15.60** |
| **0.5** | 11.60 | 11.33 | 11.41 | 12.04 | 11.75 | **11.63** |
| **0.7** | 8.24 | 8.27 | 8.52 | 8.85 | 8.61 | **8.50** |
| **0.9** | 7.45 | 7.31 | 7.68 | 7.39 | 7.50 | **7.47** |
| **1.0** | 7.19 | 7.17 | 7.16 | 7.40 | 7.09 | **7.20** |
| **Col Mean** | **11.09** | **10.97** | **11.15** | **11.24** | **11.17** | |

Reference: No Steering = 6.89, L17 no GLP = 7.18

### Final Round (Round 10) pPPL Heatmap — Greedy Sampling

| u \ steps | 25 | 50 | 100 | 200 | 400 | Row Mean |
|-----------|--------|--------|--------|--------|--------|----------|
| **0.1** | 18.32 | 18.21 | 18.43 | 18.43 | 18.39 | **18.36** |
| **0.3** | 14.37 | 14.00 | 14.27 | 13.94 | 13.95 | **14.11** |
| **0.5** | 5.93 | 5.59 | 5.62 | 5.87 | 5.80 | **5.76** |
| **0.7** | 3.12 | 3.04 | 3.08 | 3.17 | 3.11 | **3.10** |
| **0.9** | 2.81 | 2.72 | 2.71 | 2.84 | 2.85 | **2.79** |
| **1.0** | 2.76 | 2.71 | 2.81 | 2.71 | 2.72 | **2.74** |
| **Col Mean** | **7.89** | **7.71** | **7.82** | **7.83** | **7.80** | |

Reference: No Steering = 2.70, L17 no GLP = 2.76

### Final Round (Round 10) Sol Mean Prob Heatmap — Nucleus Sampling

| u \ steps | 25 | 50 | 100 | 200 | 400 | Row Mean |
|-----------|--------|--------|--------|--------|--------|----------|
| **0.1** | 0.4515 | 0.4670 | 0.5077 | 0.4391 | 0.4890 | **0.4709** |
| **0.3** | 0.4034 | 0.3785 | 0.3662 | 0.3445 | 0.3630 | **0.3711** |
| **0.5** | 0.2180 | 0.2377 | 0.2316 | 0.2753 | 0.2280 | **0.2381** |
| **0.7** | 0.2498 | 0.2889 | 0.3417 | 0.2681 | 0.2542 | **0.2805** |
| **0.9** | 0.2623 | 0.3123 | 0.3190 | 0.2963 | 0.3088 | **0.2997** |
| **1.0** | 0.3012 | 0.2911 | 0.3229 | 0.2674 | 0.2599 | **0.2885** |

### Final Round (Round 10) Sol Mean Prob Heatmap — Greedy Sampling

| u \ steps | 25 | 50 | 100 | 200 | 400 | Row Mean |
|-----------|--------|--------|--------|--------|--------|----------|
| **0.1** | 0.3577 | 0.3555 | 0.3061 | 0.3418 | 0.3530 | **0.3428** |
| **0.3** | 0.3909 | 0.4183 | 0.3495 | 0.3523 | 0.3471 | **0.3716** |
| **0.5** | 0.3076 | 0.3736 | 0.3266 | 0.3442 | 0.3204 | **0.3345** |
| **0.7** | 0.4760 | 0.4653 | 0.4952 | 0.4943 | 0.4603 | **0.4782** |
| **0.9** | 0.4923 | 0.4867 | 0.5248 | 0.4729 | 0.4995 | **0.4952** |
| **1.0** | 0.5279 | 0.5043 | 0.5048 | 0.5043 | 0.4522 | **0.4987** |

---

### Comparison Table: Nucleus vs Greedy at Round 10

| Method | Nucleus pPPL | Greedy pPPL | Nucleus Sol Prob | Greedy Sol Prob | Nucleus Sol % | Greedy Sol % |
|--------|-------------|-------------|-----------------|-----------------|---------------|--------------|
| **Reference** | 5.5246 | 5.5246 | 0.1987 | 0.1987 | 17.9% | 17.9% |
| **No Steering** | 6.89 | 2.70 | 0.2434 | 0.5178 | 22% | 49% |
| **L17 no GLP** | 7.18 | 2.76 | 0.2778 | 0.4968 | 28% | 53% |
| **u=0.1 s=100** | 16.31 | 18.43 | 0.5077 | 0.3061 | 52% | 23% |
| **u=0.3 s=100** | 15.84 | 14.27 | 0.3662 | 0.3495 | 32% | 27% |
| **u=0.5 s=100** | 11.41 | 5.62 | 0.2316 | 0.3266 | 16% | 25% |
| **u=0.7 s=100** | 8.52 | 3.08 | 0.3417 | 0.4952 | 33% | 50% |
| **u=0.9 s=100** | 7.68 | 2.71 | 0.3190 | 0.5248 | 33% | 54% |
| **u=1.0 s=100** | 7.16 | 2.81 | 0.3229 | 0.5048 | 28% | 51% |
| **UniRef50 (1000)** | 6.02 | — | 0.4221 | — | 38.8% | — |

---

### Detailed Nucleus Trajectories for All Steps (s=100, Selected u Values)

#### u=0.3 s=100 — Nucleus

| Round | pPPL Mean | Sol Mean Prob | Sol % |
|-------|-----------|---------------|-------|
| 1 | 7.7261 | 0.1942 | 16% |
| 2 | 9.8767 | 0.1629 | 11% |
| 3 | 12.3474 | 0.1585 | 11% |
| 4 | 14.1920 | 0.1794 | 14% |
| 5 | 15.4713 | 0.1875 | 11% |
| 6 | 16.1150 | 0.2421 | 18% |
| 7 | 16.3169 | 0.2711 | 19% |
| 8 | 16.1952 | 0.2749 | 18% |
| 9 | 16.0129 | 0.3362 | 27% |
| 10 | 15.8375 | 0.3662 | 32% |

#### u=0.7 s=100 — Nucleus

| Round | pPPL Mean | Sol Mean Prob | Sol % |
|-------|-----------|---------------|-------|
| 1 | 6.1514 | 0.2155 | 21% |
| 2 | 6.6382 | 0.2242 | 21% |
| 3 | 7.1096 | 0.2312 | 20% |
| 4 | 7.4311 | 0.2472 | 21% |
| 5 | 7.7386 | 0.2592 | 21% |
| 6 | 8.0970 | 0.2756 | 25% |
| 7 | 8.2908 | 0.2799 | 28% |
| 8 | 8.4742 | 0.3029 | 30% |
| 9 | 8.5311 | 0.3327 | 33% |
| 10 | 8.5153 | 0.3417 | 33% |

#### u=0.9 s=100 — Nucleus

| Round | pPPL Mean | Sol Mean Prob | Sol % |
|-------|-----------|---------------|-------|
| 1 | 6.1431 | 0.2345 | 24% |
| 2 | 6.6929 | 0.2436 | 23% |
| 3 | 7.0368 | 0.2529 | 26% |
| 4 | 7.2858 | 0.2754 | 25% |
| 5 | 7.5414 | 0.2814 | 26% |
| 6 | 7.5850 | 0.3021 | 25% |
| 7 | 7.6380 | 0.3027 | 32% |
| 8 | 7.7157 | 0.3218 | 32% |
| 9 | 7.6971 | 0.3235 | 32% |
| 10 | 7.6835 | 0.3190 | 33% |

---

### Detailed Greedy Trajectories (s=100, Selected u Values)

#### u=0.3 s=100 — Greedy

| Round | pPPL Mean | Sol Mean Prob | Sol % |
|-------|-----------|---------------|-------|
| 1 | 7.4924 | 0.1956 | 19% |
| 2 | 9.6161 | 0.1910 | 17% |
| 3 | 11.8846 | 0.1968 | 15% |
| 4 | 13.8748 | 0.1889 | 12% |
| 5 | 15.2405 | 0.2354 | 19% |
| 6 | 15.8934 | 0.2861 | 20% |
| 7 | 15.8976 | 0.2929 | 21% |
| 8 | 15.4668 | 0.3174 | 24% |
| 9 | 14.8714 | 0.3273 | 22% |
| 10 | 14.2680 | 0.3495 | 27% |

#### u=0.7 s=100 — Greedy

| Round | pPPL Mean | Sol Mean Prob | Sol % |
|-------|-----------|---------------|-------|
| 1 | 5.8178 | 0.2416 | 21% |
| 2 | 5.6485 | 0.2615 | 22% |
| 3 | 5.5948 | 0.2793 | 24% |
| 4 | 5.2127 | 0.3253 | 31% |
| 5 | 4.8778 | 0.3408 | 33% |
| 6 | 4.5389 | 0.3717 | 37% |
| 7 | 4.1946 | 0.3997 | 38% |
| 8 | 3.7845 | 0.4268 | 40% |
| 9 | 3.3840 | 0.4705 | 48% |
| 10 | 3.0796 | 0.4952 | 50% |

#### u=0.9 s=100 — Greedy

| Round | pPPL Mean | Sol Mean Prob | Sol % |
|-------|-----------|---------------|-------|
| 1 | 5.5620 | 0.2295 | 22% |
| 2 | 5.3978 | 0.2670 | 27% |
| 3 | 5.2818 | 0.2797 | 25% |
| 4 | 5.0344 | 0.3039 | 29% |
| 5 | 4.6847 | 0.3547 | 33% |
| 6 | 4.3046 | 0.3922 | 39% |
| 7 | 3.8068 | 0.4538 | 45% |
| 8 | 3.4026 | 0.4892 | 53% |
| 9 | 3.0111 | 0.5165 | 57% |
| 10 | 2.7148 | 0.5248 | 54% |

---

## Key Findings

### 1. Nucleus sampling: superlinear error accumulation

Under nucleus sampling (temperature=1.0, top_p=0.9), pPPL degrades monotonically with each round for all methods, confirming the error accumulation hypothesis from V1/V2.

- **u=0.1**: pPPL rockets from 7.89 (round 1) to 16.31 (round 10). The trajectory is superlinear — most of the damage occurs in rounds 1-5 (pPPL gain of +9.04), while rounds 5-10 add only +0.60. The pPPL effectively **saturates around round 5-6** at ~17, indicating the sequences have degenerated so far from natural proteins that further mask-predict rounds cycle through the same degraded modes.
- **u=0.5**: pPPL rises roughly linearly from 6.71 to 11.41, gaining ~0.52 per round. No saturation observed within 10 rounds.
- **u=0.7-1.0**: pPPL increases modestly (6.1 -> 7.2-8.5), with diminishing increments suggesting eventual saturation around pPPL ~8-9.

### 2. Nucleus sampling: sol shows a V-shape (fake improvement)

For low u (0.1, 0.3), solubility probability **initially drops** (rounds 1-3), then **recovers and exceeds the reference** by round 10:
- u=0.1 s=100: sol goes 0.178 -> 0.138 (round 3) -> 0.508 (round 10)
- This "improvement" is an artifact: the GLP-degraded sequences converge toward repetitive low-entropy modes that the sol predictor interprets as high-solubility. The simultaneously high pPPL (~16) confirms these are not natural proteins.

### 3. Greedy sampling: mode collapse drives pPPL below reference

Under greedy sampling (temperature=0.0), the dynamics are radically different:

- **All methods** (including No Steering and L17 no GLP) show **decreasing pPPL** with more rounds
- No Steering greedy: pPPL goes from 5.67 (round 1) to **2.70** (round 10), well below the reference (5.52)
- This is **mode collapse**: greedy iterative mask-predict converges to high-probability modes of ESM2, producing sequences with artificially low perplexity and inflated solubility (sol > 0.50)
- These sequences are not diverse or natural — they are ESM2's "favorite" patterns repeated

### 4. Greedy + GLP u=0.5: pPPL converges near reference

An interesting observation: **greedy u=0.5 s=100 produces pPPL = 5.62 at round 10**, remarkably close to the reference (5.52). The GLP projection at u=0.5 introduces just enough perturbation to counterbalance the mode-collapse tendency of greedy decoding, achieving an approximate equilibrium. However, this balance is likely coincidental and does not indicate that the generated sequences are natural.

### 5. Steps have no effect in either sampling mode

Across both nucleus and greedy, the "steps" parameter (25, 50, 100, 200, 400) produces negligible variation:
- Nucleus: column means for pPPL at round 10 are 11.09, 10.97, 11.15, 11.24, 11.17 — indistinguishable
- Greedy: column means are 7.89, 7.71, 7.82, 7.83, 7.80 — indistinguishable
- This confirms V2's finding that the flow matching ODE converges even at 25 steps. Steps can be fixed at 25 for all future experiments.

### 6. Low u (0.1-0.3) is destructive regardless of sampling mode

Both nucleus and greedy with u=0.1 show catastrophic pPPL degradation:
- Nucleus u=0.1: pPPL = 16.31 (round 10)
- Greedy u=0.1: pPPL = 18.43 (round 10, even worse!)

Greedy u=0.1 is the worst configuration overall (pPPL ~18.4), because mode collapse cannot rescue sequences that have been destroyed by aggressive GLP projection — the greedy decoder converges to the modes of a heavily distorted representation space.

### 7. High u (>= 0.9) approximates identity

For u >= 0.9, the GLP projection adds very little noise and denoises very little, making it nearly an identity function:
- Nucleus u=1.0 s=100: pPPL = 7.16, vs L17 no GLP = 7.18
- Greedy u=1.0 s=100: pPPL = 2.81, vs L17 no GLP = 2.76

The GLP has negligible effect at high u, consistent with V2 findings (single-step ΔpPPL < 0.02 for u=1.0).

### 8. Comparison with V1/V2 amplification factors

Using steps=100 values at round 10 (nucleus) vs V2 single-step ΔpPPL:

| u | V2 Single-Step ΔpPPL | V3 Round 10 ΔpPPL (vs ref) | Amplification Factor |
|---|----------------------|---------------------------|---------------------|
| 0.1 | +0.13 | +10.79 | ~83x |
| 0.3 | +0.09 | +10.31 | ~115x |
| 0.5 | +0.05 | +5.88 | ~118x |
| 0.7 | +0.03 | +2.99 | ~100x |
| 0.9 | +0.02 | +2.16 | ~108x |
| 1.0 | +0.02 | +1.63 | ~82x |

The amplification factors (82-118x) are roughly consistent with V1/V2 estimates (85-150x), confirming superlinear error accumulation as the dominant effect.

### 9. pPPL trajectory shape by u value (Nucleus)

- **u=0.1**: Concave curve — rapid rise in rounds 1-5, saturation at ~17. The representation is so corrupted that additional mask-predict rounds cannot further degrade it.
- **u=0.3**: Similar concave shape, saturating at ~16.
- **u=0.5**: Approximately linear rise, ~0.5/round. No clear saturation within 10 rounds.
- **u=0.7-1.0**: Concave with early saturation around rounds 7-8. pPPL stabilizes at ~7.2-8.5.

### 10. pPPL trajectory shape by u value (Greedy)

- **u=0.1**: Concave rise to ~18.4, saturating around round 5-6. Greedy mode collapse cannot overcome catastrophic GLP corruption.
- **u=0.3**: Rise then gentle decline after round 6-7 (pPPL peaks ~16 then drops to ~14). The mode collapse partially counteracts GLP error.
- **u=0.5**: Rise, peak around round 4-5 (~7.0), then decline to ~5.6. Near-exact balance point.
- **u=0.7-1.0**: Monotonically decreasing — mode collapse dominates, driving pPPL from ~5.5-5.8 down to ~2.7-3.1.

## Summary

V3 conclusively shows that the pPPL degradation observed in V1 is caused by superlinear error accumulation across iterative mask-predict rounds, not by single-step GLP error (which V2 showed is small). The accumulation dynamics differ fundamentally between sampling strategies:

- **Nucleus sampling**: error accumulates monotonically, with saturation for extreme u values
- **Greedy sampling**: mode collapse drives pPPL artificially low, creating sequences that look "natural" by pPPL but are repetitive and low-diversity

Neither sampling strategy produces genuinely improved protein sequences through steering+GLP. The fundamental challenge is that even small per-step errors compound multiplicatively across 10 rounds.

## Next Steps

- Investigate error correction strategies: apply GLP only at selected rounds, or reduce number of rounds
- Explore adaptive u scheduling (e.g., higher u in early rounds, lower u in later rounds)
- Measure sequence diversity metrics (e.g., pairwise identity) to quantify mode collapse in greedy
- Consider alternative generation schemes that avoid iterative mask-predict entirely
