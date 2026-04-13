# Steering Protein Language Models

Reproduction and extension of the ICML'25 paper *"Steering Protein Language Models"* (Huang et al., 2025).

This project steers protein language models (ESM2, ESM3, ProLLaMA) to generate sequences with desired properties (solubility, thermostability) using **steering vectors** — directional guidance derived from high/low property sequences, applied without fine-tuning. Optionally integrates a **Generative Latent Prior (GLP)** for on-manifold projection to preserve sequence naturalness.

## Method Overview

### Steering Vectors

Steering vectors capture the "direction" of a desired property in the model's activation space:

1. **Extract**: Run ESM2 on high-property and low-property sequences, collect per-layer mean representations
2. **Compute direction**: `steering_vector[layer] = mean(high_property) - mean(low_property)`
3. **Apply during generation**: At each transformer layer, add the steering vector and rescale to preserve the original activation norm:
   ```
   new_x = x + steering_vector[layer]
   x = new_x * (||x|| / ||new_x||)
   ```

### Generation Process

Iterative masked token prediction: 10 rounds, each masking 10% of positions, then sampling replacements via nucleus sampling (top-p=0.9). After 10 rounds every position has been re-predicted once.

### GLP On-Manifold Projection

The Generative Latent Prior (Luo et al., 2026) learns the distribution of ESM2 Layer 17 activations from natural proteins (UniRef50). After steering, it projects activations back to the learned manifold via SDEdit:

```
steered activation → normalize → add noise to level u → denoise back to manifold → denormalize
```

The parameter `u ∈ [0, 1]` controls the trade-off: small `u` preserves steering signal but stays off-manifold; large `u` returns to the manifold but washes out the steering effect.

## Key Results

### Solubility Steering (ESM2-650M, sol_easy)

| Method | Sol Ratio | pPPL ↓ | pLDDT ↑ | pTM ↑ |
|--------|:---------:|:------:|:-------:|:-----:|
| Reference (natural) | 17.9% | 5.47 | 0.726 | 0.728 |
| No Steering | 25.0% | 7.23 | 0.658 | 0.636 |
| **L17 Single-Layer Steering** | **22.0%** | **7.43** | **0.654** | **0.632** |
| All-Layer Steering | 25.0% | 15.26 | 0.336 | 0.145 |
| L17 + GLP (u=0.1) | 48.0% | 16.26 | 0.342 | 0.166 |
| L17 + GLP (u=0.9) | 29.0% | 7.19 | 0.652 | 0.631 |

- **L17 single-layer steering** achieves property improvement while preserving sequence naturalness (pPPL ≈ 7.4) and structural quality (pLDDT ≈ 0.65)
- **All-layer steering** boosts solubility but destroys protein foldability (pLDDT drops from 0.73 to 0.34)
- **GLP** shows a fundamental sol-vs-structure trade-off controlled by `u`

### Thermostability Steering (ESM2-650M, therm_easy)

| Method | Mean Tm (°C) | ΔTm | pPPL ↓ | pLDDT ↑ |
|--------|:------------:|:---:|:------:|:-------:|
| Reference | 49.4 | — | 5.27 | 0.745 |
| No Steering | 55.1 | +5.7 | 6.33 | 0.710 |
| **L17 Steering** | **55.4** | **+6.1** | **6.16** | **0.713** |
| All-Layer Steering | 48.5 | -0.8 | 5.63 | 0.455 |

L17 steering achieves +6.1°C improvement with minimal structural impact.

### Evaluation Metrics

| Metric | Tool | What it measures |
|--------|------|-----------------|
| **Sol Ratio / Tm** | Oracle predictor | Target property quality |
| **pPPL** | ESM2-3B | Sequence naturalness (lower = more protein-like) |
| **pLDDT** | ESMFold | Structural confidence per residue (0-1, higher = better) |
| **pTM** | ESMFold | Global structural quality (0-1, higher = better) |

## Project Structure

```
Steering-PLMs/
├── scripts/
│   ├── extraction/              # Extract steering vectors
│   │   ├── extract_esm2_steering_vec.py
│   │   ├── extract_esm3_steering_vec.py
│   │   └── extract_prollama_steering_vec.py
│   ├── generation/              # Sequence generation with steering
│   │   ├── steering_esm2_generation.py
│   │   ├── steering_esm3_generation.py
│   │   ├── steering_prollama_generation.py
│   │   └── steering_with_glp.py
│   ├── glp/                     # GLP on-manifold projection
│   │   └── steering_with_glp.py
│   ├── experiments/             # Analysis experiments (V1-V4)
│   │   ├── run_comprehensive_eval.py     # V1: GLP parameter grid
│   │   ├── run_single_mask_eval.py       # V2: single-step error isolation
│   │   ├── run_stepwise_eval.py          # V3: step-wise error tracking
│   │   ├── run_single_round_mask_ratio.py # V4: mask ratio analysis
│   │   └── exp_single_layer_steering.py  # 33-layer scan
│   └── pipelines/               # One-command pipeline scripts
│       ├── run_sol_steering_pipeline.sh
│       └── run_glp_pipeline.sh
├── evaluation/
│   ├── common.py                # Shared: PropertyPredictor, extract_features
│   ├── oracle/                  # Property prediction
│   │   ├── evaluate_oracle.py   # Unified eval for sol/therm
│   │   ├── solubility/          # Sol predictor weights + training
│   │   └── thermostability/     # Therm predictor weights + training
│   ├── ppl/                     # Pseudo-perplexity (ESM2-3B)
│   │   └── evaluate_ppl.py
│   └── esmfold/                 # Structure quality (ESMFold)
│       └── evaluate_esmfold.py
├── module/                      # Steerable model forward passes
│   ├── steerable_esm2.py
│   ├── steerable_esm3.py
│   └── steerable_prollama.py
├── utils/                       # Model loading, generation helpers
│   ├── esm2_utils.py
│   ├── esm3_utils.py
│   └── gen_utils.py
├── generative_latent_prior/     # GLP submodule (Luo et al.)
├── data/                        # Datasets
├── saved_steering_vectors/      # Extracted steering vectors
├── saved_predictors/            # Trained oracle predictors
├── results/                     # Original experiment results
├── new-results/                 # Reproduction results
└── docs/                        # Experiment documentation (V1-V3)
```

## Quick Start

### Environment Setup

```bash
# Main environment (ESM2, generation, oracle, pPPL)
conda create -n steering python=3.10 -y
conda activate steering
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install fair-esm pandas scipy scikit-learn tqdm einops transformers diffusers omegaconf safetensors

# ESMFold environment (separate due to torch>=2.6 requirement)
conda create -n esmfold python=3.10 -y
conda activate esmfold
pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124
pip install transformers pandas tqdm
```

### Full Solubility Pipeline

```bash
conda activate steering

# Step 1: Extract steering vectors
python scripts/extraction/extract_esm2_steering_vec.py \
    --model "650M" --num_data 100 --property "sol" \
    --data_path "data/sol_filtered.csv" \
    --theshold_pos 0.5 --theshold_neg 0.2

# Step 2: Generate with steering
python scripts/generation/steering_esm2_generation.py \
    --model "650M" --property "sol" \
    --ref_data_path "data/sol_easy.csv" \
    --output_file "results/steering_sol_easy.csv" \
    --steering --n 100 --seed 42

# Step 3: Oracle evaluation
python evaluation/oracle/evaluate_oracle.py \
    --input_csv "results/steering_sol_easy.csv" \
    --property sol --ref_csv "data/sol_easy.csv"

# Step 4: pPPL evaluation
python evaluation/ppl/evaluate_ppl.py \
    --input_csvs "results/steering_sol_easy.csv" \
    --model 3B --gpu_ids 0

# Step 5: ESMFold evaluation (use esmfold env)
conda activate esmfold
python evaluation/esmfold/evaluate_esmfold.py \
    --input_csvs "results/steering_sol_easy.csv"
```

### GLP On-Manifold Steering

```bash
# Download GLP checkpoint
huggingface-cli download Shuibai12138/glp-esm2-650m-layer17 \
    --local-dir generative_latent_prior/runs/glp-esm2-650m-layer17-d6

# Generate with L17 steering + GLP projection
python scripts/glp/steering_with_glp.py \
    --u 0.5 --n_gen 100 --output_dir results/glp
```

## Experiments

### V1: GLP Parameter Grid Search
Systematic evaluation of GLP noise level `u ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}` × denoising steps `∈ {25, 50, 100, 200, 400}`. Finding: severe sol-vs-pPPL trade-off; no configuration simultaneously improves both over L17 without GLP. See `docs/experiment_v1_comprehensive_eval.md`.

### V2: Single-Mask Error Isolation
Isolate single-step GLP error by masking one position at a time. Finding: single-step error is negligible (ΔpPPL < 0.14); the V1 degradation comes from superlinear error accumulation across 10 iterative rounds (~80-150× amplification). See `docs/experiment_v2_single_mask_eval.md`.

### V3: Step-wise Error Tracking
Record pPPL and sol at each of 10 decoding rounds, comparing nucleus vs greedy sampling. Finding: nucleus sampling shows monotonic error accumulation; greedy sampling causes mode collapse (pPPL drops below reference). See `docs/experiment_v3_stepwise_eval.md`.

### Single-Round Mask Ratio Analysis
Single round of mask-predict with mask_ratio from 0.1 to 1.0, 7 methods × 2000 sequences each. Evaluates the effect of modification extent on sol, pLDDT, and pTM in a single forward pass (no iterative accumulation). Results in `new-results/single_round_mask_ratio/`.

## Key Model Dimensions

| Model | Layers | Hidden Dim | Steering Vector Shape |
|-------|--------|------------|----------------------|
| ESM2-650M | 33 | 1280 | (33, 1280) |
| ESM3-open | 48 | 1536 | (48, 1536) |

## Important Notes

- The `--theshold_pos` / `--theshold_neg` flags use the misspelling "theshold" (not "threshold") — inherited from the original codebase
- Properties: `"sol"` (solubility, binary 0-1) and `"therm"` (thermostability, continuous Tm in °C)
- Layer 17 is the optimal single-layer steering target for ESM2-650M
- Generation scripts use fixed random seeds (`--seed 42`) for reproducibility
- GLP uses a separate conda env from the main codebase; see `generative_latent_prior/environment.yaml`

## Citation

```bibtex
@inproceedings{huang2025steering,
  title={Steering Protein Language Models},
  author={Huang, Long-Kai and Zhu, Rongyi and He, Bing and Yao, Jianhua},
  booktitle={International Conference on Machine Learning},
  year={2025},
  organization={PMLR}
}
```
