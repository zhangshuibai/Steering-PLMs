# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Review Workflow (MANDATORY)

After writing a new batch of code or making non-trivial modifications to existing code, **always run Codex CLI with gpt-5.4 at xhigh reasoning effort** to review the changes, then consider the suggestions before proceeding.

Command pattern:
```bash
export PATH="$HOME/.npm-global/bin:$PATH"
codex exec -m gpt-5.4 -c model_reasoning_effort="xhigh" \
  --sandbox read-only --skip-git-repo-check \
  "REVIEW_PROMPT_HERE" > /tmp/codex_review/<name>.log 2>&1 &
```

Exception: If the user's Codex quota is exhausted (ChatGPT Plus account), skip the review and notify the user.

The review should:
1. Identify BLOCKER / HIGH / MEDIUM / NITPICK issues with exact file:line references
2. Be done in background (not blocking user interaction)
3. Result in a short summary back to the user on which suggestions were accepted and applied

Do not launch long-running experiments before at least one round of Codex review for non-trivial changes.

## Project Overview

Implementation of ICML'25 paper "Steering Protein Language Models" (Huang et al.). Steers protein language models (ESM2, ESM3, ProLLaMA) to generate sequences with desired properties (solubility, thermostability) using steering vectors -- directional guidance derived from high/low property sequences, applied without fine-tuning. Optionally integrates Generative Latent Prior (GLP) for on-manifold projection to maintain sequence naturalness.

## Key Commands

### Full Solubility Pipeline (ESM2-650M)
```bash
bash run_sol_steering_pipeline.sh
```

### Individual Steps
```bash
# Step 1: Extract steering vectors
python extract_esm2_steering_vec.py --model "650M" --num_data 100 --property "sol" \
  --data_path "data/sol_filtered.csv" --theshold_pos 0.5 --theshold_neg 0.2

# Step 2: Generate with steering
python steering_esm2_generation.py --model "650M" --property "sol" \
  --ref_data_path "data/sol_easy.csv" --output_file "results/ESM2_gen_steering_sol_easy.csv" \
  --steering --n 100

# Step 3: Generate baseline (no --steering flag)
python steering_esm2_generation.py --model "650M" --property "sol" \
  --ref_data_path "data/sol_easy.csv" --output_file "results/ESM2_gen_no_steering_sol_easy.csv" --n 100

# Step 4: Oracle evaluation
python evaluate_generated_seqs.py --input_csv "results/ESM2_gen_steering_sol_easy.csv" \
  --predictor_path "saved_predictors/sol_predictor_final.pt" --property "sol"

# Pseudo-perplexity (naturalness)
python evaluate_ppl.py --input_csvs results/ESM2_gen_steering_sol_easy.csv --model 3B
```

### GLP On-Manifold Steering
```bash
bash run_glp_pipeline.sh   # Full GLP pipeline (extract activations, train GLP, steer+evaluate)

# Or just the steering step with a trained GLP:
python steering_with_glp.py --glp_path generative_latent_prior/runs/glp-esm2-650m-layer17-d6 \
  --gpu_gen cuda:0 --n_gen 100 --u 0.5
```

### Train Oracle Predictors
```bash
python train_sol_predictor.py      # Solubility (DeepSol dataset)
python train_therm_predictor.py    # Thermostability (Meltome dataset)
```

## Architecture

### Core Steering Mechanism
Steering vectors are injected at each transformer layer during generation:
1. `new_x = x + steering_vector[layer_idx]` -- add steering direction
2. `x = new_x * (||x|| / ||new_x||)` -- norm-preserving rescaling

Steering is implemented via monkey-patching a `steering_forward` method onto pretrained models (see `module/steerable_*.py`).

### Generation Process
Iterative masked token prediction: 10 rounds, each masking 10% of positions, then sampling replacements via top-p (p=0.9). After 10 rounds every position has been re-predicted once.

### Module Layout

- **`module/`** -- Steerable model forward passes (`steerable_esm2.py`, `steerable_esm3.py`, `steerable_prollama.py`)
- **`utils/`** -- Model loading, feature extraction, generation helpers (`esm2_utils.py`, `esm3_utils.py`, `gen_utils.py`, `opt_utils.py`)
- **Top-level scripts** -- Entry points for extraction, generation, optimization, evaluation
- **`generative_latent_prior/`** -- GLP submodule (separate paper by Luo et al.). Has its own conda environment (`environment.yaml`, Python 3.11) and training configs in `configs/`

### Data Flow
```
sol_filtered.csv -> extract steering vectors -> saved_steering_vectors/*.pt
                                                       |
ref sequences (sol_easy/hard.csv) + steering vectors -> steered generation -> results/*.csv
                                                                                    |
results/*.csv + saved_predictors/*.pt -> oracle evaluation -> results/*_scored.csv
```

### Key Model Dimensions
| Model | Layers | Hidden Dim | Steering Vector Shape |
|-------|--------|------------|----------------------|
| ESM2-650M | 33 | 1280 | (33, 1280) |
| ESM3-open | 48 | 1536 | (48, 1536) |

### Oracle Predictor Architecture
`Linear(hidden_dim, hidden_dim) -> GELU -> LayerNorm -> Linear(hidden_dim, 1)` on top of frozen ESM2 mean-pooled last-layer representations. Trained with BCEWithLogitsLoss.

## Important Notes

- The `--theshold_pos` / `--theshold_neg` flags use the misspelling "theshold" (not "threshold")
- Properties supported: `"sol"` (solubility, binary 0-1) and `"therm"` (thermostability, continuous temperature values with thresholds like 70.0/50.0)
- GLP uses a separate conda environment from the main codebase; see `generative_latent_prior/environment.yaml`
- Layer 17 is the key layer for single-layer steering in ESM2-650M (matches all-layer steering performance with better naturalness)
- Pipeline scripts have hardcoded Python/device paths that may need updating for your environment
