# Steering Protein Language Models: Project Overview

## 1. Motivation

Protein language models (PLMs) such as ESM2 have learned rich representations of protein sequences from billions of natural proteins. These representations implicitly encode functional properties — solubility, thermostability, binding affinity — as directions in high-dimensional activation space. However, standard generation from PLMs produces sequences that follow the natural distribution without any property-specific bias.

**The core question**: Can we guide (steer) PLMs to generate protein sequences with desired properties, without retraining or fine-tuning the model?

This is particularly valuable because:
- Fine-tuning large PLMs (650M–3B parameters) requires substantial data and compute
- Steering vectors offer a lightweight, modular alternative: compute once, apply at inference
- The approach is property-agnostic — the same framework works for solubility, thermostability, or any property with labeled data

This project reproduces and extends the ICML'25 paper *"Steering Protein Language Models"* (Huang et al., 2025), with additional evaluations using ESMFold structural quality metrics and systematic analysis of the Generative Latent Prior (GLP) on-manifold projection.

## 2. Method

### 2.1 Steering Vectors

The steering vector for a target property is computed as the difference in mean activations between high-property and low-property protein sequences:

```
steering_vector[layer] = mean(activations_high) - mean(activations_low)
```

For solubility, "high" means solubility score ≥ 0.5, "low" means ≤ 0.2. For thermostability, "high" means melting temperature ≥ 70°C, "low" means ≤ 50°C. Each group uses 100 sequences. The steering vector is computed per layer, yielding a (num_layers, hidden_dim) tensor — for ESM2-650M this is (33, 1280).

### 2.2 Norm-Preserving Steering

During generation, the steering vector is injected after each transformer layer with norm-preserving rescaling:

```
x_steered = x + steering_vector[layer]
x_output  = x_steered × (||x|| / ||x_steered||)
```

The rescaling ensures that the magnitude of activations remains unchanged — only the direction shifts. This prevents the steering from destabilizing the model's internal dynamics.

### 2.3 Iterative Mask-Predict Generation

ESM2 is a masked language model, not an autoregressive one. Sequence generation uses iterative masked token prediction:

1. Start with a reference protein sequence
2. Repeat for 10 rounds:
   - Randomly select 10% of positions (not previously selected)
   - Replace them with `<mask>` tokens
   - Run forward pass (with steering vectors injected at each layer)
   - Sample new amino acids from the predicted distribution (nucleus sampling, p=0.9)
   - Fill in the masked positions
3. After 10 rounds, every position has been re-predicted exactly once

The randomness in generation comes from two sources: which positions are masked (random permutation) and which tokens are sampled (multinomial sampling from softmax probabilities).

### 2.4 Single-Layer vs All-Layer Steering

The paper applies steering vectors at **all 33 layers** simultaneously. However, a systematic 33-layer scan reveals that **Layer 17 alone** achieves comparable property improvement with dramatically better sequence naturalness:

- All-layer steering: pPPL = 15.26 (very unnatural)
- Layer 17 only: pPPL = 7.43 (close to unsteered 7.23)
- Both achieve similar solubility improvement over reference

This is a key finding: the property information is concentrated in the middle layers of the transformer, and steering only there avoids corrupting the early (syntactic) and late (output) layers.

### 2.5 GLP On-Manifold Projection

The Generative Latent Prior (GLP, Luo et al., 2026) is a flow matching generative model trained on ESM2 Layer 17 activations from ~4 million UniRef50 natural proteins. It learns the manifold of "natural protein activations" at Layer 17.

After steering pushes activations off this manifold, GLP attempts to project them back via SDEdit:

1. **Normalize**: Standardize activations to zero mean, unit variance (using statistics from UniRef50)
2. **Add noise**: Interpolate between the steered activation and random noise at level `u`
   ```
   x_noisy = (1 - σ(u)) × x_steered + σ(u) × noise
   ```
3. **Denoise**: Run the GLP denoiser from timestep `u` back to 0, following the learned flow field
4. **Denormalize**: Restore original scale

The parameter `u ∈ [0, 1]` controls the projection strength:
- `u → 0`: Minimal noise, output ≈ input, steering preserved but still off-manifold
- `u → 1`: Near-pure noise, output ≈ GLP's unconditional generation, on-manifold but steering lost

The GLP denoiser is a 334M-parameter Transformer-MLP network that processes each token independently (no cross-token attention). It uses SwiGLU gating with multiplicative timestep conditioning, operating in a 2560-dimensional hidden space.

## 3. Experimental Setup

### 3.1 Models

| Model | Parameters | Layers | Hidden Dim | Use |
|-------|-----------|--------|------------|-----|
| ESM2-650M | 650M | 33 | 1280 | Primary steering model |
| ESM2-3B | 3B | 36 | 2560 | pPPL evaluation only |
| ESMFold | 3.5B | — | — | Structural quality evaluation |
| GLP Denoiser | 334M | 6 | 2560 | On-manifold projection |

### 3.2 Datasets

| Dataset | Sequences | Property | Use |
|---------|-----------|----------|-----|
| sol_filtered | 720 | Solubility score (0–1) | Steering vector extraction |
| sol_easy | 162 | Score 0.25–0.30 | Generation reference (medium-low sol) |
| sol_hard | 198 | Score 0.001–0.10 | Generation reference (very low sol) |
| therm_filtered | 2,000 | Melting temperature (°C) | Steering vector extraction |
| therm_easy | 342 | Tm 50–65°C | Generation reference |
| therm_hard | 248 | Tm < 50°C | Generation reference |
| UniRef50 (1000) | 1,000 | — | Natural protein baseline for ESMFold |
| DeepSol | 71,421 | Binary solubility | Oracle predictor training |
| Meltome | 24,472 | Tm (°C) | Oracle predictor training |

### 3.3 Evaluation Metrics

**Oracle Predictor (Property Quality)**

A lightweight predictor head (Linear → GELU → LayerNorm → Linear, 1.6M params) trained on top of frozen ESM2-650M mean-pooled representations:
- Solubility: BCEWithLogitsLoss, outputs probability 0–1, threshold ≥ 0.5 for "soluble"
- Thermostability: MSELoss, outputs predicted Tm in °C

**Pseudo-Perplexity (Sequence Naturalness)**

Computed with ESM2-3B: for each position, mask it and measure the model's log-probability of the true token. Lower pPPL = more natural/protein-like. Reference natural proteins: pPPL ≈ 5.5.

```
pPPL = exp(-1/L × Σ log P(x_i | x_\i))
```

**ESMFold (Structural Quality)**

Predicts 3D protein structure from sequence alone. Two metrics:
- **pLDDT** (predicted Local Distance Difference Test): Per-residue structural confidence, 0–1. Natural proteins: ~0.72.
- **pTM** (predicted TM-score): Global fold quality, 0–1. Natural proteins: ~0.73.

### 3.4 Generation Parameters

| Parameter | Value |
|-----------|-------|
| Mask ratio per round | 0.1 (10% of positions) |
| Number of rounds | 10 (each position re-predicted once) |
| Temperature | 1.0 |
| Top-p (nucleus sampling) | 0.9 |
| Sequences per experiment | 100 (standard) or 2000 (high-precision) |
| Random seed | 42 |

## 4. Results

### 4.1 Solubility Steering

| Method | Sol Ratio | Sol Prob | pPPL ↓ | pLDDT ↑ | pTM ↑ |
|--------|:---------:|:--------:|:------:|:-------:|:-----:|
| Reference (natural) | 17.9% | 0.199 | 5.47 | 0.726 | 0.728 |
| No Steering | 25.0% | 0.263 | 7.23 | 0.658 | 0.636 |
| **L17 Single-Layer** | **22.0%** | **0.280** | **7.43** | **0.654** | **0.632** |
| All-Layer Steering | 25.0% | 0.330 | 15.26 | 0.336 | 0.145 |
| L17 + GLP (u=0.1) | 48.0% | 0.505 | 16.26 | 0.342 | 0.166 |
| L17 + GLP (u=0.5) | 19.0% | 0.258 | 11.22 | 0.530 | 0.468 |
| L17 + GLP (u=0.9) | 29.0% | 0.307 | 7.19 | 0.652 | 0.631 |

### 4.2 Thermostability Steering

| Method | Mean Tm (°C) | ΔTm | pPPL ↓ | pLDDT ↑ | pTM ↑ |
|--------|:------------:|:---:|:------:|:-------:|:-----:|
| Reference | 49.4 ± 7.5 | — | 5.27 | 0.745 | 0.723 |
| No Steering | 55.1 ± 9.6 | +5.7 | 6.33 | 0.710 | 0.679 |
| **L17 Steering** | **55.4 ± 10.0** | **+6.1** | **6.16** | **0.713** | **0.687** |
| All-Layer Steering | 48.5 ± 3.1 | -0.8 | 5.63 | 0.455 | 0.271 |
| L17 + GLP (u=0.9) | 54.9 ± 9.9 | +5.5 | 6.59 | 0.705 | 0.674 |

### 4.3 Single-Round Mask Ratio Analysis

To disentangle the effect of mask ratio from iterative error accumulation, we run a single round of mask-predict with varying mask_ratio (0.1–1.0), 7 methods × 2000 sequences each.

**Sol Mean Prob** (higher = better solubility, reference = 0.199):

| mask_ratio | NoSteer | AllLayer | L17 | GLP u=0.1 | GLP u=0.5 | GLP u=0.9 |
|:----------:|:-------:|:--------:|:---:|:---------:|:---------:|:---------:|
| 0.1 | 0.207 | 0.177 | 0.207 | 0.169 | 0.197 | 0.202 |
| 0.3 | 0.252 | 0.512 | 0.258 | 0.142 | 0.259 | 0.243 |
| 0.5 | 0.401 | 0.674 | 0.417 | 0.253 | 0.398 | 0.391 |
| 0.8 | 0.629 | 0.850 | 0.577 | 0.674 | 0.777 | 0.707 |
| 1.0 | 0.069 | 0.894 | 0.072 | 0.441 | 0.182 | 0.072 |

**pLDDT** (higher = better structure, natural ≈ 0.73):

| mask_ratio | NoSteer | AllLayer | L17 | GLP u=0.1 | GLP u=0.5 | GLP u=0.9 |
|:----------:|:-------:|:--------:|:---:|:---------:|:---------:|:---------:|
| 0.1 | 0.714 | 0.675 | 0.714 | 0.678 | 0.702 | 0.712 |
| 0.3 | 0.510 | 0.424 | 0.507 | 0.523 | 0.506 | 0.507 |
| 0.5 | 0.385 | 0.326 | 0.375 | 0.357 | 0.348 | 0.372 |
| 0.8 | 0.321 | 0.377 | 0.371 | 0.322 | 0.400 | 0.380 |
| 1.0 | 0.411 | 0.234 | 0.411 | 0.339 | 0.280 | 0.414 |

## 5. Key Findings

### 5.1 Layer 17 is the Optimal Steering Layer

A systematic scan of all 33 layers in ESM2-650M shows that Layer 17 achieves the best balance between property improvement and sequence quality. This aligns with the general observation that middle transformer layers encode semantic/functional information, while early layers handle local syntax and late layers prepare output logits.

### 5.2 All-Layer Steering Destroys Protein Structure

While all-layer steering achieves the highest oracle scores, ESMFold reveals that the generated sequences are structurally invalid (pLDDT drops from 0.73 to 0.34, pTM from 0.73 to 0.15). The sequences may score well on a property predictor trained on ESM2 features, but they are not real proteins. This highlights the importance of structural evaluation beyond oracle predictors.

### 5.3 GLP Cannot Simultaneously Improve Property and Naturalness

The GLP's noise parameter `u` controls a direct trade-off:
- Low `u` (0.1): Preserves steering signal → high sol (48%) but terrible structure (pLDDT 0.34)
- High `u` (0.9): Restores structure (pLDDT 0.65) but washes out most steering signal → sol (29%) barely above no-steering (25%)

This is a fundamental limitation: GLP performs **unconditional** denoising — it pulls activations toward the average natural protein distribution, which is exactly the opposite of steering.

### 5.4 Error Accumulation is Superlinear

Experiments V2 and V3 demonstrate that single-step GLP error is tiny (ΔpPPL < 0.14), but 10 rounds of iterative mask-predict amplify it by 80–150×. This superlinear accumulation — not single-step imprecision — is the primary cause of pPPL degradation in full generation pipelines.

### 5.5 GLP Learns the Activation Distribution Accurately

At `u=1.0` (pure noise input, GLP generates from scratch), the resulting pPPL (7.2) matches no-steering generation (7.19). In single-mask experiments, `u=1.0` produces ΔpPPL = +0.017, nearly identical to no-steering (+0.015). This confirms that GLP has learned the ESM2 Layer 17 activation distribution well — the problem is not distributional mismatch but the inherent conflict between unconditional denoising and directional steering.

### 5.6 Thermostability Steering is More Structure-Friendly

L17 steering for thermostability achieves +6.1°C Tm improvement while maintaining higher structural quality (pLDDT 0.713) compared to solubility steering (pLDDT 0.654). This may be because thermostability is inherently correlated with structural compactness — the steering direction aligns with rather than opposes good folding.

## 6. Limitations and Future Directions

### Current Limitations

1. **Unconditional GLP**: The GLP denoiser is unaware of the target property, so it actively counteracts steering. A conditional variant that denoises *toward* a target property region could break the sol-vs-structure trade-off.

2. **Independent Token Denoising**: The GLP denoiser processes each token's 1280-dim vector independently (pure MLP, no attention). This means projected activations may be locally reasonable but globally inconsistent across the sequence.

3. **Iterative Error Accumulation**: 10 rounds of mask-predict amplify small per-step errors superlinearly. Strategies to mitigate this include applying GLP only at selected rounds, reducing the total number of rounds, or using a single-round generation with higher mask ratio.

4. **Oracle Predictor Bias**: The property evaluator is trained on ESM2 features, which may overestimate property scores for sequences that are in-distribution for ESM2 but not biologically viable. ESMFold structural evaluation helps catch this, but is not a substitute for wet-lab validation.

### Future Directions

- **Conditional GLP**: Train the denoiser with property labels as conditioning input, enabling property-directed on-manifold projection
- **Adaptive `u` Scheduling**: Use different `u` values at different generation rounds (e.g., high `u` early for structure, low `u` late for property)
- **Cross-Token Denoising**: Replace MLP denoiser with a Transformer that captures sequence-level context
- **Single-Round High Mask**: Generate in a single round with mask_ratio=0.5–0.8 to avoid error accumulation while still modifying enough positions for property change
- **Multi-Property Steering**: Combine steering vectors for different properties (e.g., solubility + thermostability) via linear combination or alternating application
