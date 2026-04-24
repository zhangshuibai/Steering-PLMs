---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Helvetica', 'Arial', sans-serif;
    font-size: 26px;
  }
  h1 { color: #1a365d; }
  h2 { color: #2c5282; }
  code { background: #f7fafc; padding: 2px 6px; border-radius: 3px; }
  table { margin: 0 auto; border-collapse: collapse; }
  th, td { padding: 6px 14px; border: 1px solid #cbd5e0; }
  th { background: #edf2f7; }
  .small { font-size: 20px; color: #4a5568; }
  .big { font-size: 38px; color: #1a365d; font-weight: bold; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Prior-Aware Masked Diffusion for Protein Language Model Steering

### A cheap density filter for post-hoc quality control

<br>

Shuibai Zhang — UW–Madison
<span class="small">Course presentation · 10 min</span>

---

## Problem: steer a PLM toward a property

- **Goal**: generate proteins that are soluble / thermostable
- **Tool**: steer a frozen ESM-2-650M via activation-space vectors
  - Add a learned direction $v_\ell$ at each layer $\ell$, then rescale norm
  - No fine-tuning, works in ~10 min per 100 sequences
- **Trade-off**: stronger steering $\Rightarrow$ better property score, but sequences drift off the natural-protein manifold (low pLDDT, non-foldable)

> **Research question**: *Can we keep the property gain but reject the unnatural outputs — cheaply?*

---

## The failure mode: over-steering ≈ random amino acids

<div class="big" style="text-align:center; margin-top:20px">Over-steered sequences look like random AA in embedding space</div>

|  | Mahalanobis² (L17) | pLDDT | Oracle (solubility) |
|---|---:|---:|---:|
| Random AA (baseline) | **608** | **0.26** | — |
| L17 α=10 (over-steer) | **619** | **0.35** | 0.71 |
| allL α=2 (balanced) | ~700 | 0.48 | 0.66 |
| Natural UniRef50 | ~D=1280 | 0.75+ | — |

<span class="small">Random AA and α=10 sit on the same Mahalanobis "shell" — the model can't tell them apart from its own hidden-state prior.</span>

---

## Our insight

1. Steering pushes hidden states **off** the natural-protein manifold.
2. Off-manifold sequences are detectable by a **density score** in PLM embedding space.
3. Under a Gaussian prior on ESM-2 L17 activations:
   $$\text{Mahal}^2(h) = \sum_{i=1}^{D}\!\left(\frac{h_i - \mu_i}{\sigma_i}\right)^{\!2} \sim \chi^2(D),\ D=1280$$
4. χ²(D) is tightly concentrated around $D$ with std $\sqrt{2D}\approx 50.6$
   → reject anything with $\text{Mahal}^2 < D - k\sqrt{2D}$ (k=1 default)

---

## Method: a 10 KB filter

<br>

```
 Generate N = 500 candidates ──►  ESM-2 forward @ layer 17
                                        │
                                  h ∈ ℝ^(L × 1280)
                                        │
                          Mahal²  = Σᵢ ((hᵢ − μᵢ)/σᵢ)²        ← 34 ms / seq
                                        │
                          Mahal² ≥ D − k√(2D)?
                                        │
                   ┌────────────────┴────────────────┐
                 accept                             reject
```

- **Prior**: `(mean, var)` of ESM-2 L17 activations on UniRef50 → **10 KB file**
- Reuses the single ESM-2 forward you already did for generation

---

## Main result: filter gain across 4 splits

| Task (best setting) | n | Δ pLDDT | Δ Oracle |
|---|---:|---:|---:|
| sol_easy · allL α=2   | 500 | **+0.063** | **+0.119** |
| sol_hard · allL α=2   | 500 | **+0.068** | +0.092 |
| therm_easy · L17 α=1  | 500 | **+0.063** | +0.087 |
| therm_hard · L17 α=1  | 500 | **+0.067** | +0.075 |

**Reproducibility** (sol_easy allL α=2 across seeds ∈ {0, 1, 42, 100, 123}):

<div class="big" style="text-align:center">Δ pLDDT = 0.061 ± 0.009 &nbsp;&nbsp; Δ Oracle = 0.102 ± 0.015</div>

---

## Cost: nearly free

| Proxy | ms / seq | Peak GPU | Storage | Slowdown |
|---|---:|---:|---:|---:|
| **Mahalanobis² (ours)** | **34** | **2.7 GB** | **10 KB** | **1×** |
| GLP flow residual | 283 | 4.1 GB | 1.3 GB | 8× |
| Pseudo-perplexity 650M | 472 | 2.7 GB | 2.5 GB | 14× |
| Pseudo-perplexity 3B | 1 513 | 12.9 GB | 11 GB | 44× |

<br>

<span class="small">Generating + filtering 10 k sequences: <b>5.7 min</b> with Mahal vs <b>4.2 h</b> with ppl-3B.</span>

---

## Scope: where the filter works (and doesn't)

✅ **Works** — structural-quality tasks
- Solubility, thermostability
- pLDDT improves consistently across strengths k ∈ [0.3, 3]
- Short proteins benefit most (+0.12 for 60–150 aa)

❌ **Fails** — direct fitness tasks
- TrpB (enzyme activity, WT-locked 4-site mutations): Δ ≈ 0
- GFP (fluorescence, narrow landscape): Δ negative
- **Why**: Mahalanobis is a *naturalness* proxy, not a *fitness* proxy

> Honest limitation: this is a structural filter, not a universal oracle.

---

## Why Mahalanobis² is a reasonable density

- Under a fitted Gaussian prior, $-\log p(h) = \tfrac{1}{2}\,\text{Mahal}^2(h) + \text{const}$
- Natural ESM-2 L17 activations are **approximately** Gaussian
  → literature supports a single-layer Gaussian prior on L17 hidden states
- **Ongoing**: comparing Mahal² to exact GLP flow NLL (Hutchinson trace estimator) on 4 400 sequences
  - Target Spearman $r \geq 0.85$ would certify Mahal² as a valid density approximation
  - If lower: we reframe as a norm-based structural score (filter still works either way)

---

## Takeaways

1. **Over-steering collapses PLMs toward random-AA territory in hidden space.**
2. A **10 KB χ² filter** recovers +0.06 pLDDT (5-seed reproducible) at 34 ms/sequence — **44× cheaper** than ppl-3B.
3. **Scope-limited**: structural quality tasks only; fitness still needs an oracle.
4. **Drop-in** add-on to any PLM steering pipeline — one forward pass, one line of math.

<br>

<div class="big" style="text-align:center">Cheap prior-aware rejection sampling &rarr; better naturalness, no retraining.</div>

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thanks — Questions?

<br>

**Code + data**: [github.com/zhangshuibai/Steering-PLMs](https://github.com/zhangshuibai/Steering-PLMs)
**Contact**: szhang967@wisc.edu

<br>

<span class="small">Backup slides: k-sweep · length-bin ablation · multi-layer Mahalanobis · full cost table</span>

---

<!-- Backup slide: k-sweep -->

## Backup — threshold robustness (k-sweep on sol)

<span class="small">Filter is stable across a wide range of $k$. We use k=1 (matches χ²(D) 1σ lower tail) as default.</span>

| k | Threshold | Acceptance rate | Δ pLDDT (avg) |
|---:|---:|---:|---:|
| 0.3 | D − 0.3√(2D) | ~40% | +0.045 |
| 0.5 | D − 0.5√(2D) | ~30% | +0.054 |
| **1.0** | D − √(2D) | ~12% | **+0.063** |
| 1.5 | D − 1.5√(2D) | ~6% | +0.067 |
| 2.0 | D − 2√(2D) | ~3% | +0.069 |

<span class="small">Smaller k ⇒ more accepted, weaker filter. Larger k ⇒ fewer accepted, stronger filter. k=1 balances yield and quality.</span>

---

<!-- Backup slide: length bins -->

## Backup — filter gain by protein length

Sol_easy, L17 α=1 setting:

| Length bin (aa) | n accepted | Base pLDDT | Filtered pLDDT | Δ |
|---|---:|---:|---:|---:|
| 71–148 | 52 | 0.54 | **0.65** | **+0.11** |
| 148–186 | 68 | 0.66 | 0.70 | +0.05 |
| 186–216 | 85 | 0.76 | 0.78 | +0.02 |
| 216–256 | 62 | 0.69 | 0.75 | +0.05 |

Shorter proteins gain more — they have more "room" to go wrong in generation,
and the filter catches their drift more effectively.

---

<!-- Speaker notes — not displayed -->

<!--
### Speaker notes & pacing (target 10 min)

1. Title (10 s): Introduce project and goal.
2. Problem (1 min): set up steering, no fine-tuning, trade-off.
3. Failure mode (1 min): highlight Mahal=608 for random ≈ 619 for α=10 — punchline.
4. Insight (45 s): χ² theory + threshold intuition.
5. Method (1 min 15): walk through the pipeline diagram + 10 KB cost.
6. Main result (1 min 30): Δ pLDDT across 4 splits, multi-seed robustness.
7. Cost (1 min): 44× slowdown for ppl-3B vs 34 ms for Mahal.
8. Scope (1 min 15): honest about fitness task failure.
9. Why it works (45 s): density interpretation, ongoing NLL validation.
10. Takeaways (30 s): 4 bullets.
11. Thanks / Q&A (15 s).

Total: ~10 min. Backups for k-sweep and length bins if questions.
-->
