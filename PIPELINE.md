# Solubility Steering Pipeline (ESM2-650M) - Detailed Walkthrough

Paper: "Steering Protein Language Models" (ICML'25)

## Overview

The pipeline has 4 steps:

1. **Extract Steering Vectors** - Compute directional vectors from high/low solubility sequences
2. **Steered Generation** - Generate sequences guided by steering vectors
3. **Baseline Generation** - Generate sequences without steering (control group)
4. **Oracle Evaluation** - Score all generated sequences with a trained solubility predictor

One-command execution:
```bash
bash run_sol_steering_pipeline.sh
```

---

## Step 1: Extract Steering Vectors

**Command:**
```bash
python extract_esm2_steering_vec.py \
    --model "650M" --num_data 100 --property "sol" \
    --data_path "data/sol_filtered.csv" \
    --theshold_pos 0.5 --theshold_neg 0.2
```

**Input:** `data/sol_filtered.csv`
- 720 protein sequences with `sequence` and `score` (0~1 continuous solubility probability)

**Process (`extract_esm2_steering_vec.py`):**

1. Split sequences by thresholds:
   - `pos_seqs`: score >= 0.5 (high solubility), take first 100
   - `neg_seqs`: score <= 0.2 (low solubility), take first 100

2. Load ESM2-650M model (`utils/esm2_utils.py:load_esm2_model`)

3. Extract all 33 layers' mean-pooled representations (`utils/esm2_utils.py:extract_esm2_features`):
   - Each sequence -> ESM2 forward pass -> per-layer token representations (excluding BOS/EOS) -> mean pooling
   - Result: `pos_seq_repr_mat` shape = `(33, 100, 1280)` (33 layers x 100 seqs x 1280 dim)
   - Same for `neg_seq_repr_mat`

4. Compute per-layer mean across sequences:
   - `pos_steering_vectors[i]` = mean of pos group at layer i, shape `(1280,)`
   - `neg_steering_vectors[i]` = mean of neg group at layer i, shape `(1280,)`

**Output:** `saved_steering_vectors/650M_sol_steering_vectors.pt` (332KB)
- Tuple of `(pos_steering_vectors, neg_steering_vectors)`, each `torch.Size([33, 1280])`
- The actual steering direction `pos - neg` is computed at generation time

---

## Step 2: Steered Generation (Easy + Hard)

**Command (Easy example):**
```bash
python steering_esm2_generation.py \
    --model "650M" --property "sol" --device "cuda:7" \
    --ref_data_path "data/sol_easy.csv" \
    --output_file "results/ESM2_gen_steering_sol_easy.csv" \
    --steering --n 100
```

**Inputs:**
- `data/sol_easy.csv`: 162 reference sequences, score 0.251~0.300 (medium-low solubility)
- `data/sol_hard.csv`: 198 reference sequences, score 0.001~0.100 (very low solubility)
- `saved_steering_vectors/650M_sol_steering_vectors.pt` from Step 1

**Process (`steering_esm2_generation.py`):**

1. Load reference sequences and ESM2-650M model
2. Inject `steering_forward` method into model (`module/steerable_esm2.py`)
3. Compute steering vector: `steering_vectors = (pos_sv - neg_sv) * alpha` (alpha=1.0)

4. Generate 100 sequences (cycling through references via `ref_seqs[i % len(ref_seqs)]`):

   For each reference, call `generate_sequences()` (`utils/esm2_utils.py:156-177`):

   a. **10 rounds of iterative masked prediction** (`rounds = ceil(1.0 / mask_ratio) = 10`):
      - Each round: randomly select 10% of positions -> set to `<mask>` token
      - Previously selected positions are excluded from future rounds

   b. **Forward pass with steering** via `pred_tokens()` (`utils/esm2_utils.py:127-154`):
      - Calls `model.steering_forward()` (`module/steerable_esm2.py:3-69`)
      - **Core steering operation** at each transformer layer:
        ```python
        new_x = x + steering_vectors[layer_idx]      # add steering direction
        x = new_x * (||x|| / ||new_x||)              # normalize back to original magnitude
        ```
        This changes the direction of activations without altering their L2 norm.
      - Output logits are restricted to 20 amino acid tokens (indices 4:24)

   c. **Sample new tokens**:
      - Softmax with temperature=1.0
      - Top-p (nucleus) sampling with p=0.9 (`utils/gen_utils.py:sample_top_p`)
      - Replace masked positions with sampled tokens

   d. After 10 rounds, every position has been re-predicted once -> complete new sequence

5. Decode to amino acid strings

**Outputs:**
- `results/ESM2_gen_steering_sol_easy.csv`: 100 sequences, length 71~256 (mean 181)
- `results/ESM2_gen_steering_sol_hard.csv`: 100 sequences, length 59~250 (mean 160)
- Each CSV has a single `sequence` column

---

## Step 3: Baseline Generation (No Steering)

**Command (Easy example):**
```bash
python steering_esm2_generation.py \
    --model "650M" --property "sol" --device "cuda:7" \
    --ref_data_path "data/sol_easy.csv" \
    --output_file "results/ESM2_gen_no_steering_sol_easy.csv" \
    --n 100
    # Note: no --steering flag
```

**Process:** Identical to Step 2, except `steering_vectors=None`. In `pred_tokens()`:
```python
if steering_vectors is not None:
    outputs = model.steering_forward(tokens=tokens, steering_vectors=steering_vectors)
else:
    outputs = model(tokens=tokens)   # <- this path, no directional intervention
```

**Outputs:**
- `results/ESM2_gen_no_steering_sol_easy.csv`: 100 sequences
- `results/ESM2_gen_no_steering_sol_hard.csv`: 100 sequences

Note: Sequence lengths match the steering group (same reference sequences), but the actual amino acids differ due to the absence of steering + sampling randomness.

---

## Step 4: Oracle Evaluation

**Command (Easy+Steering example):**
```bash
python evaluate_generated_seqs.py \
    --input_csv "results/ESM2_gen_steering_sol_easy.csv" \
    --predictor_path "saved_predictors/sol_predictor_final.pt" \
    --property "sol" \
    --ref_csv "data/sol_easy.csv" \
    --device "cuda:7"
```

**Inputs:**
- `results/ESM2_gen_steering_sol_easy.csv`: generated sequences from Step 2
- `saved_predictors/sol_predictor_final.pt`: trained solubility oracle (6.3MB)
- `data/sol_easy.csv`: original reference sequences (for comparison)

**Process (`evaluate_generated_seqs.py`):**

1. Load ESM2-650M, extract last-layer mean-pooled features for generated sequences
   - Same extraction as predictor training: layer 33 representations -> exclude BOS/EOS -> mean pool -> `(1280,)` per sequence
   - 100 sequences -> `gen_features` shape = `(100, 1280)`

2. Load predictor (architecture: `Linear(1280,1280) -> GELU -> LayerNorm -> Linear(1280,1)`)

3. Predict and post-process:
   ```python
   pred_score = predictor(features)           # raw logit
   pred_prob  = sigmoid(pred_score)            # probability 0~1
   pred_label = 1 if pred_prob >= 0.5 else 0   # binary classification
   ```

4. If `ref_csv` provided: also extract features and score reference sequences, compute delta

**Outputs:** `results/ESM2_gen_steering_sol_easy_scored.csv`
- Columns: `sequence, pred_score, pred_prob, pred_label`

All 4 scored outputs:

| Output File | Source |
|-------------|--------|
| `results/ESM2_gen_steering_sol_easy_scored.csv` | Easy ref + Steering |
| `results/ESM2_gen_no_steering_sol_easy_scored.csv` | Easy ref + No Steering |
| `results/ESM2_gen_steering_sol_hard_scored.csv` | Hard ref + Steering |
| `results/ESM2_gen_no_steering_sol_hard_scored.csv` | Hard ref + No Steering |

---

## Data Flow Diagram

```
data/sol_filtered.csv (720 seqs, score 0~1)
    |
    |  Step 1: extract_esm2_steering_vec.py
    |  pos(score>=0.5) take 100, neg(score<=0.2) take 100
    |  ESM2-650M extract 33-layer x 1280-dim mean representations
    |  pos_mean - neg_mean = steering direction per layer
    v
saved_steering_vectors/650M_sol_steering_vectors.pt  (33x1280 x2)
    |
    |  Step 2 & 3: steering_esm2_generation.py
    |  For each ref seq: 10 rounds of mask-predict
    |  Each layer: x += steering_vec, then normalize to original norm
    |------------------------------+
    |                              |
    |  +steering                   |  no steering (control)
    v                              v
data/sol_easy.csv (162 seqs)    data/sol_easy.csv
  -> results/ESM2_gen_steering_   -> results/ESM2_gen_no_steering_
     sol_easy.csv (100 seqs)         sol_easy.csv (100 seqs)

data/sol_hard.csv (198 seqs)    data/sol_hard.csv
  -> results/ESM2_gen_steering_   -> results/ESM2_gen_no_steering_
     sol_hard.csv (100 seqs)         sol_hard.csv (100 seqs)
    |                              |
    |  Step 4: evaluate_generated_seqs.py
    |  ESM2 extract features -> sol_predictor_final.pt score
    |  sigmoid -> probability, compare delta_prob
    v                              v
results/*_scored.csv            results/*_scored.csv
(sequence, pred_score,          (sequence, pred_score,
 pred_prob, pred_label)          pred_prob, pred_label)
```

---

## Results Summary

| Group | Mean Solubility Prob | Soluble Ratio | Delta prob (vs ref) |
|-------|---------------------|---------------|---------------------|
| **Easy + Steering** | 0.3586 | **32.0%** | **+0.1565** |
| Easy + No Steering | 0.2484 | 20.0% | +0.0463 |
| Easy Reference | 0.2022 | 17.9% | -- |
| | | | |
| **Hard + Steering** | 0.3536 | **32.0%** | **+0.2709** |
| Hard + No Steering | 0.1622 | 11.0% | +0.0795 |
| Hard Reference | 0.0827 | 5.6% | -- |

Key findings:
- Steering consistently improves solubility over both the reference and the no-steering baseline
- The effect is stronger on the Hard set (delta +0.27 vs +0.08), demonstrating that steering can meaningfully shift low-solubility sequences toward higher solubility
- Soluble ratio increases from 5.6% -> 32.0% (Hard) and 17.9% -> 32.0% (Easy)

---

## Oracle Predictor Training

The solubility oracle predictor (`saved_predictors/sol_predictor_final.pt`) was trained separately:

- **Data:** DeepSol dataset (62,478 train / 6,942 val / 2,001 test sequences)
- **Features:** ESM2-650M last-layer mean-pooled representations (frozen, dim=1280)
- **Architecture:** Same as ESM2's `lm_head` (Linear -> GELU -> LayerNorm -> Linear)
- **Loss:** BCEWithLogitsLoss
- **Val performance (repo thresholds pos>=0.5, neg<=0.2):** Acc=0.739, F1=0.714
- **Paper targets:** Acc>=0.708, F1>=0.677

Training script: `train_sol_predictor.py`

---

## File Structure

```
Steering-PLMs/
  data/
    sol_filtered.csv          # 720 seqs for steering vector extraction
    sol_easy.csv              # 162 seqs, score 0.25~0.30 (generation reference)
    sol_hard.csv              # 198 seqs, score 0.001~0.10 (generation reference)
    deepsol/data/             # DeepSol dataset for predictor training
  saved_steering_vectors/
    650M_sol_steering_vectors.pt   # Step 1 output (33x1280 x2)
  saved_predictors/
    sol_predictor_final.pt         # Trained solubility oracle
    therm_predictor_nocdhit.pt     # Trained thermostability oracle
  results/
    ESM2_gen_steering_sol_easy.csv         # Step 2 output
    ESM2_gen_steering_sol_hard.csv         # Step 2 output
    ESM2_gen_no_steering_sol_easy.csv      # Step 3 output
    ESM2_gen_no_steering_sol_hard.csv      # Step 3 output
    ESM2_gen_steering_sol_easy_scored.csv  # Step 4 output
    ESM2_gen_steering_sol_hard_scored.csv  # Step 4 output
    ESM2_gen_no_steering_sol_easy_scored.csv  # Step 4 output
    ESM2_gen_no_steering_sol_hard_scored.csv  # Step 4 output
  run_sol_steering_pipeline.sh    # One-command full pipeline
  evaluate_generated_seqs.py      # Oracle evaluation script
  train_sol_predictor.py          # Solubility predictor training
  train_therm_predictor.py        # Thermostability predictor training
```
