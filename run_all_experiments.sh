#!/bin/bash
set -e

# ============================================================
# Re-run all non-GLP experiments with new code structure
# Output: new-results/
# GPU: 0,1 (A100 40GB)
# Estimated: ~8 hours total
# ============================================================

eval "$(conda shell.bash hook)"
conda activate steering
cd /data/szhang967/Steering-PLMs

LOG="new-results/experiment.log"
mkdir -p new-results/baseline new-results/single_layer_steering new-results/ppl

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "Experiment started at $(date)"
echo "============================================================"

# ============================================================
# Phase 1: Generation + Oracle eval (~25 min, GPU 0)
# ============================================================

echo ""
echo ">>> Step 1: Extract steering vectors"
CUDA_VISIBLE_DEVICES=0 python scripts/extraction/extract_esm2_steering_vec.py \
    --model "650M" --num_data 100 --property "sol" \
    --data_path "data/sol_filtered.csv" \
    --theshold_pos 0.5 --theshold_neg 0.2
echo "[DONE] Step 1 at $(date)"

echo ""
echo ">>> Step 2: All-layer steering generation (sol_easy, 100 seqs)"
CUDA_VISIBLE_DEVICES=0 python scripts/generation/steering_esm2_generation.py \
    --model "650M" --property "sol" \
    --ref_data_path "data/sol_easy.csv" \
    --output_file "new-results/baseline/steering_sol_easy.csv" \
    --steering --n 100
echo "[DONE] Step 2 at $(date)"

echo ""
echo ">>> Step 3: No-steering baseline (sol_easy, 100 seqs)"
CUDA_VISIBLE_DEVICES=0 python scripts/generation/steering_esm2_generation.py \
    --model "650M" --property "sol" \
    --ref_data_path "data/sol_easy.csv" \
    --output_file "new-results/baseline/no_steering_sol_easy.csv" \
    --n 100
echo "[DONE] Step 3 at $(date)"

echo ""
echo ">>> Step 4: All-layer steering generation (sol_hard, 100 seqs)"
CUDA_VISIBLE_DEVICES=0 python scripts/generation/steering_esm2_generation.py \
    --model "650M" --property "sol" \
    --ref_data_path "data/sol_hard.csv" \
    --output_file "new-results/baseline/steering_sol_hard.csv" \
    --steering --n 100
echo "[DONE] Step 4 at $(date)"

echo ""
echo ">>> Step 5: No-steering baseline (sol_hard, 100 seqs)"
CUDA_VISIBLE_DEVICES=0 python scripts/generation/steering_esm2_generation.py \
    --model "650M" --property "sol" \
    --ref_data_path "data/sol_hard.csv" \
    --output_file "new-results/baseline/no_steering_sol_hard.csv" \
    --n 100
echo "[DONE] Step 5 at $(date)"

echo ""
echo ">>> Step 6: Oracle evaluation (4 groups)"
for group in steering_sol_easy no_steering_sol_easy steering_sol_hard no_steering_sol_hard; do
    if [[ "$group" == *easy* ]]; then
        REF="data/sol_easy.csv"
    else
        REF="data/sol_hard.csv"
    fi
    CUDA_VISIBLE_DEVICES=0 python evaluation/oracle/evaluate_oracle.py \
        --input_csv "new-results/baseline/${group}.csv" \
        --property sol --ref_csv "$REF" --device cuda:0
    echo "[DONE] Oracle: $group at $(date)"
done

echo ""
echo ">>> Step 7-8: Single-layer steering scan (33 layers × 100 seqs) + oracle eval"
CUDA_VISIBLE_DEVICES=0 python -c "
import sys, os
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import types
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

from module.steerable_esm2 import steering_forward
from utils.esm2_utils import load_esm2_model, generate_sequences
from evaluation.common import PropertyPredictor, load_predictor, extract_features

# Load models
model, alphabet = load_esm2_model('650M')
device = 'cuda:0'
model = model.to(device)
model.eval()
model.steering_forward = types.MethodType(steering_forward, model)
batch_converter = alphabet.get_batch_converter()

# Load steering vectors
pos_sv, neg_sv = torch.load('saved_steering_vectors/650M_sol_steering_vectors.pt', weights_only=False)
sv_all = (pos_sv - neg_sv).to(device)

# Load predictor
predictor = load_predictor('evaluation/oracle/solubility/sol_predictor_final.pt', device=device)

# Load reference sequences
ref_df = pd.read_csv('data/sol_easy.csv')
ref_seqs = ref_df['sequence'].tolist()

# Evaluate reference
ref_features = extract_features(ref_seqs, model, alphabet, device)
with torch.no_grad():
    ref_scores = predictor(ref_features.to(device)).cpu()
ref_probs = torch.sigmoid(ref_scores).numpy()
ref_mean_prob = ref_probs.mean()
ref_sol_ratio = float((ref_probs >= 0.5).mean())

output_dir = 'new-results/single_layer_steering'
n_gen = 100
results = []

for layer in range(33):
    print(f'\n=== Layer {layer} ===')
    # Create single-layer steering vector
    sv_single = torch.zeros_like(sv_all)
    sv_single[layer] = sv_all[layer]

    # Generate sequences one by one (generate_sequences takes single token tensor)
    gen_seqs = []
    for i in tqdm(range(n_gen), desc=f'Layer {layer}'):
        seq = ref_seqs[i % len(ref_seqs)]
        _, _, seq_token = batch_converter([('protein', seq)])
        seq_token = seq_token.to(device)
        new_seq = generate_sequences(seq_token, model, sv_single, 0.1, alphabet, temperature=1.0, top_p=0.9)
        gen_seqs.append(new_seq)

    # Save sequences
    csv_path = os.path.join(output_dir, f'layer_{layer}.csv')
    pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

    # Oracle eval
    features = extract_features(gen_seqs, model, alphabet, device)
    with torch.no_grad():
        scores = predictor(features.to(device)).cpu()
    probs = torch.sigmoid(scores).numpy()
    mean_prob = float(probs.mean())
    sol_ratio = float((probs >= 0.5).mean())

    # Save scored
    scored_df = pd.DataFrame({
        'sequence': gen_seqs,
        'pred_prob': probs.tolist(),
        'pred_label': (probs >= 0.5).astype(int).tolist(),
    })
    scored_df.to_csv(os.path.join(output_dir, f'layer_{layer}_scored.csv'), index=False)

    results.append({
        'layer': layer,
        'sol_mean_prob': mean_prob,
        'sol_ratio': sol_ratio,
        'n_seqs': len(gen_seqs),
    })
    print(f'  sol_mean_prob={mean_prob:.4f}, sol_ratio={sol_ratio:.2f}')

# Save summary
summary = {
    'experiment': 'single_layer_steering',
    'model': 'ESM2-650M',
    'n_gen': n_gen,
    'ref_data': 'data/sol_easy.csv',
    'ref_mean_prob': float(ref_mean_prob),
    'ref_sol_ratio': float(ref_sol_ratio),
    'results': results,
}
with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('\n[DONE] Single-layer steering scan + oracle eval')
"
echo "[DONE] Steps 7-8 at $(date)"

# ============================================================
# Phase 2: pPPL evaluation (~7 hr, GPU 0+1)
# ============================================================

echo ""
echo ">>> Step 9: pPPL for baseline groups (4 × 100 seqs)"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/ppl/evaluate_ppl.py \
    --input_csvs \
        new-results/baseline/steering_sol_easy.csv \
        new-results/baseline/no_steering_sol_easy.csv \
        new-results/baseline/steering_sol_hard.csv \
        new-results/baseline/no_steering_sol_hard.csv \
        data/sol_easy.csv \
        data/sol_hard.csv \
    --labels \
        "steering_easy" "no_steering_easy" \
        "steering_hard" "no_steering_hard" \
        "ref_easy" "ref_hard" \
    --model 3B --gpu_ids 0 1 \
    --output_csv new-results/ppl/baseline_ppl.csv
echo "[DONE] Step 9 at $(date)"

echo ""
echo ">>> Step 10: pPPL for single-layer steering (33 layers × 100 seqs)"
# Build input args
INPUT_CSVS=""
LABELS=""
for i in $(seq 0 32); do
    INPUT_CSVS="$INPUT_CSVS new-results/single_layer_steering/layer_${i}.csv"
    LABELS="$LABELS layer_${i}"
done

CUDA_VISIBLE_DEVICES=0,1 python evaluation/ppl/evaluate_ppl.py \
    --input_csvs $INPUT_CSVS \
    --labels $LABELS \
    --model 3B --gpu_ids 0 1 \
    --output_csv new-results/ppl/single_layer_ppl.csv
echo "[DONE] Step 10 at $(date)"

echo ""
echo "============================================================"
echo "All experiments completed at $(date)"
echo "Results in new-results/"
echo "============================================================"
