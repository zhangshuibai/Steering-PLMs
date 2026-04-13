#!/bin/bash
set -e

# ============================================================
# Resume from Step 7 (Step 1-6 already completed)
# ============================================================

eval "$(conda shell.bash hook)"
conda activate steering
cd /data/szhang967/Steering-PLMs

LOG="new-results/experiment.log"
exec > >(tee -a "$LOG") 2>&1
echo ""
echo "============================================================"
echo "Resuming from Step 7 at $(date)"
echo "============================================================"

# ============================================================
# Step 7-8: Single-layer steering scan (33 layers × 100 seqs) + oracle eval
# ============================================================

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
ref_mean_prob = float(ref_probs.mean())
ref_sol_ratio = float((ref_probs >= 0.5).mean())

output_dir = 'new-results/single_layer_steering'
n_gen = 100
results = []

for layer in range(33):
    print(f'\n=== Layer {layer} ===')
    sv_single = torch.zeros_like(sv_all)
    sv_single[layer] = sv_all[layer]

    # Generate one by one
    gen_seqs = []
    for i in tqdm(range(n_gen), desc=f'Layer {layer}'):
        seq = ref_seqs[i % len(ref_seqs)]
        _, _, seq_token = batch_converter([('protein', seq)])
        seq_token = seq_token.to(device)
        new_seq = generate_sequences(seq_token, model, sv_single, 0.1, alphabet, temperature=1.0, top_p=0.9)
        gen_seqs.append(new_seq)

    csv_path = os.path.join(output_dir, f'layer_{layer}.csv')
    pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

    # Oracle eval
    features = extract_features(gen_seqs, model, alphabet, device)
    with torch.no_grad():
        scores = predictor(features.to(device)).cpu()
    probs = torch.sigmoid(scores).numpy()
    mean_prob = float(probs.mean())
    sol_ratio = float((probs >= 0.5).mean())

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

summary = {
    'experiment': 'single_layer_steering',
    'model': 'ESM2-650M',
    'n_gen': n_gen,
    'ref_data': 'data/sol_easy.csv',
    'ref_mean_prob': ref_mean_prob,
    'ref_sol_ratio': ref_sol_ratio,
    'results': results,
}
with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print('\n[DONE] Single-layer steering scan + oracle eval')
"
echo "[DONE] Steps 7-8 at $(date)"

# ============================================================
# Step 9: pPPL for baseline groups
# ============================================================

echo ""
echo ">>> Step 9: pPPL for baseline groups (4 generated + 2 reference)"
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

# ============================================================
# Step 10: pPPL for single-layer steering (33 layers)
# ============================================================

echo ""
echo ">>> Step 10: pPPL for single-layer steering (33 layers × 100 seqs)"
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
