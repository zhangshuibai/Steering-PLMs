#!/bin/bash
set -e

# ============================================================
# Single-Round Multi-Mask Experiment
# 7 methods × 10 mask_ratios × 2000 seqs
# Phase 1: Generate + Oracle (steering env, GPU 0)
# Phase 2: ESMFold on 200/group (esmfold env, GPU 0)
# ============================================================

eval "$(conda shell.bash hook)"
conda activate steering
cd /data/szhang967/Steering-PLMs

OUTDIR="new-results/single_round_mask_ratio"
LOG="$OUTDIR/experiment.log"
mkdir -p "$OUTDIR"

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "Single-Round Experiment started at $(date)"
echo "============================================================"

# ============================================================
# Phase 1: Generate + Oracle
# ============================================================

CUDA_VISIBLE_DEVICES=0 python -c "
import sys, os, math, json, time
sys.path.insert(0, '.')
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import types

torch.manual_seed(42)
np.random.seed(42)

from module.steerable_esm2 import steering_forward
from utils.esm2_utils import load_esm2_model, decode
from utils.gen_utils import sample_top_p
from evaluation.common import load_predictor, extract_features
from scripts.glp.steering_with_glp import (
    load_glp, build_glp_projection_fn, steering_forward_with_glp
)

device = 'cuda:0'
OUTDIR = '$OUTDIR'
N_GEN = 2000
MASK_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Load models
print('Loading ESM2-650M...')
model, alphabet = load_esm2_model('650M', device=device)
model.steering_forward = types.MethodType(steering_forward, model)
model.steering_forward_glp = types.MethodType(steering_forward_with_glp, model)
batch_converter = alphabet.get_batch_converter()
mask_idx = alphabet.mask_idx

# Load steering vectors
pos_sv, neg_sv = torch.load('saved_steering_vectors/650M_sol_steering_vectors.pt', weights_only=False)
sv_all = (pos_sv - neg_sv).to(device)
sv_l17 = torch.zeros_like(sv_all)
sv_l17[17] = sv_all[17]

# Load GLP
glp_model = load_glp('generative_latent_prior/runs/glp-esm2-650m-layer17-d6', device)
glp_fns = {}
for u in [0.1, 0.5, 0.9, 1.0]:
    glp_fns[u] = build_glp_projection_fn(glp_model, u=u, num_timesteps=25)

# Load predictor
predictor = load_predictor('evaluation/oracle/solubility/sol_predictor_final.pt', device=device)

# Reference sequences
ref_seqs = pd.read_csv('data/sol_easy.csv')['sequence'].tolist()

# Reference oracle
ref_feats = extract_features(ref_seqs, model, alphabet, device)
with torch.no_grad():
    ref_probs = torch.sigmoid(predictor(ref_feats.to(device)).cpu()).numpy()
ref_sol_prob = float(ref_probs.mean())
ref_sol_ratio = float((ref_probs >= 0.5).mean())
print(f'Reference: sol_prob={ref_sol_prob:.4f}, sol_ratio={ref_sol_ratio*100:.1f}%')

# Define methods
methods = {
    'no_steering':      {'sv': None,   'glp': None},
    'all_layer_no_glp': {'sv': sv_all, 'glp': None},
    'L17_no_glp':       {'sv': sv_l17, 'glp': None},
    'L17_glp_u0.1':     {'sv': sv_all, 'glp': glp_fns[0.1]},
    'L17_glp_u0.5':     {'sv': sv_all, 'glp': glp_fns[0.5]},
    'L17_glp_u0.9':     {'sv': sv_all, 'glp': glp_fns[0.9]},
    'L17_glp_u1.0':     {'sv': sv_all, 'glp': glp_fns[1.0]},
}

def generate_single_round(seq, sv, mask_ratio, glp_fn=None):
    _, _, tokens = batch_converter([('protein', seq)])
    tokens = tokens.to(device).clone()
    length = tokens.size(1) - 2
    mask_size = min(math.ceil(length * mask_ratio), length)
    if mask_size == 0:
        return decode(alphabet, tokens[:, 1:-1], onehot=False)[0]
    indices = torch.randperm(length)[:mask_size]
    mask_positions = indices + 1
    seq_token = tokens.clone()
    seq_token[0, mask_positions] = mask_idx

    with torch.no_grad():
        if glp_fn is not None:
            outputs = model.steering_forward_glp(
                tokens=seq_token, steering_vectors=sv,
                glp_project_fn=glp_fn, glp_layer=17)
        elif sv is not None:
            outputs = model.steering_forward(tokens=seq_token, steering_vectors=sv)
        else:
            outputs = model(tokens=seq_token)

    logits = outputs['logits'][0, :, 4:24]
    probs_dist = torch.softmax(logits / 1.0, dim=-1)
    pred = sample_top_p(probs_dist, 0.9) + 4
    pred[0] = tokens[0, 0]
    pred[-1] = tokens[0, -1]
    tokens[0, mask_positions] = pred[mask_positions]
    return decode(alphabet, tokens[:, 1:-1], onehot=False)[0]

all_results = []

for method_name, cfg in methods.items():
    sv = cfg['sv']
    glp_fn = cfg['glp']
    method_dir = os.path.join(OUTDIR, method_name)
    os.makedirs(method_dir, exist_ok=True)

    for mr in MASK_RATIOS:
        label = f'{method_name}/mr{mr:.1f}'
        print(f'\n>>> {label} (N={N_GEN})')

        # Reset seed per group for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        t0 = time.time()
        gen_seqs = []
        for i in tqdm(range(N_GEN), desc=label):
            seq = ref_seqs[i % len(ref_seqs)]
            gen_seq = generate_single_round(seq, sv, mr, glp_fn)
            gen_seqs.append(gen_seq)
        elapsed = time.time() - t0

        # Save sequences
        csv_path = os.path.join(method_dir, f'mr{mr:.1f}.csv')
        pd.DataFrame({'sequence': gen_seqs}).to_csv(csv_path, index=False)

        # Oracle evaluation
        feats = extract_features(gen_seqs, model, alphabet, device, batch_size=16)
        with torch.no_grad():
            scores = predictor(feats.to(device)).cpu()
        probs = torch.sigmoid(scores).numpy()
        sol_prob = float(probs.mean())
        sol_ratio = float((probs >= 0.5).mean())

        result = {
            'method': method_name,
            'mask_ratio': mr,
            'n_seqs': N_GEN,
            'sol_mean_prob': sol_prob,
            'sol_ratio': sol_ratio,
            'time_sec': elapsed,
        }
        all_results.append(result)
        print(f'  sol_prob={sol_prob:.4f}, sol_ratio={sol_ratio*100:.1f}%, time={elapsed:.0f}s')

# Save summary
summary = {
    'experiment': 'single_round_mask_ratio',
    'n_gen': N_GEN,
    'ref_sol_prob': ref_sol_prob,
    'ref_sol_ratio': ref_sol_ratio,
    'results': all_results,
}
with open(os.path.join(OUTDIR, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\n[DONE] Phase 1 complete. Summary saved.')
"
echo "[DONE] Phase 1 at $(date)"

# ============================================================
# Phase 2: ESMFold (200 per group)
# ============================================================

echo ""
echo ">>> Phase 2: ESMFold evaluation (200 per group)"
echo "Switching to esmfold env..."

eval "$(conda shell.bash hook)"
conda activate esmfold

METHODS="no_steering all_layer_no_glp L17_no_glp L17_glp_u0.1 L17_glp_u0.5 L17_glp_u0.9 L17_glp_u1.0"
MRS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

INPUT_CSVS=""
LABELS=""
for m in $METHODS; do
    for mr in $MRS; do
        csv="$OUTDIR/${m}/mr${mr}.csv"
        if [ -f "$csv" ]; then
            INPUT_CSVS="$INPUT_CSVS $csv"
            LABELS="$LABELS ${m}_mr${mr}"
        fi
    done
done

CUDA_VISIBLE_DEVICES=0 python evaluation/esmfold/evaluate_esmfold.py \
    --input_csvs $INPUT_CSVS \
    --labels $LABELS \
    --max_seqs 200 \
    --output_csv "$OUTDIR/esmfold_results.csv" \
    --device cuda:0

echo "[DONE] Phase 2 ESMFold at $(date)"

echo ""
echo "============================================================"
echo "All single-round experiments completed at $(date)"
echo "Results in $OUTDIR/"
echo "============================================================"
