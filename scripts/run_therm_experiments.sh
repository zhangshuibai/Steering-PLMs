#!/bin/bash
set -e

# ============================================================
# Thermostability Steering Experiments (ESM2-650M)
# Mirrors solubility experiments: extract vectors → generate → eval
#
# Thresholds: pos >= 70.0°C, neg <= 50.0°C
# Data: therm_filtered.csv (2000 seqs), therm_easy (342), therm_hard (248)
# ============================================================

eval "$(conda shell.bash hook)"
conda activate steering
cd /data/szhang967/Steering-PLMs

LOG="new-results/therm/experiment.log"
mkdir -p new-results/therm/baseline new-results/therm/single_layer new-results/therm/glp new-results/therm/ppl new-results/therm/esmfold

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "Thermostability Experiments started at $(date)"
echo "============================================================"

# ============================================================
# Step 1: Extract therm steering vectors
# ============================================================
echo ""
echo ">>> Step 1: Extract therm steering vectors"
CUDA_VISIBLE_DEVICES=0 python scripts/extraction/extract_esm2_steering_vec.py \
    --model "650M" --num_data 100 --property "therm" \
    --data_path "data/therm_filtered.csv" \
    --theshold_pos 70.0 --theshold_neg 50.0
echo "[DONE] Step 1 at $(date)"

# ============================================================
# Step 2-5: Generate sequences (easy + hard × steering + no-steering)
# ============================================================
echo ""
echo ">>> Step 2-5: Generate sequences"
for diff in easy hard; do
    for steer_flag in "--steering" ""; do
        if [ -n "$steer_flag" ]; then
            label="steering_therm_${diff}"
        else
            label="no_steering_therm_${diff}"
        fi
        echo "--- $label ---"
        CUDA_VISIBLE_DEVICES=0 python scripts/generation/steering_esm2_generation.py \
            --model "650M" --property "therm" --seed 42 \
            --ref_data_path "data/therm_${diff}.csv" \
            --output_file "new-results/therm/baseline/${label}.csv" \
            $steer_flag --n 100
        echo "[DONE] $label at $(date)"
    done
done

# ============================================================
# Step 6: Oracle evaluation (therm predictor)
# ============================================================
echo ""
echo ">>> Step 6: Oracle evaluation"
for diff in easy hard; do
    for prefix in steering no_steering; do
        group="${prefix}_therm_${diff}"
        echo "--- Oracle: $group ---"
        CUDA_VISIBLE_DEVICES=0 python evaluation/oracle/evaluate_oracle.py \
            --input_csv "new-results/therm/baseline/${group}.csv" \
            --property therm \
            --predictor_path "evaluation/oracle/thermostability/therm_predictor_nocdhit.pt" \
            --ref_csv "data/therm_${diff}.csv" \
            --device cuda:0
        echo "[DONE] Oracle: $group at $(date)"
    done
done

# ============================================================
# Step 7: L17 single-layer steering
# ============================================================
echo ""
echo ">>> Step 7: L17 single-layer steering (therm_easy)"
CUDA_VISIBLE_DEVICES=0 python -c "
import sys, os
sys.path.insert(0, '.')
import torch, numpy as np, pandas as pd, json, types
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

from module.steerable_esm2 import steering_forward
from utils.esm2_utils import load_esm2_model, generate_sequences
from evaluation.common import load_predictor, extract_features

model, alphabet = load_esm2_model('650M')
device = 'cuda:0'
model = model.to(device).eval()
model.steering_forward = types.MethodType(steering_forward, model)
batch_converter = alphabet.get_batch_converter()

pos_sv, neg_sv = torch.load('saved_steering_vectors/650M_therm_steering_vectors.pt', weights_only=False)
sv_all = (pos_sv - neg_sv).to(device)

# L17 only
sv_single = torch.zeros_like(sv_all)
sv_single[17] = sv_all[17]

ref_seqs = pd.read_csv('data/therm_easy.csv')['sequence'].tolist()

gen_seqs = []
for i in tqdm(range(100), desc='L17 therm'):
    seq = ref_seqs[i % len(ref_seqs)]
    _, _, seq_token = batch_converter([('protein', seq)])
    seq_token = seq_token.to(device)
    new_seq = generate_sequences(seq_token, model, sv_single, 0.1, alphabet, temperature=1.0, top_p=0.9)
    gen_seqs.append(new_seq)

pd.DataFrame({'sequence': gen_seqs}).to_csv('new-results/therm/single_layer/L17_therm_easy.csv', index=False)
print(f'Generated {len(gen_seqs)} sequences')
"
echo "[DONE] Step 7 at $(date)"

# Oracle eval for L17
CUDA_VISIBLE_DEVICES=0 python evaluation/oracle/evaluate_oracle.py \
    --input_csv "new-results/therm/single_layer/L17_therm_easy.csv" \
    --property therm \
    --predictor_path "evaluation/oracle/thermostability/therm_predictor_nocdhit.pt" \
    --ref_csv "data/therm_easy.csv" \
    --device cuda:0
echo "[DONE] L17 Oracle at $(date)"

# ============================================================
# Step 8: GLP steering (u=0.5, 0.9)
# ============================================================
echo ""
echo ">>> Step 8: GLP steering (therm, u=0.5, 0.9)"
for u in 0.5 0.9; do
    echo "--- GLP u=$u ---"
    CUDA_VISIBLE_DEVICES=0 python scripts/glp/steering_with_glp.py \
        --u $u --n_gen 100 --seed 42 \
        --sv_path "saved_steering_vectors/650M_therm_steering_vectors.pt" \
        --ref_data "data/therm_easy.csv" \
        --predictor_path "evaluation/oracle/thermostability/therm_predictor_nocdhit.pt" \
        --property therm \
        --output_dir "new-results/therm/glp" \
        --device cuda:0
    echo "[DONE] GLP u=$u at $(date)"
done

# ============================================================
# Step 9: pPPL evaluation (all groups)
# ============================================================
echo ""
echo ">>> Step 9: pPPL evaluation"
CUDA_VISIBLE_DEVICES=0,1 python evaluation/ppl/evaluate_ppl.py \
    --input_csvs \
        new-results/therm/baseline/steering_therm_easy.csv \
        new-results/therm/baseline/no_steering_therm_easy.csv \
        new-results/therm/baseline/steering_therm_hard.csv \
        new-results/therm/baseline/no_steering_therm_hard.csv \
        new-results/therm/single_layer/L17_therm_easy.csv \
        new-results/therm/glp/L17_glp_u0.5.csv \
        new-results/therm/glp/L17_glp_u0.9.csv \
        data/therm_easy.csv \
        data/therm_hard.csv \
    --labels \
        steering_easy no_steering_easy \
        steering_hard no_steering_hard \
        L17_easy \
        "L17+GLP_u0.5" "L17+GLP_u0.9" \
        ref_easy ref_hard \
    --model 3B --gpu_ids 0 1 \
    --output_csv new-results/therm/ppl/therm_ppl.csv
echo "[DONE] Step 9 pPPL at $(date)"

# ============================================================
# Step 10: ESMFold evaluation
# ============================================================
echo ""
echo ">>> Step 10: ESMFold evaluation"
echo "Switching to esmfold env..."

eval "$(conda shell.bash hook)"
conda activate esmfold

CUDA_VISIBLE_DEVICES=1 python evaluation/esmfold/evaluate_esmfold.py \
    --input_csvs \
        new-results/therm/baseline/steering_therm_easy.csv \
        new-results/therm/baseline/no_steering_therm_easy.csv \
        new-results/therm/single_layer/L17_therm_easy.csv \
        new-results/therm/glp/L17_glp_u0.5.csv \
        new-results/therm/glp/L17_glp_u0.9.csv \
        data/therm_easy.csv \
    --labels \
        all_layer_steering no_steering \
        L17_steering \
        "L17+GLP_u0.5" "L17+GLP_u0.9" \
        ref_easy \
    --output_csv new-results/therm/esmfold/therm_esmfold.csv \
    --device cuda:0
echo "[DONE] Step 10 ESMFold at $(date)"

echo ""
echo "============================================================"
echo "All therm experiments completed at $(date)"
echo "Results in new-results/therm/"
echo "============================================================"
