#!/bin/bash
set -e

# ============================================================
# Re-run sol + therm experiments with fixed layer alignment
# (extract_esm2_features now uses repr_layers=range(1, n_layer+1))
# All L17 experiments now use self.layers[16] = ESM2 API repr_layers[17]
#
# Steps:
#   1. Re-extract steering vectors (new convention)
#   2. Generate: reference, no_steering, all_layer, L17, L17+GLP u={0.1,0.5,0.9,1.0}
#   3. Oracle evaluation
#   4. ESMFold (200/group)
#   pPPL deferred
# ============================================================

eval "$(conda shell.bash hook)"
conda activate steering
cd /data/szhang967/Steering-PLMs

OUTROOT="new-results/fixed_layer"
LOG="$OUTROOT/experiment.log"
mkdir -p "$OUTROOT"

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "Fixed-layer sol+therm experiments started at $(date)"
echo "============================================================"

GPU=3

# ============================================================
# Step 1: Re-extract steering vectors with fixed convention
# ============================================================
echo ""
echo ">>> Step 1: Re-extract steering vectors"
for prop in sol therm; do
    SV="saved_steering_vectors/650M_${prop}_steering_vectors_fixed.pt"
    if [ -f "$SV" ]; then
        echo "  [skip] $SV exists"
        continue
    fi
    if [ "$prop" == "sol" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python scripts/extraction/extract_esm2_steering_vec.py \
            --model "650M" --num_data 100 --property "sol" \
            --data_path "data/sol_filtered.csv" \
            --theshold_pos 0.5 --theshold_neg 0.2 \
            --save_folder saved_steering_vectors \
            2>&1
        mv saved_steering_vectors/650M_sol_steering_vectors.pt "$SV"
    else
        CUDA_VISIBLE_DEVICES=$GPU python scripts/extraction/extract_esm2_steering_vec.py \
            --model "650M" --num_data 100 --property "therm" \
            --data_path "data/therm_filtered.csv" \
            --theshold_pos 70.0 --theshold_neg 50.0 \
            --save_folder saved_steering_vectors \
            2>&1
        mv saved_steering_vectors/650M_therm_steering_vectors.pt "$SV"
    fi
    echo "[DONE] $prop steering vec at $(date)"
done

# ============================================================
# Generation + Oracle for each property × setting
# ============================================================

generate_and_eval() {
    local PROP=$1      # sol or therm
    local DIFF=$2      # easy or hard
    local SUBDIR=$3    # output subdir name
    local SV_PATH=$4
    local REF_DATA=$5

    local OUTDIR="$OUTROOT/$SUBDIR"
    mkdir -p "$OUTDIR"

    echo ""
    echo ">>> [$SUBDIR] generation"

    # Reference
    if [ ! -f "$OUTDIR/reference.csv" ]; then
        head -101 "$REF_DATA" > "$OUTDIR/reference.csv"  # header + 100 rows
    fi

    # No Steering
    if [ ! -f "$OUTDIR/no_steering.csv" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python scripts/generation/steering_esm2_generation.py \
            --model "650M" --property "$PROP" --seed 42 \
            --ref_data_path "$REF_DATA" \
            --output_file "$OUTDIR/no_steering.csv" \
            --n 100
        echo "  [gen done] no_steering at $(date)"
    fi

    # All-Layer
    if [ ! -f "$OUTDIR/all_layer.csv" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python scripts/generation/steering_esm2_generation.py \
            --model "650M" --property "$PROP" --seed 42 \
            --ref_data_path "$REF_DATA" \
            --output_file "$OUTDIR/all_layer.csv" \
            --steering --sv_from saved_steering_vectors \
            --n 100
        echo "  [gen done] all_layer at $(date)"
    fi

    # L17 (single layer at self.layers[16])
    if [ ! -f "$OUTDIR/L17.csv" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python -c "
import sys, os, types, torch, numpy as np, pandas as pd
sys.path.insert(0, '.')
torch.manual_seed(42); np.random.seed(42)
from module.steerable_esm2 import steering_forward
from utils.esm2_utils import load_esm2_model, generate_sequences
model, alphabet = load_esm2_model('650M')
model.steering_forward = types.MethodType(steering_forward, model)
bc = alphabet.get_batch_converter()
pos_sv, neg_sv = torch.load('$SV_PATH', weights_only=False)
sv = torch.zeros_like(pos_sv - neg_sv).cuda()
sv[16] = (pos_sv - neg_sv)[16].cuda()  # self.layers[16] = L17
ref_seqs = pd.read_csv('$REF_DATA')['sequence'].tolist()
gen = []
from tqdm import tqdm
for i in tqdm(range(100)):
    seq = ref_seqs[i % len(ref_seqs)]
    _, _, t = bc([('protein', seq)]); t = t.cuda()
    gen.append(generate_sequences(t, model, sv, 0.1, alphabet, temperature=1.0, top_p=0.9))
pd.DataFrame({'sequence': gen}).to_csv('$OUTDIR/L17.csv', index=False)
"
        echo "  [gen done] L17 at $(date)"
    fi

    # L17+GLP variants
    for u in 0.1 0.5 0.9 1.0; do
        local fname="L17_GLP_u${u}.csv"
        if [ ! -f "$OUTDIR/$fname" ]; then
            CUDA_VISIBLE_DEVICES=$GPU python scripts/glp/steering_with_glp.py \
                --u $u --n_gen 100 --seed 42 \
                --sv_path "$SV_PATH" \
                --ref_data "$REF_DATA" \
                --predictor_path "evaluation/oracle/solubility/sol_predictor_final.pt" \
                --property "$PROP" \
                --output_dir "$OUTDIR" \
                --device cuda:0 \
                --num_timesteps 25
            # Rename output
            mv "$OUTDIR/L16_glp_u${u}.csv" "$OUTDIR/$fname" 2>/dev/null || true
            mv "$OUTDIR/L16_glp_u${u}_scored.csv" "$OUTDIR/${fname%.csv}_scored.csv" 2>/dev/null || true
            echo "  [gen done] L17_GLP_u$u at $(date)"
        fi
    done

    # Oracle eval
    echo "  >>> Oracle evaluation"
    if [ "$PROP" == "sol" ]; then
        local PRED="evaluation/oracle/solubility/sol_predictor_final.pt"
    else
        local PRED="evaluation/oracle/thermostability/therm_predictor_nocdhit.pt"
    fi
    for csv in "$OUTDIR"/*.csv; do
        local base=$(basename "$csv" .csv)
        if [[ "$base" == *_scored* ]] || [[ "$base" == *_summary* ]]; then continue; fi
        local scored="$OUTDIR/${base}_scored.csv"
        if [ -f "$scored" ]; then continue; fi
        CUDA_VISIBLE_DEVICES=$GPU python evaluation/oracle/evaluate_oracle.py \
            --input_csv "$csv" \
            --property "$PROP" \
            --predictor_path "$PRED" \
            --ref_csv "$REF_DATA" \
            --device cuda:0 2>&1 | tail -5
    done

    echo "[DONE] $SUBDIR at $(date)"
}

# Sol easy
generate_and_eval sol easy sol_easy \
    "saved_steering_vectors/650M_sol_steering_vectors_fixed.pt" \
    "data/sol_easy.csv"

# Sol hard
generate_and_eval sol hard sol_hard \
    "saved_steering_vectors/650M_sol_steering_vectors_fixed.pt" \
    "data/sol_hard.csv"

# Therm easy
generate_and_eval therm easy therm_easy \
    "saved_steering_vectors/650M_therm_steering_vectors_fixed.pt" \
    "data/therm_easy.csv"

# Therm hard
generate_and_eval therm hard therm_hard \
    "saved_steering_vectors/650M_therm_steering_vectors_fixed.pt" \
    "data/therm_hard.csv"

echo ""
echo "============================================================"
echo "Phase 1 complete at $(date)"
echo "Starting Phase 2: ESMFold"
echo "============================================================"

# ============================================================
# Phase 2: ESMFold
# ============================================================
eval "$(conda shell.bash hook)"
conda activate esmfold

for subdir in sol_easy sol_hard therm_easy therm_hard; do
    OUTDIR="$OUTROOT/$subdir"
    INPUTS=""
    LABELS=""
    for csv in "$OUTDIR"/*.csv; do
        base=$(basename "$csv" .csv)
        if [[ "$base" == *_scored* ]] || [[ "$base" == *_summary* ]]; then continue; fi
        INPUTS="$INPUTS $csv"
        LABELS="$LABELS ${subdir}_${base}"
    done
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/esmfold/evaluate_esmfold.py \
        --input_csvs $INPUTS \
        --labels $LABELS \
        --max_seqs 200 \
        --output_csv "$OUTDIR/esmfold_results.csv" \
        --device cuda:0
    echo "[DONE] ESMFold $subdir at $(date)"
done

echo ""
echo "============================================================"
echo "All fixed-layer sol+therm experiments completed at $(date)"
echo "Results in $OUTROOT/"
echo "============================================================"
