#!/bin/bash
set -euo pipefail

# ============================================================
# Phase B: Alpha scaling on fitness benchmarks (TrpB/CreiLOV/GFP)
#
# For each dataset, tests the same 9 settings as Phase A:
#   L17 α ∈ {1, 3, 5} × {no-GLP, GLP u=0.5, GLP u=0.9}     = 9
#   allL α ∈ {1, 3} × {no-GLP, L17-GLP u=0.5, L17-GLP u=0.9} = 6
#   Total: 15 settings × 4 benchmarks (TrpB, CreiLOV, GFP_R4T2, GFP_R1T8)
#
# Each dataset uses its native generation protocol but with alpha scaling applied.
# ============================================================

eval "$(conda shell.bash hook)"
conda activate steering
cd /data/szhang967/Steering-PLMs

OUTROOT="new-results/glp_deviation/phase_b"
mkdir -p "$OUTROOT"
LOG="$OUTROOT/experiment.log"
exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "Phase B (fitness benchmark alpha scaling) started at $(date)"
echo "============================================================"

GPU=3

# Reduced set compared to Phase A (to keep time manageable)
SETTINGS=()
for ALPHA in 1 3 5; do
    SETTINGS+=("L17_a${ALPHA}")
    for U in 0.5 0.9; do
        SETTINGS+=("L17_a${ALPHA}_GLP_u${U}")
    done
done
for ALPHA in 1 3; do
    SETTINGS+=("allL_a${ALPHA}")
    for U in 0.5 0.9; do
        SETTINGS+=("allL_a${ALPHA}_L17GLP_u${U}")
    done
done
echo "Will run ${#SETTINGS[@]} settings per dataset"

is_complete() {
    local file=$1
    local expected=$2
    [ -f "$file" ] && [ "$(wc -l < "$file")" -eq "$((expected + 1))" ]
}

run_benchmark() {
    local NAME=$1       # subdir name
    local REF=$2
    local SV=$3
    local PROTOCOL=$4   # arg for generate_alpha.py
    local PROTOCOL_ARGS=$5  # extra protocol args
    local N_GEN=$6

    local OUTDIR="$OUTROOT/$NAME"
    local EVALDIR="$OUTDIR/_eval"
    mkdir -p "$OUTDIR" "$EVALDIR"

    for S in "${SETTINGS[@]}"; do
        local CSV="$OUTDIR/${S}.csv"
        if is_complete "$CSV" "$N_GEN"; then
            echo "  [skip] $NAME/$S"
            continue
        fi
        rm -f "$CSV"
        CUDA_VISIBLE_DEVICES=$GPU python scripts/glp_deviation/generate_alpha.py \
            --setting "$S" --ref_data "$REF" --sv_path "$SV" \
            --n_gen "$N_GEN" --output_csv "$CSV" --device cuda:0 --seed 42 \
            --protocol "$PROTOCOL" $PROTOCOL_ARGS
        echo "[DONE] $NAME/$S at $(date)"
    done
}

# ---------- TrpB ----------
# Protocol: 1 round, fixed 4 sites at 0-indexed 182,183,226,227
# We need to prepare the reference CSV of low-fitness TrpB variants
# (same as before fitness benchmark)
PREP_REF_TRPB="$OUTROOT/refs/trpb_low_fitness.csv"
mkdir -p "$(dirname $PREP_REF_TRPB)"
if [ ! -f "$PREP_REF_TRPB" ]; then
    python -c "
import pandas as pd, numpy as np
df = pd.read_csv('data/benchmarks/processed/trpb/trpb_processed.csv')
low = df[df['fitness'] < df['fitness'].quantile(0.1)].reset_index(drop=True)
rng = np.random.RandomState(42)
idx = rng.choice(len(low), size=min(200, len(low)), replace=False)
low.iloc[idx][['sequence']].to_csv('$PREP_REF_TRPB', index=False)
print(f'Prepared {min(200, len(low))} TrpB low-fitness refs')
"
fi
run_benchmark trpb "$PREP_REF_TRPB" \
    "saved_steering_vectors/650M_trpb_fitness_steering_vectors.pt" \
    fixed_sites "--fixed_positions 182 183 226 227 --n_rounds 1" \
    200

# ---------- CreiLOV ----------
PREP_REF_CREILOV="$OUTROOT/refs/creilov_low_fitness.csv"
if [ ! -f "$PREP_REF_CREILOV" ]; then
    python -c "
import pandas as pd, numpy as np
df = pd.read_csv('data/benchmarks/processed/creilov/creilov_processed.csv')
low = df[df['fitness'] < df['fitness'].quantile(0.1)].reset_index(drop=True)
rng = np.random.RandomState(42)
idx = rng.choice(len(low), size=min(200, len(low)), replace=False)
low.iloc[idx][['sequence']].to_csv('$PREP_REF_CREILOV', index=False)
print(f'Prepared {min(200, len(low))} CreiLOV low-fitness refs')
"
fi
run_benchmark creilov "$PREP_REF_CREILOV" \
    "saved_steering_vectors/650M_creilov_fitness_steering_vectors.pt" \
    fixed_sites "--fixed_positions_from data/benchmarks/processed/creilov/creilov_processed.csv --n_rounds 1" \
    200

# ---------- GFP R=4×T=2 (Kirjner hard) ----------
PREP_REF_GFP="$OUTROOT/refs/gfp_hard.csv"
if [ ! -f "$PREP_REF_GFP" ]; then
    python -c "
import pandas as pd, numpy as np
df = pd.read_csv('data/benchmarks/processed/gfp_kirjner/hard.csv')
rng = np.random.RandomState(42)
idx = rng.choice(len(df), size=min(200, len(df)), replace=False)
df.iloc[idx][['sequence']].to_csv('$PREP_REF_GFP', index=False)
print(f'Prepared {min(200, len(df))} GFP hard refs')
"
fi
run_benchmark gfp_hard_R4T2 "$PREP_REF_GFP" \
    "saved_steering_vectors/650M_gfp_fitness_steering_vectors.pt" \
    n_per_round "--n_rounds 4 --n_positions 2" \
    200

# ---------- GFP R=1×T=8 ----------
run_benchmark gfp_hard_R1T8 "$PREP_REF_GFP" \
    "saved_steering_vectors/650M_gfp_fitness_steering_vectors.pt" \
    n_per_round "--n_rounds 1 --n_positions 8" \
    200

echo ""
echo ">>> Phase B generation done at $(date). Starting oracle..."

# ---------- Oracle eval ----------
eval_trpb() {
    local OUTDIR=$1
    local EVALDIR="$OUTDIR/_eval"
    for csv in "$OUTDIR"/*.csv; do
        [ -e "$csv" ] || continue
        local base
        base=$(basename "$csv" .csv)
        local scored="$EVALDIR/${base}_scored.csv"
        if is_complete "$scored" 200; then continue; fi
        rm -f "$scored"
        # TrpB: use lookup via score_fitness_benchmark.py
        CUDA_VISIBLE_DEVICES=$GPU python scripts/fitness_benchmarks/score_fitness_benchmark.py \
            --input_csv "$csv" --dataset trpb \
            --output_csv "$scored" \
            --summary_json "$EVALDIR/${base}_summary.json" \
            --device cuda:0 2>&1 | tail -2
    done
}

eval_oracle_generic() {
    local OUTDIR=$1
    local DATASET=$2
    local PRED=$3
    local EVALDIR="$OUTDIR/_eval"
    for csv in "$OUTDIR"/*.csv; do
        [ -e "$csv" ] || continue
        local base
        base=$(basename "$csv" .csv)
        local scored="$EVALDIR/${base}_scored.csv"
        if is_complete "$scored" 200; then continue; fi
        rm -f "$scored"
        CUDA_VISIBLE_DEVICES=$GPU python scripts/fitness_benchmarks/score_fitness_benchmark.py \
            --input_csv "$csv" --dataset "$DATASET" \
            --predictor_path "$PRED" \
            --output_csv "$scored" \
            --summary_json "$EVALDIR/${base}_summary.json" \
            --device cuda:0 2>&1 | tail -2
    done
}

eval_trpb "$OUTROOT/trpb"
eval_oracle_generic "$OUTROOT/creilov" creilov \
    "evaluation/oracle/creilov/creilov_predictor_final.pt"
eval_oracle_generic "$OUTROOT/gfp_hard_R4T2" gfp \
    "evaluation/oracle/gfp_sarkisyan/gfp_sarkisyan_predictor_final.pt"
eval_oracle_generic "$OUTROOT/gfp_hard_R1T8" gfp \
    "evaluation/oracle/gfp_sarkisyan/gfp_sarkisyan_predictor_final.pt"

echo ""
echo ">>> Oracle done at $(date). Starting ESMFold..."

conda activate esmfold
for NAME in trpb creilov gfp_hard_R4T2 gfp_hard_R1T8; do
    OUTDIR="$OUTROOT/$NAME"
    EVALDIR="$OUTDIR/_eval"
    INPUTS=()
    LABELS=()
    for csv in "$OUTDIR"/*.csv; do
        [ -e "$csv" ] || continue
        base=$(basename "$csv" .csv)
        INPUTS+=("$csv")
        LABELS+=("${NAME}_${base}")
    done
    if [ ${#INPUTS[@]} -eq 0 ]; then continue; fi
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/esmfold/evaluate_esmfold.py \
        --input_csvs "${INPUTS[@]}" --labels "${LABELS[@]}" --max_seqs 100 \
        --output_csv "$EVALDIR/esmfold_results.csv" --device cuda:0
    echo "[DONE] ESMFold $NAME at $(date)"
done

echo ""
echo "============================================================"
echo "Phase B all done at $(date)"
echo "============================================================"
