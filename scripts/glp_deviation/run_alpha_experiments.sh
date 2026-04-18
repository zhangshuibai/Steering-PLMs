#!/bin/bash
set -euo pipefail

# ============================================================
# Alpha scaling + All-Layer + L17-GLP experiments
#
# Settings per dataset:
#   L17 single-layer α ∈ {1, 2, 3, 5, 10}
#     × {no GLP, GLP u=0.5, GLP u=0.9}  = 15
#   All-layer α ∈ {1, 2, 3}
#     × {no GLP, L17-GLP u=0.5, L17-GLP u=0.9}  = 9
#   Total: 24 settings × 4 datasets (sol_easy, sol_hard, therm_easy, therm_hard)
#   500 seqs per setting
#
# RNG: per-sequence deterministic (same seq_idx = same mask positions
# across all settings, enabling controlled GLP vs non-GLP comparison).
# ============================================================

eval "$(conda shell.bash hook)"
conda activate steering
cd /data/szhang967/Steering-PLMs

OUTROOT="new-results/glp_deviation"
mkdir -p "$OUTROOT"
LOG="$OUTROOT/alpha_experiment.log"

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "Alpha scaling experiments started at $(date)"
echo "============================================================"

GPU=3
N_GEN=500
ESMFOLD_N=100

SETTINGS=()
for ALPHA in 1 2 3 5 10; do
    SETTINGS+=("L17_a${ALPHA}")
    for U in 0.5 0.9; do
        SETTINGS+=("L17_a${ALPHA}_GLP_u${U}")
    done
done
for ALPHA in 1 2 3; do
    SETTINGS+=("allL_a${ALPHA}")
    for U in 0.5 0.9; do
        SETTINGS+=("allL_a${ALPHA}_L17GLP_u${U}")
    done
done
echo "Will run ${#SETTINGS[@]} settings per dataset, $N_GEN seqs each"

# Completeness-aware skip
is_complete() {
    local file=$1
    local expected=$2
    [ -f "$file" ] && [ "$(wc -l < "$file")" -eq "$((expected + 1))" ]
}

run_dataset() {
    local PROP=$1
    local DIFF=$2
    local REF="data/${PROP}_${DIFF}.csv"
    local SV="saved_steering_vectors/650M_${PROP}_steering_vectors.pt"
    local OUTDIR="$OUTROOT/${PROP}_${DIFF}_alpha"
    local EVALDIR="$OUTDIR/_eval"
    mkdir -p "$OUTDIR" "$EVALDIR"

    if [ ! -f "$SV" ]; then
        echo "ERROR: steering vector $SV not found"
        return 1
    fi

    for S in "${SETTINGS[@]}"; do
        local CSV="$OUTDIR/${S}.csv"
        if is_complete "$CSV" "$N_GEN"; then
            echo "  [skip complete] ${PROP}_${DIFF}/$S"
            continue
        fi
        rm -f "$CSV"
        CUDA_VISIBLE_DEVICES=$GPU python scripts/glp_deviation/generate_alpha.py \
            --setting "$S" --ref_data "$REF" --sv_path "$SV" \
            --n_gen "$N_GEN" --output_csv "$CSV" --device cuda:0 --seed 42
        echo "[DONE] ${PROP}_${DIFF}/$S at $(date)"
    done
}

# Run all 4 splits
run_dataset sol easy
run_dataset sol hard
run_dataset therm easy
run_dataset therm hard

echo ""
echo ">>> Generation done at $(date). Starting oracle..."

for TASK in sol_easy sol_hard therm_easy therm_hard; do
    OUTDIR="$OUTROOT/${TASK}_alpha"
    EVALDIR="$OUTDIR/_eval"
    PROP="${TASK%%_*}"
    if [ "$PROP" == "sol" ]; then
        PRED="evaluation/oracle/solubility/sol_predictor_final.pt"
    else
        PRED="evaluation/oracle/thermostability/therm_predictor_nocdhit.pt"
    fi
    for csv in "$OUTDIR"/*.csv; do
        [ -e "$csv" ] || continue
        base=$(basename "$csv" .csv)
        scored="$EVALDIR/${base}_scored.csv"
        if is_complete "$scored" "$N_GEN"; then
            echo "  [skip complete] oracle $TASK/$base"
            continue
        fi
        rm -f "$scored"
        CUDA_VISIBLE_DEVICES=$GPU python evaluation/oracle/evaluate_oracle.py \
            --input_csv "$csv" --property "$PROP" --predictor_path "$PRED" \
            --output_csv "$scored" --device cuda:0 2>&1 | tail -2
    done
done

echo ""
echo ">>> Oracle done at $(date). Starting ESMFold..."

conda activate esmfold
for TASK in sol_easy sol_hard therm_easy therm_hard; do
    OUTDIR="$OUTROOT/${TASK}_alpha"
    EVALDIR="$OUTDIR/_eval"
    INPUTS=()
    LABELS=()
    for csv in "$OUTDIR"/*.csv; do
        [ -e "$csv" ] || continue
        base=$(basename "$csv" .csv)
        INPUTS+=("$csv")
        LABELS+=("${TASK}_${base}")
    done
    if [ ${#INPUTS[@]} -eq 0 ]; then continue; fi
    CUDA_VISIBLE_DEVICES=$GPU python evaluation/esmfold/evaluate_esmfold.py \
        --input_csvs "${INPUTS[@]}" --labels "${LABELS[@]}" --max_seqs $ESMFOLD_N \
        --output_csv "$EVALDIR/esmfold_results.csv" --device cuda:0
    echo "[DONE] ESMFold $TASK at $(date)"
done

echo ""
echo "============================================================"
echo "All done at $(date)"
echo "============================================================"
