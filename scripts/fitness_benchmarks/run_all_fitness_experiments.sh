#!/bin/bash
set -e

# ============================================================
# Fitness benchmarks: GFP, CreiLOV, TrpB
# Priority order: GFP > CreiLOV > TrpB
# Phase 1 (GPU 0, steering env): generation + oracle scoring
# Phase 2 (GPU 1, esmfold env): ESMFold (pLDDT/pTM) on 200/group
# pPPL is deferred to later.
# ============================================================

eval "$(conda shell.bash hook)"
conda activate steering
cd /data/szhang967/Steering-PLMs

OUTROOT="new-results/fitness_benchmarks"
LOG="$OUTROOT/experiment.log"
mkdir -p "$OUTROOT"

exec > >(tee -a "$LOG") 2>&1
echo "============================================================"
echo "Fitness benchmark experiments started at $(date)"
echo "============================================================"

DEVICE="cuda:0"
N_REFS=1000
SETTINGS="reference no_steering all_layer L17 L17_GLP_u0.1 L17_GLP_u0.5 L17_GLP_u0.9 L17_GLP_u1.0 random"

# ============================================================
# Step 0a: Prepare Kirjner GFP splits (if not cached)
# ============================================================

if [ ! -f "data/benchmarks/processed/gfp_kirjner/hard.csv" ]; then
    echo ">>> Preparing Kirjner GFP splits"
    python scripts/fitness_benchmarks/prepare_kirjner_gfp_splits.py
    echo "[DONE] Kirjner splits at $(date)"
fi

# ============================================================
# Step 0b: Extract steering vectors (if not cached)
# ============================================================

for dataset in gfp creilov trpb; do
    SV="saved_steering_vectors/650M_${dataset}_fitness_steering_vectors.pt"
    if [ ! -f "$SV" ]; then
        echo ""
        echo ">>> Extract steering vectors: $dataset"
        CUDA_VISIBLE_DEVICES=3 python scripts/fitness_benchmarks/extract_fitness_steering_vec.py \
            --dataset $dataset --n 100
        echo "[DONE] Steering vec: $dataset at $(date)"
    fi
done

# ============================================================
# Run a dataset x (difficulty, protocol) combination
# $1=dataset, $2=subdir, $3=--difficulty arg (or empty), $4=--gen_protocol arg
# ============================================================
run_group() {
    local dataset=$1
    local subdir=$2
    local diff_flag=$3
    local proto_flag=$4

    local OUTDIR="$OUTROOT/$subdir"
    mkdir -p "$OUTDIR"

    echo ""
    echo ">>> [$subdir] generation + oracle"

    for setting in $SETTINGS; do
        local csv="$OUTDIR/${setting}.csv"
        if [ -f "$csv" ]; then
            echo "  [skip] $setting already exists"
            continue
        fi
        CUDA_VISIBLE_DEVICES=3 python scripts/fitness_benchmarks/generate_fitness_benchmark.py \
            --dataset $dataset \
            $diff_flag \
            $proto_flag \
            --setting $setting \
            --n_refs $N_REFS \
            --n_gen $N_REFS \
            --output_dir "$OUTDIR" \
            --device $DEVICE \
            --seed 42
        echo "  [gen done] $setting at $(date)"
    done

    # Oracle scoring
    for setting in $SETTINGS; do
        local csv="$OUTDIR/${setting}.csv"
        local scored="$OUTDIR/${setting}_scored.csv"
        if [ -f "$scored" ]; then continue; fi
        CUDA_VISIBLE_DEVICES=3 python scripts/fitness_benchmarks/score_fitness_benchmark.py \
            --input_csv "$csv" \
            --dataset $dataset \
            --output_csv "$scored" \
            --summary_json "$OUTDIR/${setting}_summary.json" \
            --device $DEVICE
    done

    echo "[DONE] $subdir at $(date)"
}

# ============================================================
# GFP (highest priority): hard R=4xT=2, hard R=1xT=8, medium R=4xT=2
# ============================================================

run_group gfp gfp_hard_R4T2  "--difficulty hard"   "--gen_protocol R4T2"
run_group gfp gfp_hard_R1T8  "--difficulty hard"   "--gen_protocol R1T8"
run_group gfp gfp_medium_R4T2 "--difficulty medium" "--gen_protocol R4T2"

# ============================================================
# CreiLOV: single round, 15 fixed sites
# ============================================================

run_group creilov creilov "" ""

# ============================================================
# TrpB: single round, 4 fixed sites
# ============================================================

run_group trpb trpb "" ""

echo ""
echo "============================================================"
echo "Phase 1 complete at $(date)"
echo "Starting Phase 2: ESMFold"
echo "============================================================"

# ============================================================
# Phase 2: ESMFold (200 per group)
# ============================================================

conda activate esmfold

run_esmfold() {
    local subdir=$1
    local OUTDIR="$OUTROOT/$subdir"
    local INPUTS=""
    local LABELS=""
    for setting in $SETTINGS; do
        local csv="$OUTDIR/${setting}.csv"
        if [ -f "$csv" ]; then
            INPUTS="$INPUTS $csv"
            LABELS="$LABELS ${subdir}_${setting}"
        fi
    done
    CUDA_VISIBLE_DEVICES=3 python evaluation/esmfold/evaluate_esmfold.py \
        --input_csvs $INPUTS \
        --labels $LABELS \
        --max_seqs 200 \
        --output_csv "$OUTDIR/esmfold_results.csv" \
        --device cuda:0
    echo "[DONE] ESMFold $subdir at $(date)"
}

for subdir in gfp_hard_R4T2 gfp_hard_R1T8 gfp_medium_R4T2 creilov trpb; do
    run_esmfold $subdir
done

echo ""
echo "============================================================"
echo "All fitness benchmark experiments completed at $(date)"
echo "Results in $OUTROOT/"
echo "============================================================"
