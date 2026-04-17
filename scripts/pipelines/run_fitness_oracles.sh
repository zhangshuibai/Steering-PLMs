#!/bin/bash
set -euo pipefail

# Train wet-lab fitness oracles for the strong-recommend benchmarks listed in
# data/benchmarks/protein_fitness_datasets.csv:
#   - TrpB (Johnston)
#   - CreiLOV combinatorial
#   - GFP Sarkisyan
#
# Usage:
#   bash scripts/pipelines/run_fitness_oracles.sh
#
# Or override the default raw-file locations:
#   TRPB_RAW=/path/to/data.zip \
#   CREILOV_RAW=/path/to/sb2c00662_si_002.xlsx \
#   GFP_SARKISYAN_RAW=/path/to/amino_acid_genotypes_to_brightness.tsv \
#   bash scripts/pipelines/run_fitness_oracles.sh
#
# The prepare_* scripts normalize the raw tables into:
#   data/benchmarks/processed/<dataset>/<dataset>_processed.csv
# Then the dataset-specific train_* wrappers write:
#   evaluation/oracle/<dataset>/<dataset>_predictor_final.pt

eval "$(conda shell.bash hook)"
conda activate steering

ROOT_DIR="/data/szhang967/Steering-PLMs"
cd "$ROOT_DIR"

TRPB_RAW="${TRPB_RAW:-}"
CREILOV_RAW="${CREILOV_RAW:-}"
GFP_SARKISYAN_RAW="${GFP_SARKISYAN_RAW:-}"

echo "=========================================="
echo "Wet-Lab Fitness Oracle Pipeline"
echo "=========================================="

if [[ -n "$TRPB_RAW" && -f "$TRPB_RAW" ]]; then
    python scripts/benchmarks/prepare_trpb.py --input_path "$TRPB_RAW"
elif [[ -f "data/benchmarks/raw/trpb/data.zip" ]]; then
    python scripts/benchmarks/prepare_trpb.py
fi
if [[ -n "$CREILOV_RAW" && -f "$CREILOV_RAW" ]]; then
    python scripts/benchmarks/prepare_creilov.py --input_path "$CREILOV_RAW"
elif [[ -f "data/benchmarks/raw/creilov/sb2c00662_si_002.xlsx" ]]; then
    python scripts/benchmarks/prepare_creilov.py
fi
if [[ -n "$GFP_SARKISYAN_RAW" && -f "$GFP_SARKISYAN_RAW" ]]; then
    python scripts/benchmarks/prepare_gfp_sarkisyan.py --input_path "$GFP_SARKISYAN_RAW"
elif [[ -f "data/benchmarks/raw/gfp_sarkisyan/amino_acid_genotypes_to_brightness.tsv" ]]; then
    python scripts/benchmarks/prepare_gfp_sarkisyan.py
fi

if [[ -f "data/benchmarks/processed/trpb/trpb_processed.csv" ]]; then
    python evaluation/oracle/trpb/train_trpb_predictor.py
else
    echo "Skipping TrpB: processed CSV missing"
fi

if [[ -f "data/benchmarks/processed/creilov/creilov_processed.csv" ]]; then
    python evaluation/oracle/creilov/train_creilov_predictor.py
else
    echo "Skipping CreiLOV: processed CSV missing"
fi

if [[ -f "data/benchmarks/processed/gfp_sarkisyan/gfp_sarkisyan_processed.csv" ]]; then
    python evaluation/oracle/gfp_sarkisyan/train_gfp_sarkisyan_predictor.py
else
    echo "Skipping GFP Sarkisyan: processed CSV missing"
fi

echo "=========================================="
echo "Pipeline complete"
echo "Targets for first usable run:"
echo "  TrpB:          Spearman >= 0.65"
echo "  CreiLOV:       Spearman >= 0.50"
echo "  GFP Sarkisyan: Spearman >= 0.70"
echo "=========================================="
