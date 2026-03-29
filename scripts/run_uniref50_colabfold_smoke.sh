#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 2 ]; then
  echo "Usage: $0 [sol|therm] [random|aspo]" >&2
  exit 2
fi

property="${1:-sol}"
mask_strategy="${2:-random}"

case "$property" in
  sol)
    input_csv="data/lysozyme_uniref50_2k/preprocessed/N_sol_test.csv"
    family_score_col="sol_prob"
    input_score_col="sol_prob"
    ;;
  therm)
    input_csv="data/lysozyme_uniref50_2k/preprocessed/N_therm_test.csv"
    family_score_col="therm_score"
    input_score_col="therm_score"
    ;;
  *)
    echo "Unsupported property: $property" >&2
    exit 2
    ;;
esac

case "$mask_strategy" in
  random)
    modes=(no_steering naive_steering alignment_steering)
    ;;
  aspo)
    modes=(naive_steering alignment_steering)
    ;;
  *)
    echo "Unsupported mask strategy: $mask_strategy" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${COLABFOLD_DB_DIR:?Set COLABFOLD_DB_DIR to your local ColabFold/MMseqs database directory.}"

colabfold_data_dir="${COLABFOLD_DATA_DIR:-${HOME}/.cache/colabfold}"
device="${DEVICE:-cuda}"
n="${N_SMOKE:-2}"
ppl_model="${PPL_MODEL:-150M}"
output_dir="${OUTPUT_DIR:-results/colabfold_smoke_${property}_${mask_strategy}}"
cache_csv="${PLDDT_CACHE_CSV:-results/shared_plddt_cache.csv}"

mkdir -p "$colabfold_data_dir"

python mvp_eval_pipeline.py \
  --property "$property" \
  --input_csv "$input_csv" \
  --family_csv data/lysozyme_uniref50_2k/preprocessed/lysozyme_train.csv \
  --family_score_col "$family_score_col" \
  --input_score_col "$input_score_col" \
  --natural_db_path data/lysozyme_uniref50_2k/preprocessed/lysozyme_train.csv \
  --output_dir "$output_dir" \
  --mask_strategy "$mask_strategy" \
  --modes "${modes[@]}" \
  --device "$device" \
  --n "$n" \
  --ppl_model "$ppl_model" \
  --compute_plddt \
  --plddt_backend colabfold \
  --colabfold_batch_cmd ./scripts/run_colabfold_local.sh \
  --colabfold_data_dir "$colabfold_data_dir" \
  --plddt_cache_csv "$cache_csv"
