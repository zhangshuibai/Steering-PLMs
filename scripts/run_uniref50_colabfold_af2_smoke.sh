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

: "${COLABFOLD_CMD:?Set COLABFOLD_CMD to a colabfold_batch executable, for example /content/cf_env/bin/colabfold_batch.}"
: "${COLABFOLD_DATA_DIR:?Set COLABFOLD_DATA_DIR to a writable AlphaFold weights directory.}"

device="${DEVICE:-cuda}"
n="${N_SMOKE:-2}"
ppl_model="${PPL_MODEL:-150M}"
msa_mode="${COLABFOLD_MSA_MODE:-mmseqs2_uniref_env}"
output_dir="${OUTPUT_DIR:-results/colabfold_af2_smoke_${property}_${mask_strategy}}"
cache_csv="${PLDDT_CACHE_CSV:-results/shared_plddt_cache.csv}"
colabfold_output_dir="${COLABFOLD_OUTPUT_DIR:-results/colabfold_af2_outputs}"
host_url="${COLABFOLD_HOST_URL:-}"

host_args=()
if [ -n "$host_url" ]; then
  host_args=(--colabfold_host_url "$host_url")
fi

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
  --colabfold_batch_cmd "$COLABFOLD_CMD" \
  --colabfold_msa_mode "$msa_mode" \
  --colabfold_model_type alphafold2_ptm \
  --colabfold_rank plddt \
  --colabfold_num_models 1 \
  --colabfold_num_seeds 1 \
  --colabfold_data_dir "$COLABFOLD_DATA_DIR" \
  "${host_args[@]}" \
  --plddt_cache_csv "$cache_csv" \
  --colabfold_output_dir "$colabfold_output_dir"
