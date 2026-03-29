#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 2 ]; then
  echo "Usage: $0 [sol|therm|all] [random|aspo|all]" >&2
  exit 2
fi

property_arg="${1:-all}"
mask_strategy_arg="${2:-all}"

case "$property_arg" in
  sol)
    properties=(sol)
    ;;
  therm)
    properties=(therm)
    ;;
  all)
    properties=(sol therm)
    ;;
  *)
    echo "Unsupported property: $property_arg" >&2
    exit 2
    ;;
esac

case "$mask_strategy_arg" in
  random)
    mask_strategies=(random)
    ;;
  aspo)
    mask_strategies=(aspo)
    ;;
  all)
    mask_strategies=(random aspo)
    ;;
  *)
    echo "Unsupported mask strategy: $mask_strategy_arg" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

colabfold_cmd="${COLABFOLD_CMD:-colabfold_batch}"
colabfold_data_dir="${COLABFOLD_DATA_DIR:-${HOME}/.cache/colabfold}"
device="${DEVICE:-cuda}"
ppl_model="${PPL_MODEL:-150M}"
skip_ppl="${SKIP_PPL:-1}"
msa_mode="${COLABFOLD_MSA_MODE:-mmseqs2_uniref_env}"
output_root="${OUTPUT_ROOT:-results/mvp_colabfold_af2}"
cache_csv="${PLDDT_CACHE_CSV:-results/shared_plddt_cache.csv}"
colabfold_output_root="${COLABFOLD_OUTPUT_ROOT:-results/colabfold_af2_outputs}"
host_url="${COLABFOLD_HOST_URL:-}"
max_test_seqs="${N:-}"
num_rounds="${NUM_ROUNDS:-}"

mkdir -p "$colabfold_data_dir" "$output_root" "$colabfold_output_root"

for property in "${properties[@]}"; do
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
  esac

  for mask_strategy in "${mask_strategies[@]}"; do
    case "$mask_strategy" in
      random)
        modes=(no_steering naive_steering alignment_steering)
        ;;
      aspo)
        modes=(naive_steering alignment_steering)
        ;;
    esac

    output_dir="${output_root}/${property}_${mask_strategy}"
    colabfold_output_dir="${colabfold_output_root}/${property}_${mask_strategy}"

    cmd=(
      python mvp_eval_pipeline.py
      --property "$property"
      --input_csv "$input_csv"
      --family_csv data/lysozyme_uniref50_2k/preprocessed/lysozyme_train.csv
      --family_score_col "$family_score_col"
      --input_score_col "$input_score_col"
      --natural_db_path data/lysozyme_uniref50_2k/preprocessed/lysozyme_train.csv
      --output_dir "$output_dir"
      --mask_strategy "$mask_strategy"
      --modes "${modes[@]}"
      --device "$device"
      --ppl_model "$ppl_model"
      --compute_plddt
      --plddt_backend colabfold
      --colabfold_batch_cmd "$colabfold_cmd"
      --colabfold_msa_mode "$msa_mode"
      --colabfold_model_type alphafold2_ptm
      --colabfold_rank plddt
      --colabfold_num_models 1
      --colabfold_num_seeds 1
      --colabfold_data_dir "$colabfold_data_dir"
      --plddt_cache_csv "$cache_csv"
      --colabfold_output_dir "$colabfold_output_dir"
    )

    if [ -n "$host_url" ]; then
      cmd+=(--colabfold_host_url "$host_url")
    fi
    if [ -n "$max_test_seqs" ]; then
      cmd+=(--n "$max_test_seqs")
    fi
    if [ -n "$num_rounds" ]; then
      cmd+=(--num_rounds "$num_rounds")
    fi
    if [ "$skip_ppl" != "0" ]; then
      cmd+=(--skip_ppl)
    fi

    printf 'Running %s %s with ColabFold/MMseqs2...\n' "$property" "$mask_strategy"
    "${cmd[@]}"
  done
done
