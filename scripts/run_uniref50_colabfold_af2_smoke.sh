#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 2 ]; then
  echo "Usage: $0 [sol|therm] [random|aspo]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

property="${1:-sol}"
mask_strategy="${2:-random}"

N="${N:-2}" \
SKIP_PPL="${SKIP_PPL:-1}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-results/colabfold_af2_smoke}" \
COLABFOLD_OUTPUT_ROOT="${COLABFOLD_OUTPUT_ROOT:-results/colabfold_af2_smoke_outputs}" \
./scripts/run_uniref50_colabfold_af2_eval.sh "$property" "$mask_strategy"
