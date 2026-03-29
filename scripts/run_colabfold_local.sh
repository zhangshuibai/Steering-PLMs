#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_fasta> <output_dir>" >&2
  exit 2
fi

input_fasta="$1"
output_dir="$2"

: "${COLABFOLD_DB_DIR:?Set COLABFOLD_DB_DIR to your local ColabFold/MMseqs database directory.}"

msa_dir="${output_dir}/msa"
pred_dir="${output_dir}/predictions"

mkdir -p "$msa_dir" "$pred_dir"

# The official local ColabFold flow is:
# 1. colabfold_search <queries.fa> <db_dir> <msa_dir>
# 2. colabfold_batch <msa_dir> <pred_dir>
MMSEQS_IGNORE_INDEX="${MMSEQS_IGNORE_INDEX:-1}" \
  colabfold_search "$input_fasta" "$COLABFOLD_DB_DIR" "$msa_dir"

colabfold_batch "$msa_dir" "$pred_dir"

