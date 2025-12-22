# Example usage:
#   TASK="ailastatutes" LANGUAGE="en" CORPUS_PATH="" QRELS_PATH="" OUTPUT_PATH="" OVERWRITE=1 bash run_preparation.sh

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../../.." && pwd )"

TASK="${TASK:-${1:-}}"
LANGUAGE="${LANGUAGE:-en}"
ITEM="${ITEM:-pair}"
CORPUS_PATH="${CORPUS_PATH:-}"
QRELS_PATH="${QRELS_PATH:-}"
QUERIES_PATH="${QUERIES_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

python "${PROJECT_ROOT}/data_generation/data_preparation/code/item_filter.py" \
  --task "${TASK}" \
  --language "${LANGUAGE}" \
  --item "${ITEM}" \
  ${CORPUS_PATH:+--corpus_path "${CORPUS_PATH}"} \
  ${QRELS_PATH:+--qrels_path "${QRELS_PATH}"} \
  ${OUTPUT_PATH:+--output_path "${OUTPUT_PATH}"} \
  ${QUERIES_PATH:+--queries_path "${QUERIES_PATH}"} \
  ${OVERWRITE_FLAG}

