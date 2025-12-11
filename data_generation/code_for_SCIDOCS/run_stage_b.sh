#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_ROOT="${SAVE_ROOT:-${DATA_AUG_ORIGINAL_ROOT:-${SCRIPT_DIR}/../../original_data}}"
INPUT_PATH="${INPUT_PATH:-${SAVE_ROOT}/scidocs/scirep_citation_train/en_scirep.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
NUM_POS_NEIGHBORS="${NUM_POS_NEIGHBORS:-2}"
NUM_HARD_NEGATIVES="${NUM_HARD_NEGATIVES:-2}"
SEARCH_DEPTH="${SEARCH_DEPTH:-50}"
INDEX_FACTORY="${INDEX_FACTORY:-FlatIP}"
BATCH_SIZE="${BATCH_SIZE:-32}"
OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

python "${SCRIPT_DIR}/add_scincl_neighbors.py" \
  --input_path "${INPUT_PATH}" \
  ${OUTPUT_PATH:+--output_path "${OUTPUT_PATH}"} \
  --num_pos_neighbors "${NUM_POS_NEIGHBORS}" \
  --num_hard_negatives "${NUM_HARD_NEGATIVES}" \
  --search_depth "${SEARCH_DEPTH}" \
  --index_factory "${INDEX_FACTORY}" \
  --batch_size "${BATCH_SIZE}" \
  ${OVERWRITE_FLAG}
