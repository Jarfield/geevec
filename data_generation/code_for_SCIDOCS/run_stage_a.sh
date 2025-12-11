#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_ROOT="${SAVE_ROOT:-${DATA_AUG_ORIGINAL_ROOT:-${SCRIPT_DIR}/../../original_data}}"
OUTPUT_PATH="${OUTPUT_PATH:-}" # optional explicit path
NUM_SAMPLES="${NUM_SAMPLES:--1}"
SEED="${SEED:-13}"
MIN_YEAR="${MIN_YEAR:-2015}"
FOS_FILTER="${FOS_FILTER:-Computer Science}"
MIN_TEXT_LEN="${MIN_TEXT_LEN:-30}"
OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

python "${SCRIPT_DIR}/build_scirepeval_scidocs_like.py" \
  --save_root "${SAVE_ROOT}" \
  ${OUTPUT_PATH:+--output_path "${OUTPUT_PATH}"} \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --min_year "${MIN_YEAR}" \
  --fos "${FOS_FILTER}" \
  --min_text_len "${MIN_TEXT_LEN}" \
  ${OVERWRITE_FLAG}
