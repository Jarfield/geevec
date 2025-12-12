#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com

export HF_HUB_CACHE="/data/share/project/shared_models/.cache"
export HF_DATASETS_CACHE="/data/share/project/shared_datasets/.cache"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_ROOT="${SAVE_ROOT:-/data/share/project/psjin/data/generated_data}"
OUTPUT_PATH="${OUTPUT_PATH:-}" # optional explicit path
NUM_SAMPLES="${NUM_SAMPLES:--1}"
SEED="${SEED:-13}"
MIN_YEAR="${MIN_YEAR:-0}"
FOS_FILTER="${FOS_FILTER:-}"
MIN_TEXT_LEN="${MIN_TEXT_LEN:-200}"
POS_PER_QUERY="${POS_PER_QUERY:-5}"
NEG_PER_QUERY="${NEG_PER_QUERY:-10}"

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
  --pos_per_query "${POS_PER_QUERY}" \
  --neg_per_query "${NEG_PER_QUERY}" \
  --mteb_qrels_dir "/data/share/project/shared_models/.cache/datasets--mteb--scidocs/snapshots/955fa095d8dfece60ea5b5d8a1377a6e8b6c8b93/qrels" \
  ${OVERWRITE_FLAG}