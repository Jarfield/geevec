#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"

export HF_HUB_CACHE="/data/share/project/shared_models/.cache"
export HF_DATASETS_CACHE="/data/share/project/shared_datasets/.cache"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_ROOT="${SAVE_ROOT:-/data/share/project/psjin/data/generated_data}"
INPUT_PATH="${INPUT_PATH:-${SAVE_ROOT}/scidocs/scirep_citation_train/en_scirep.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-}"

NUM_POS_NEIGHBORS="${NUM_POS_NEIGHBORS:-5}"
NUM_HARD_NEGATIVES="${NUM_HARD_NEGATIVES:-25}"
SEARCH_DEPTH="${SEARCH_DEPTH:-100}"
INDEX_FACTORY="${INDEX_FACTORY:-FlatIP}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# NEW: filter short texts (e.g., query titles) out of FAISS index to keep pos/neg as abstracts
MIN_DOC_LEN="${MIN_DOC_LEN:-200}"

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
  --min_doc_len "${MIN_DOC_LEN}" \
  ${OVERWRITE_FLAG}
