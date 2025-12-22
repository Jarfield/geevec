#!/usr/bin/env bash

set -euo pipefail

INPUT_PATH=${INPUT_PATH:-/data/share/project/psjin/data/generated_data/task/preparation/en_pair_filtered.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/data/share/project/psjin/data/generated_data/task/contamination_check}
OUTPUT_PATH="${OUTPUT_DIR}/rewrite_results.jsonl"
MODEL_NAME=${MODEL_NAME:-Qwen2-5-72B-Instruct}
MODEL_TYPE=${MODEL_TYPE:-open-source}
PORT=${PORT:-8000}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
MAX_TOKENS=${MAX_TOKENS:-256}
LIMIT=${LIMIT:--1}

mkdir -p "${OUTPUT_DIR}"

python -m data_generation.contamination_check.code.rewrite_queries \
  --input_path "${INPUT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --model "${MODEL_NAME}" \
  --model_type "${MODEL_TYPE}" \
  --port "${PORT}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" \
  --limit "${LIMIT}"
