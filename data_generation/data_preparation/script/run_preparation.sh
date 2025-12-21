#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../../.." && pwd )"

TASK="${TASK:-${1:-}}"
if [[ -z "${TASK}" ]]; then
  echo "Usage: TASK=<task_name> [LANGUAGE=en] [CORPUS_PATH=...] [QRELS_PATH=...] [OUTPUT_PATH=...] [OVERWRITE=1] $0"
  exit 1
fi

LANGUAGE="${LANGUAGE:-en}"
CORPUS_PATH="${CORPUS_PATH:-}"
QRELS_PATH="${QRELS_PATH:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

python "${PROJECT_ROOT}/data_generation/data_preparation/code/corpus_filter.py" \
  --task "${TASK}" \
  --language "${LANGUAGE}" \
  ${CORPUS_PATH:+--corpus_path "${CORPUS_PATH}"} \
  ${QRELS_PATH:+--qrels_path "${QRELS_PATH}"} \
  ${OUTPUT_PATH:+--output_path "${OUTPUT_PATH}"} \
  ${OVERWRITE_FLAG}
