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

# task-language=( \
#   "ailastatutes-en" \
#   "arguana-en" \
#   "covidretrieval-zh" \
#   "scidocs-en" \
# )
# items=( "corpus" "pair" )
# for tl in "${task-language[@]}"; do
#   for i in "${items[@]}"; do
    # t=${tl%-*}  # 从右向左删掉第一个 - 及其右边的内容 -> 得到 task
    # l=${tl##*-} # 从左向右删掉最后一个 - 及其左边的内容 -> 得到 language
    # TASK="${t}"
    # LANGUAGE="${l}"
    # ITEM="${i}"
    echo "Running item filter for task: ${TASK}, language: ${LANGUAGE}, item: ${ITEM}"
    python "${PROJECT_ROOT}/data_generation/data_preparation/code/item_filter.py" \
    --task "${TASK}" \
    --language "${LANGUAGE}" \
    --item "${ITEM}" \
    ${CORPUS_PATH:+--corpus_path "${CORPUS_PATH}"} \
    ${QRELS_PATH:+--qrels_path "${QRELS_PATH}"} \
    ${OUTPUT_PATH:+--output_path "${OUTPUT_PATH}"} \
    ${QUERIES_PATH:+--queries_path "${QUERIES_PATH}"} \
    ${OVERWRITE_FLAG}
#   done
# done


