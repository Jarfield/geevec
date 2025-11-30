#!/usr/bin/env bash
set -euo pipefail

########################
# 配置区
source /root/miniconda3/bin/activate /share/project/psjin/envs/psjin_embedder
########################

# 根目录
DATA_ROOT="/share/project/shared_datasets/UK-LEX"

INPUT_DIR="${DATA_ROOT}/uk-lex69.jsonl"

# 输出一个合并后的 corpus
OUTPUT_FILE="${DATA_ROOT}/uklex_all_ailastatutes_corpus.jsonl"


MIN_DESC_CHARS=200
MAX_PART_CHARS=3000

echo "Input dir : ${INPUT_DIR}"
echo "Output file: ${OUTPUT_FILE}"
echo "Min desc chars: ${MIN_DESC_CHARS}"
echo "Max part chars: ${MAX_PART_CHARS}"

python build_corpus.py \
  --input_file "${INPUT_DIR}" \
  --output_file "${OUTPUT_FILE}" \
  --min_desc_chars "${MIN_DESC_CHARS}" \
  --max_part_chars "${MAX_PART_CHARS}"
