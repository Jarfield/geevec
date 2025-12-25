#!/usr/bin/env bash

set -euo pipefail

# --- 路径配置 ---
DEFAULT_DATA_DIR=${INPUT_DIR:-/data/share/project/psjin/data/generated_data}

# --- 模型与任务配置 ---
MODEL_NAME=${MODEL_NAME:-Qwen2-5-72B-Instruct}
MODEL_TYPE=${MODEL_TYPE:-open-source}
PORT=${PORT:-8000}

# 必须指定的任务参数 (请根据实际情况修改默认值)
TASK_TYPE=${TASK_TYPE:-} 
LANGUAGE=${LANGUAGE:-}

# --- 推理参数 ---
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
MAX_TOKENS=${MAX_TOKENS:-1024}
LIMIT=${LIMIT:--1}

# --- 逻辑开关 ---
# 如果你想保留原始字段，取消注释下面这行，并在 python 命令中加入 $PASSTHROUGH_FLAG
# PASSTHROUGH_FLAG="--passthrough"
DROP_SUSPICIOUS_FLAG="--drop_suspicious"

TASK_LANG_LIST=(
  # "ailastatutes-en"
  # "arguana-en"
  "covidretrieval-zh"
  "scidocs-en"
)

for tl in "${TASK_LANG_LIST[@]}"; do
  # 2. 截断逻辑是正确的
  t=${tl%-*}  # 从右向左删掉第一个 - 及其右边的内容 -> 得到 task
  l=${tl##*-} # 从左向右删掉最后一个 - 及其左边的内容 -> 得到 language
  TASK_TYPE="${t}"
  LANGUAGE="${l}"
  INPUT_PATH="${DEFAULT_DATA_DIR}/${t}/preparation/${l}_pair_filtered.jsonl"
  OUTPUT_DIR="${DEFAULT_DATA_DIR}/${t}/contamination_check"
  OUTPUT_PATH="${OUTPUT_DIR}/${l}_rewritten_pairs.jsonl"
  mkdir -p "${OUTPUT_DIR}"
  
  echo "Starting query rewrite: ${TASK_TYPE} (${LANGUAGE}) using ${MODEL_NAME}..."
  
  python -m data_generation.contamination_check.code.rewrite_queries \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --model "${MODEL_NAME}" \
    --model_type "${MODEL_TYPE}" \
    --port "${PORT}" \
    --task_type "${TASK_TYPE}" \
    --language "${LANGUAGE}" \
    --temperature "${TEMPERATURE}" \
    --max_tokens "${MAX_TOKENS}" \
    --limit "${LIMIT}" \
    ${DROP_SUSPICIOUS_FLAG} \
    ${PASSTHROUGH_FLAG:-}
  
  echo "Rewrite complete. Output saved to: ${OUTPUT_PATH}"
done