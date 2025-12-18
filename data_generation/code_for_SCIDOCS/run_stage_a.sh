#!/usr/bin/env bash
set -euo pipefail

# ================= 环境变量配置 =================
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/share/project/shared_models/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/share/project/shared_datasets/.cache}"

# vLLM OpenAI-compatible endpoint
export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ================= 路径配置 =================
# 1. 根目录
SAVE_ROOT="${SAVE_ROOT:-/data/share/project/psjin/data/generated_data/scidocs/scirep_train}"

# 2. 这里的 Test Path 就是你指定的“锚点源” (en_original.jsonl)
# 脚本会从这里面读取 pos abstract 来总结宏观主题
TEST_ANCHOR_PATH="${TEST_ANCHOR_PATH:-/data/share/project/psjin/data/exported_original/scidocs/en/scidocs/original_pairs/en_original.jsonl}"

# 3. 输出文件定义
BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${SAVE_ROOT}/en_scirep_base.jsonl}"
FILTERED_OUTPUT_PATH="${FILTERED_OUTPUT_PATH:-${SAVE_ROOT}/en_scirep_base_filtered_macro.jsonl}"

# ================= 参数配置 =================

# --- Stage 1: Export 参数 ---
NUM_SAMPLES="${NUM_SAMPLES:--1}"     # -1 表示全量导出
SEED="${SEED:-13}"
MIN_ABS_LEN="${MIN_ABS_LEN:-200}"

# --- Stage 2: Filter 参数 (控制过滤程度) ---
TOPIC_SUMMARY_MODEL="${TOPIC_SUMMARY_MODEL:-Qwen2-5-72B-Instruct}"
TOPIC_SUMMARY_ENDPOINT="${TOPIC_SUMMARY_ENDPOINT:-${VLLM_ENDPOINT}}"

# [宽松度参数 1] 采样多少条测试集数据来总结？(越多越全)
TOPIC_SUMMARY_MAX_DOCS="${TOPIC_SUMMARY_MAX_DOCS:-1000}"

# [宽松度参数 2] 让 LLM 吐出多少个关键词？(越多越容易命中)
TOPIC_KEYWORDS_PER_CHUNK="${TOPIC_KEYWORDS_PER_CHUNK:-80}"

# [严格度参数] Query 至少命中几个词才保留？(越大越严)
MIN_ANCHOR_OVERLAP="${MIN_ANCHOR_OVERLAP:-1}"

# [分词参数] 忽略太短的词 (如 is, at)
ANCHOR_MIN_TOKEN_LEN="${ANCHOR_MIN_TOKEN_LEN:-3}"

# 覆写开关
OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-1}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

# ================= 执行流程 =================

# -------------------------
# 1) 运行导出脚本：生成 base JSONL (训练集)
# -------------------------
echo "[1/2] Exporting base JSONL (Train Set)..."
# python "${SCRIPT_DIR}/export_scirepeval_base.py" \
#   --save_root "${SAVE_ROOT}" \
#   --output_path "${BASE_OUTPUT_PATH}" \
#   --num_samples "${NUM_SAMPLES}" \
#   --seed "${SEED}" \
#   --min_abs_len "${MIN_ABS_LEN}" \
#   ${OVERWRITE_FLAG}

echo "    -> Base Train Data: ${BASE_OUTPUT_PATH}"

# -------------------------
# 2) 运行过滤脚本：Test Set 提取主题 -> 过滤 Train Set
# -------------------------
# 假设你的新 Python 脚本名为 stage_a_filter.py
PYTHON_FILTER_SCRIPT="${SCRIPT_DIR}/stage_a_filter.py"

echo "[2/2] Filtering by MACRO topics from Test Set..."
echo "    -> Anchor Source (Test): ${TEST_ANCHOR_PATH}"
echo "    -> Filtering Target:     ${BASE_OUTPUT_PATH}"
echo "    -> Output Destination:   ${FILTERED_OUTPUT_PATH}"

python "${PYTHON_FILTER_SCRIPT}" \
  --train_path "${BASE_OUTPUT_PATH}" \
  --test_path "${TEST_ANCHOR_PATH}" \
  --output_path "${FILTERED_OUTPUT_PATH}" \
  --model "${TOPIC_SUMMARY_MODEL}" \
  --endpoint "${TOPIC_SUMMARY_ENDPOINT}" \
  --max_test_samples "${TOPIC_SUMMARY_MAX_DOCS}" \
  --keywords_per_chunk "${TOPIC_KEYWORDS_PER_CHUNK}" \
  --min_overlap "${MIN_ANCHOR_OVERLAP}" \
  --min_token_len "${ANCHOR_MIN_TOKEN_LEN}" \
  ${OVERWRITE_FLAG}

echo "[DONE] Process finished."