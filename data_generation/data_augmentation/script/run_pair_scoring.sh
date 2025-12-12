#!/bin/bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/share/project/psjin}"
CODE_DIR="$REPO_ROOT/code/data_generation/data_augmentation/code"
OVERWRITE="true"
TASK_TYPE="${TASK_TYPE:-scidocs}"
LANGUAGE="${LANGUAGE:-en}"
MODEL_NAME="${MODEL_NAME:-Qwen2-5-72B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-open-source}"
PORT="${PORT:-8000}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
POS_THRESHOLD="${POS_THRESHOLD:-4.0}"
NEG_THRESHOLD="${NEG_THRESHOLD:-2.0}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data/.cache}"  # optional LLM cache for scoring

# 输入/输出路径需根据实际数据目录填写
INPUT_PATH="${INPUT_PATH:-/data/share/project/psjin/data/generated_data/scidocs/scirep_citation_train/en_scirep.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/share/project/psjin/data/generated_data/scidocs/scirep_citation_train_scored/en_scirep.jsonl}"

python "$CODE_DIR/run_pair_scoring.py" \
  --task_type "$TASK_TYPE" \
  --language "$LANGUAGE" \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --model "$MODEL_NAME" \
  --model_type "$MODEL_TYPE" \
  --port "$PORT" \
  --num_processes "$NUM_PROCESSES" \
  --pos_threshold "$POS_THRESHOLD" \
  --neg_threshold "$NEG_THRESHOLD" \
  --num_samples "$NUM_SAMPLES" \
  ${CACHE_DIR:+--cache_dir "$CACHE_DIR"} \
  ${OVERWRITE:+--overwrite}

