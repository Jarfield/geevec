#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root so the script works from anywhere.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CODE_DIR="${REPO_ROOT}/data_generation/data_augmentation/code"
cd "${CODE_DIR}"

# Core toggles. Override them via environment variables, e.g.
#   TASK_TYPE=scidocs LANGUAGES="en" ./script/run_generation.sh
TASK_TYPE="${TASK_TYPE:-covidretrieval}"
LANGUAGES=(${LANGUAGES:-zh})

NUM_EXAMPLES=${NUM_EXAMPLES:-10}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
NUM_VARIANTS_PER_DOC=${NUM_VARIANTS_PER_DOC:-1}
NUM_ROUNDS=${NUM_ROUNDS:-5}
NUM_PROCESSES=${NUM_PROCESSES:-8}

CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/.cache}"
EXAMPLES_DIR="${EXAMPLES_DIR:-${DATA_AUG_ROOT:-${REPO_ROOT}}/data_generation/examples/data_generation}"

CORPUS_PATH="${CORPUS_PATH:-}"
QRELS_PATH="${QRELS_PATH:-}"

SAVE_ROOT="${REPO_ROOT}/data/generated_data"
# 根据传入的模式设置 SAVE_DIR
if [ "$MODE" == "prod" ]; then
  SAVE_DIR="${SAVE_ROOT}/${TASK_TYPE}/generation_results/prod_augmentation"
elif [ "$MODE" == "test" ]; then
  SAVE_DIR="${SAVE_ROOT}/${TASK_TYPE}/generation_results/test_augmentation"
else
  echo "Invalid mode. Please use 'prod' or 'test'."
  exit 1
fi

# 创建目录
mkdir -p "${SAVE_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen2-5-72B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-open-source}"
PORT="${PORT:-8000}"
OVERWRITE=${OVERWRITE:-1}

for LANGUAGE in "${LANGUAGES[@]}"; do
    echo "Generating for language: ${LANGUAGE} (task: ${TASK_TYPE})"

    extra_args=()
    if [ "${OVERWRITE}" -eq 1 ]; then
        extra_args+=("--overwrite")
    fi

    if [ -n "${CORPUS_PATH}" ]; then
        extra_args+=("--corpus_path" "${CORPUS_PATH}")
    fi

    if [ -n "${QRELS_PATH}" ]; then
        extra_args+=("--qrels_path" "${QRELS_PATH}")
    fi

    cmd=(
        python run_generation.py
        --task_type "${TASK_TYPE}"
        --save_dir "${SAVE_DIR}"
        --examples_dir "${EXAMPLES_DIR}"
        --num_examples "${NUM_EXAMPLES}"
        --cache_dir "${CACHE_DIR}"
        --language "${LANGUAGE}"
        --num_samples "${NUM_SAMPLES}"
        --num_variants_per_doc "${NUM_VARIANTS_PER_DOC}"
        --model "${MODEL_NAME}"
        --model_type "${MODEL_TYPE}"
        --port "${PORT}"
        --num_processes "${NUM_PROCESSES}"
        --num_rounds "${NUM_ROUNDS}"
    )

    if [ ${#extra_args[@]} -gt 0 ]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "${cmd[@]}"
    "${cmd[@]}" 2>&1 | tee "${SAVE_DIR}/log_${LANGUAGE}_${TASK_TYPE}.txt"
done
