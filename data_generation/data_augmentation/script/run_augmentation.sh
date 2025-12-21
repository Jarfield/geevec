#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root so the script works from anywhere.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CODE_DIR="${REPO_ROOT}/data_augmentation/code"
cd "${CODE_DIR}"

# Core toggles. Override them via environment variables, e.g.
#   TASK=ailastatutes LANGUAGES="en" ./script/run_augmentation.sh
TASK="${TASK:-covidretrieval}"
LANGUAGES=(${LANGUAGES:-zh})

NUM_EXAMPLES=${NUM_EXAMPLES:-10}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
NUM_VARIANTS_PER_DOC=${NUM_VARIANTS_PER_DOC:-1}
NUM_ROUNDS=${NUM_ROUNDS:-5}
NUM_PROCESSES=${NUM_PROCESSES:-8}

CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data/.cache}"
EXAMPLES_DIR="${EXAMPLES_DIR:-${DATA_AUG_ROOT:-${REPO_ROOT}}/data/examples/data_generation}"

# 从命令行参数获取模式，默认为 "prod"
MODE="${1:-prod}"

SAVE_ROOT="${REPO_ROOT}/data/generated_data"
# 根据传入的模式设置 SAVE_DIR
if [ "$MODE" == "prod" ]; then
  SAVE_DIR="${SAVE_ROOT}/${TASK}/generation_results/prod_augmentation"
elif [ "$MODE" == "test" ]; then
  SAVE_DIR="${SAVE_ROOT}/${TASK}/generation_results/test_augmentation"
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

# Ensure CORPUS_PATH and QRELS_PATH have defaults if not set
CORPUS_PATH="${CORPUS_PATH:-}"
QRELS_PATH="${QRELS_PATH:-}"

# Generate for each language
for LANGUAGE in "${LANGUAGES[@]}"; do
    echo "Generating for language: ${LANGUAGE} (task: ${TASK})"

    extra_args=()
    if [ "${OVERWRITE}" -eq 1 ]; then
        extra_args+=("--overwrite")
    fi

    # Only add CORPUS_PATH and QRELS_PATH if they are set
    if [ -n "${CORPUS_PATH}" ] && [ -f "${CORPUS_PATH}" ]; then
        extra_args+=("--corpus_path" "${CORPUS_PATH}")
    elif [ -n "${CORPUS_PATH}" ]; then
        echo "Warning: CORPUS_PATH is set but the file does not exist: ${CORPUS_PATH}"
    fi

    if [ -n "${QRELS_PATH}" ] && [ -f "${QRELS_PATH}" ]; then
        extra_args+=("--qrels_path" "${QRELS_PATH}")
    elif [ -n "${QRELS_PATH}" ]; then
        echo "Warning: QRELS_PATH is set but the file does not exist: ${QRELS_PATH}"
    fi

    cmd=(
        python run_augmentation.py
        --task "${TASK}"
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
    "${cmd[@]}" 2>&1 | tee "${SAVE_DIR}/log_${LANGUAGE}_${TASK}.txt"
done
