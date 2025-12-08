#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root and move into the code directory so imports work regardless of invocation location.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CODE_DIR="${REPO_ROOT}/data_generation/data_augmentation/code"
cd "${CODE_DIR}"

# Core toggles (override via environment variables before running this script).
TASK_TYPE="${TASK_TYPE:-covidretrieval}"
LANGUAGE="${LANGUAGE:-zh}"

NUM_VARIANTS_PER_SEED=${NUM_VARIANTS_PER_SEED:-1}
NUM_THREADS=${NUM_THREADS:-8}
NUM_SEED_SAMPLES=${NUM_SEED_SAMPLES:--1}

CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/.cache}"
CORPUS_PATH="${CORPUS_PATH:-}"  # optional override
QRELS_PATH="${QRELS_PATH:-}"    # optional override

MODEL_NAME="${MODEL_NAME:-Qwen2-5-72B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-open-source}"
PORT="${PORT:-8000}"
OVERWRITE=${OVERWRITE:-0}

# Where to save the synthesized corpus.
SAVE_ROOT="${SAVE_ROOT:-${DATA_AUG_GENERATED_ROOT:-/data/share/project/psjin/data/generated_data}}"
SAVE_DIR="${SAVE_ROOT}/${TASK_TYPE}/generation_results/generated_corpus"
mkdir -p "${SAVE_DIR}"
SAVE_PATH="${SAVE_DIR}/${LANGUAGE}_synth_corpus.jsonl"

extra_args=()
if [ -n "${CORPUS_PATH}" ]; then
    extra_args+=("--corpus_path" "${CORPUS_PATH}")
fi
if [ -n "${QRELS_PATH}" ]; then
    extra_args+=("--qrels_path" "${QRELS_PATH}")
fi
if [ "${OVERWRITE}" -eq 1 ]; then
    extra_args+=("--overwrite")
fi

cmd=(
    python run_corpus_generation.py
    --task_type "${TASK_TYPE}"
    --language "${LANGUAGE}"
    --save_path "${SAVE_PATH}"
    --cache_dir "${CACHE_DIR}"
    --model "${MODEL_NAME}"
    --model_type "${MODEL_TYPE}"
    --port "${PORT}"
    --num_variants_per_seed "${NUM_VARIANTS_PER_SEED}"
    --num_threads "${NUM_THREADS}"
    --num_seed_samples "${NUM_SEED_SAMPLES}"
    "${extra_args[@]}"
)

echo "Running: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "${SAVE_DIR}/log_${LANGUAGE}_${TASK_TYPE}.txt"
