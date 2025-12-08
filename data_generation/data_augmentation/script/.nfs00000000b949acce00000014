#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"  # keep relative paths stable

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH="${MODEL_PATH:-/share/project/shared_models/Qwen2-5-72B-Instruct}"
SERVE_NAME="${SERVE_NAME:-Qwen2-5-72B-Instruct}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
PARALLEL_SIZE="${PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DEPLOY_SCRIPT="${DEPLOY_SCRIPT:-${REPO_ROOT}/data_generation/code_for_AILAStatutes/vllm_deploy/run_open_source_llm.py}"

"${PYTHON_BIN}" "${DEPLOY_SCRIPT}" \
--model_path "${MODEL_PATH}" \
--serve_name "${SERVE_NAME}" \
--max_length "${MAX_LENGTH}" \
--parallel_size "${PARALLEL_SIZE}" \
--gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"