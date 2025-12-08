#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CODE_DIR="${REPO_ROOT}/data_generation/data_augmentation/code"
cd "${CODE_DIR}"

# Optional: activate a specific environment before running
if [[ -n "${CONDA_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_PREFIX:-$HOME/miniconda3}/bin/activate" "${CONDA_ENV}"
fi

PYTHON_BIN=${PYTHON_BIN:-python}
EXAMPLES_SAVE_DIR=${EXAMPLES_SAVE_DIR:-${DATA_AUG_GENERATED_ROOT:-${REPO_ROOT}/data_generation/generated_data}/ailastatutes/generation_results/examples}
"${PYTHON_BIN}" -m gen_examples.examples --save_dir "${EXAMPLES_SAVE_DIR}"
