#!/usr/bin/env bash

# Simple wrapper for download_model.py
# Example:
#   bash download_model.sh --repo-id Qwen/Qwen2.5-3B --revision main

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/download_model.py" "$@"
