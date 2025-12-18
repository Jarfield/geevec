#!/usr/bin/env bash

# Simple wrapper for download_dataset.py
# Example:
#   bash download_dataset.sh --dataset wikitext --subset wikitext-2-raw-v1 --splits train validation test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/download_dataset.py" "$@"
