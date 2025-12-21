#!/usr/bin/env bash

# Backwards compatibility wrapper for run_augmentation.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "${SCRIPT_DIR}/run_augmentation.sh" "$@"
