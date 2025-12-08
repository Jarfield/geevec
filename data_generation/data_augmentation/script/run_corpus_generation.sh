#!/usr/bin/env bash
# 快速调用语料改写脚本，运行前请替换路径。

set -euo pipefail

python "$(dirname "$0")/../code/run_corpus_generation.py" \
  --task_type miracl \
  --language en \
  --corpus_path /path/to/source.jsonl \
  --save_path /path/to/generated/synth.jsonl
