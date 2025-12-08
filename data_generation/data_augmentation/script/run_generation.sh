#!/usr/bin/env bash
# 用于便捷地运行查询生成脚本，请根据需要调整参数。

set -euo pipefail

python "$(dirname "$0")/../code/run_generation.py" \
  --task_type covidretrieval \
  --language zh \
  --save_dir /path/to/save \
  --corpus_path /path/to/corpus.arrow \
  --qrels_path /path/to/qrels.arrow
