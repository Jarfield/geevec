#!/bin/bash
set -euo pipefail

# 将原始元数据（corpus + queries + qrels）转换为训练三元组格式。
# 数据路径与字段名默认从 data_preparation/code/task_configs.py 读取，通常只需设置 TASK_TYPE/LANGUAGE。

TASK_TYPE=${TASK_TYPE:-scidocs}
LANGUAGE=${LANGUAGE:-en}

SAVE_ROOT=${SAVE_ROOT:-/data/share/project/psjin/data/exported_original/${TASK_TYPE}/${LANGUAGE}}
CORPUS_PATH=${CORPUS_PATH:-}
QUERIES_PATH=${QUERIES_PATH:-}
QRELS_PATH=${QRELS_PATH:-}
OUTPUT_PATH=${OUTPUT_PATH:-}

CORPUS_ID_KEY=${CORPUS_ID_KEY:-}
CORPUS_TEXT_KEY=${CORPUS_TEXT_KEY:-}
CORPUS_TITLE_KEY=${CORPUS_TITLE_KEY:-}
QUERY_ID_KEY=${QUERY_ID_KEY:-}
QUERY_TEXT_KEY=${QUERY_TEXT_KEY:-}
QRELS_QID_KEY=${QRELS_QID_KEY:-}
QRELS_PID_KEY=${QRELS_PID_KEY:-}
QRELS_SCORE_KEY=${QRELS_SCORE_KEY:-}

POSITIVE_SCORE=${POSITIVE_SCORE:-1}
MIN_LEN=${MIN_LEN:-}
MAX_QUERIES=${MAX_QUERIES:--1}

OVERWRITE=${OVERWRITE:-1}

CMD=(python /data/share/project/psjin/code/data_generation/data_augmentation/code/export_original_pairs.py --task_type "${TASK_TYPE}" --language "${LANGUAGE}" --positive_score "${POSITIVE_SCORE}" --max_queries "${MAX_QUERIES}")

[[ -n "${SAVE_ROOT}" ]] && CMD+=(--save_root "${SAVE_ROOT}")
[[ -n "${CORPUS_PATH}" ]] && CMD+=(--corpus_path "${CORPUS_PATH}")
[[ -n "${QUERIES_PATH}" ]] && CMD+=(--queries_path "${QUERIES_PATH}")
[[ -n "${QRELS_PATH}" ]] && CMD+=(--qrels_path "${QRELS_PATH}")
[[ -n "${OUTPUT_PATH}" ]] && CMD+=(--output_path "${OUTPUT_PATH}")

[[ -n "${CORPUS_ID_KEY}" ]] && CMD+=(--corpus_id_key "${CORPUS_ID_KEY}")
[[ -n "${CORPUS_TEXT_KEY}" ]] && CMD+=(--corpus_text_key "${CORPUS_TEXT_KEY}")
[[ -n "${CORPUS_TITLE_KEY}" ]] && CMD+=(--corpus_title_key "${CORPUS_TITLE_KEY}")
[[ -n "${QUERY_ID_KEY}" ]] && CMD+=(--query_id_key "${QUERY_ID_KEY}")
[[ -n "${QUERY_TEXT_KEY}" ]] && CMD+=(--query_text_key "${QUERY_TEXT_KEY}")
[[ -n "${QRELS_QID_KEY}" ]] && CMD+=(--qrels_qid_key "${QRELS_QID_KEY}")
[[ -n "${QRELS_PID_KEY}" ]] && CMD+=(--qrels_pid_key "${QRELS_PID_KEY}")
[[ -n "${QRELS_SCORE_KEY}" ]] && CMD+=(--qrels_score_key "${QRELS_SCORE_KEY}")
[[ -n "${MIN_LEN}" ]] && CMD+=(--min_len "${MIN_LEN}")

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
