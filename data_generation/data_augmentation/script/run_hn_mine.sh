#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${REPO_ROOT}"  # keep relative paths stable

TASK_TYPE="${TASK_TYPE:-covidretrieval}"
LANGUAGES=(${LANGUAGES:-zh})
ROUNDS=${ROUNDS:-5}

DATA_ROOT="${DATA_AUG_GENERATED_ROOT:-${REPO_ROOT}/data_generation/generated_data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/${TASK_TYPE}/generation_results/prod_augmentation}"
SAVE_DIR="${SAVE_DIR:-${DATA_ROOT}/${TASK_TYPE}/generation_results/hn_mine_data_augmentation}"
INDEX_SAVE_DIR="${INDEX_SAVE_DIR:-${DATA_ROOT}/${TASK_TYPE}/generation_results/hn_mine_index_augmentation}"
CORPUS_DIR="${CORPUS_DIR:-${DATA_AUG_ROOT:-/data/share/project/shared_datasets}/MMTEB/C-MTEB___covid_retrieval/default/0.0.0/1271c7809071a13532e05f25fb53511ffce77117}"

QUERY_INSTRUCTION_FOR_RETRIEVAL="${QUERY_INSTRUCTION_FOR_RETRIEVAL:-Given a question on COVID-19, retrieve news articles that answer the question.}"
EMBEDDER_NAME_OR_PATH="${EMBEDDER_NAME_OR_PATH:-/data/share/project/shared_models/Qwen3-Embedding-8B}"
EMBEDDER_MODEL_CLASS="${EMBEDDER_MODEL_CLASS:-decoder-only-base}"
POOLING_METHOD="${POOLING_METHOD:-last_token}"
CACHE_DIR="${CACHE_DIR:-/data/share/project/shared_models}"

PYTHON_BIN=${PYTHON_BIN:-python}
MINE_SCRIPT="${MINE_SCRIPT:-${REPO_ROOT}/data_generation/code_for_CovidRetrieval/hn_mine/mine_v2_modified.py}"

for LANGUAGE in "${LANGUAGES[@]}"; do
    echo "Processing language: ${LANGUAGE}"

    for ((round=1; round<=ROUNDS; round++)); do
        input_file="${DATA_DIR}/${LANGUAGE}/${TASK_TYPE}/${LANGUAGE}-triplets_round${round}.jsonl"
        output_file="${SAVE_DIR}/${LANGUAGE}/${TASK_TYPE}/${LANGUAGE}-triplets_round${round}.jsonl"
        round_index_dir="${INDEX_SAVE_DIR}/Qwen3-Embedding-8B/${LANGUAGE}/round${round}"

        echo "  Round ${round}: input  = ${input_file}"
        echo "  Round ${round}: output = ${output_file}"
        echo "  Round ${round}: index  = ${round_index_dir}"

        "${PYTHON_BIN}" "${MINE_SCRIPT}" \
            --embedder_name_or_path "${EMBEDDER_NAME_OR_PATH}" \
            --embedder_model_class "${EMBEDDER_MODEL_CLASS}" \
            --pooling_method "${POOLING_METHOD}" \
            --cache_dir "${CACHE_DIR}" \
            --query_instruction_for_retrieval "${QUERY_INSTRUCTION_FOR_RETRIEVAL}" \
            --query_instruction_format_for_retrieval 'Instruct: {}\nQuery:{}' \
            --trust_remote_code False \
            --batch_size 128 \
            --embedder_query_max_length 512 \
            --embedder_passage_max_length 512 \
            --input_file "${input_file}" \
            --output_file "${output_file}" \
            --candidate_pool "${CORPUS_DIR}/covid_retrieval-corpus.arrow" \
            --index_save_dir "${round_index_dir}" \
            --search_top_k 1000 \
            --negative_number 63 \
            --use_gpu_for_searching True
    done

    echo "Finished processing language: ${LANGUAGE}"
done

