#!/bin/bash

# 配置路径
CODE_ROOT="/data/share/project/psjin/code"
DATA_ROOT="/data/share/project/psjin/data/generated_data"
EMBEDDER_PATH="/data/share/project/shared_models/bge-multilingual-gemma2"
INDEX_CACHE_ROOT="/data/share/project/psjin/data/.cache/faiss_indices"

# 任务列表 (task language)
task_language=(
    # "ailastatutes en"
    "arguana en"
    "covidretrieval zh"
    "scidocs en"
)

for entry in "${task_language[@]}"; do
    read -r task lang <<< "$entry"
    
    echo "----------------------------------------------------------"
    echo "Processing Task: $task ($lang)"
    
    INPUT_FILE="${DATA_ROOT}/${task}/contamination_check/${lang}_rewritten_pairs.jsonl"
    FORMATTED_FILE="${DATA_ROOT}/${task}/contamination_check/${lang}_rewritten_pairs_formatted.jsonl"
    FINAL_OUTPUT_FILE="${DATA_ROOT}/${task}/contamination_check/${lang}_rewritten_pairs_final.jsonl"
    INDEX_DIR="${INDEX_CACHE_ROOT}/${task}_${lang}_index"

    # Step 1: 格式化 (Rewritten -> Query/Pos Pair)
    echo "[Step 1] Formatting pairs..."
    python ${CODE_ROOT}/data_generation/contamination_check/code/format_pairs.py \
        --input $INPUT_FILE \
        --output $FORMATTED_FILE

    # Step 2: 挖掘硬负例 (Hard Negative Mining)
    # 调用你提供的 mine.py
    echo "[Step 2] Mining hard negatives..."
    python ${CODE_ROOT}/data_generation/hn_mine/code/mine.py \
        --embedder_name_or_path "/data/share/project/shared_models/Qwen3-Embedding-8B"  \
        --embedder_model_class "decoder-only-base" \
        --pooling_method last_token \
        --cache_dir $HF_HUB_CACHE \
        --query_instruction_for_retrieval "" \
        --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
        --trust_remote_code False \
        --batch_size 128 \
        --embedder_query_max_length 512 \
        --embedder_passage_max_length 512 \
        --task_type "$task" \
        --language "$lang" \
        --input_file "$FORMATTED_FILE" \
        --output_file "$FINAL_OUTPUT_FILE" \
        --candidate_pool "" \
        --index_save_dir "$INDEX_CACHE_ROOT/{$task}_{$lang}/hn_mine_index" \
        --search_top_k 1000 \
        --negative_number 63 \
        --use_gpu_for_searching True

    echo "Task $task Done. Final file: $FINAL_OUTPUT_FILE"
done