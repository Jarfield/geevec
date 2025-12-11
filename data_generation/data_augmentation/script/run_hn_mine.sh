#!/bin/bash
REPO_ROOT="${REPO_ROOT:-/data/share/project/psjin}"

TASK_TYPE="${TASK_TYPE:-scidocs}"

DATA_ROOT="$REPO_ROOT/data"
DATA_DIR="$DATA_ROOT/exported_original/$TASK_TYPE"
CORPUS_DIR=""

SAVE_DIR="$DATA_ROOT/exported_original/$TASK_TYPE"
INDEX_SAVE_DIR="$DATA_ROOT/exported_original/$TASK_TYPE"

query_instruction_for_retrieval=""

languages=("en")

for language in "${languages[@]}"; do
    echo "Processing language: $language"
    INPUT_FILE="$DATA_DIR/$language/$TASK_TYPE/original_pairs/${language}_original.jsonl"
    echo "Input file: $INPUT_FILE"
    OUTPUT_FILE="$SAVE_DIR/$language/$TASK_TYPE/hn_mine_data/${language}_original.jsonl"
    echo "Output file: $OUTPUT_FILE"
    python $REPO_ROOT/code/data_generation/data_augmentation/code/mine.py \
    --embedder_name_or_path "/data/share/project/shared_models/Qwen3-Embedding-8B" \
    --embedder_model_class "decoder-only-base" \
    --pooling_method last_token \
    --cache_dir "/data/share/project/shared_models" \
    --query_instruction_for_retrieval "$query_instruction_for_retrieval" \
    --query_instruction_format_for_retrieval 'Instruct: {}\nQuery: {}' \
    --trust_remote_code False \
    --batch_size 128 \
    --embedder_query_max_length 512 \
    --embedder_passage_max_length 512 \
    --task_type "$TASK_TYPE" \
    --language "$language" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --candidate_pool "$CORPUS_DIR" \
    --index_save_dir "$INDEX_SAVE_DIR/$language/$TASK_TYPE/hn_mine_index" \
    --search_top_k 1000 \
    --negative_number 63 \
    --use_gpu_for_searching True

    echo $cmd
    eval $cmd
    echo "Finished processing language: $language"
done
