#!/bin/bash

# Use HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com

# Cache directories
export HF_HUB_CACHE="/share/project/shared_models"
export HF_DATASETS_CACHE="/share/project/shared_datasets/MMTEB"

# Activate environment
source /root/miniconda3/bin/activate /share/project/psjin/envs/psjin_embedder

eval_root="/share/project/shared_datasets/MMTEB"
results_output_folder="$eval_root/results_output_folder"
model_path="/share/project/shared_models/Qwen3-Embedding-0.6B"

# Download only the single task: CovidRetrieval
cmd="python /share/project/psjin/evaluation/mmteb/code/main.py \
    --benchmark_name CovidRetrieval \
    --only_download_data True \
    --results_output_folder $results_output_folder \
    --model_name_or_path $model_path \
    --use_fp16 True \
    --prompt_template 'Instruct: {} Query: {}' \
    --assert_prompts_exist False \
    --normalize_embeddings True \
    --batch_size 32 \
    --max_length 512 \
    "

echo "$cmd"
eval "$cmd"
