#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://huggingface.co

# set model cache dir
export HF_HUB_CACHE="/share/project/shared_models"

# activate environment
source /root/anaconda3/bin/activate /share/project/jianlv/envs/jianlv_mteb

eval_root="/share/project/jianlv/Embedder-SOTA/evaluation/mmteb"
results_output_folder="$eval_root/results_output_folder"

# model to evaluate
model_path="BAAI/bge-multilingual-gemma2"

# download data only
cd $eval_root/code

cmd="python main_all_tasks.py \
    --benchmark_name 'MTEB(Multilingual, v2)' \
    --only_download_data True \
    --results_output_folder $results_output_folder \
    --model_name_or_path $model_path \
    --use_fp16 True \
    --prompt_template '<instruct>{}\n<query>' \
    --assert_prompts_exist True \
    --normalize_embeddings True \
    --batch_size 32 \
    --max_length 512 \
    "

echo $cmd
eval $cmd
