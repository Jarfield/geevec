#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://hf-mirror.com
export MTEB_CACHE="/data/share/project/shared_datasets/mteb"

# set model cache dir
export HF_HUB_CACHE="/data/share/project/shared_models"

# activate environment
# source /root/miniconda3/bin/activate /share/project/psjin/envs/psjin_embedder

eval_root="/data/share/project/psjin/code/evaluation/mmteb"
results_output_folder="$eval_root/results_output_folder"

cd "$eval_root/code" || exit 1

tasks=(
  # "AILAStatutes"
  # "ArguAna"
  # "BelebeleRetrieval"
  # "CovidRetrieval"
  # "SCIDOCS"
  # "SpartQA"
  # "TRECCOVID"
  # "WinoGrande"
  "StatcanDialogueDatasetRetrieval"
  "TwitterHjerneRetrieval"
)

model_paths=(
  "/data/share/project/psjin/model/geevec-qwen3-8b-v2-test-w-syn/merged_model"
  "/data/share/project/psjin/model/geevec-qwen3-8b-v1-test-w-syn/merged_model"
  "/data/share/project/psjin/model/geevec-qwen3-8b-v1-test-wo-syn/merged_model"
)

: << 'EOF'
# 先逐模型、逐任务跑 main.py
for model_path in "${model_paths[@]}"; do
    # 先取上一级目录，再取它的 basename，得到真正想要的 tag
    model_dir=$(dirname "$model_path")
    model_tag=$(basename "$model_dir")

    for task in "${tasks[@]}"; do
        out_dir="$results_output_folder/$model_tag/$task"
        mkdir -p "$out_dir"

        cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
            --benchmark_name \"$task\" \
            --results_output_folder \"$out_dir\" \
            --model_name_or_path \"$model_path\" \
            --use_fp16 True \
            --prompt_template 'Instruct:{}\nQuery:' \
            --assert_prompts_exist True \
            --normalize_embeddings True \
            --batch_size 8 \
            --max_length 512 \
            "

        echo "$cmd"
        eval "$cmd"
    done
done

EOF

for model_path in "${model_paths[@]}"; do
    model_dir=$(dirname "$model_path")
    model_tag=$(basename "$model_dir")

    python "$eval_root/code/summary.py" \
        "$results_output_folder/$model_tag"
done
