#!/bin/bash

# set huggingface mirror
export HF_ENDPOINT=https://hf-mirror.com
export MTEB_CACHE="/data/share/project/shared_datasets/mteb"

# set model cache dir
export HF_HUB_CACHE="/data/share/project/shared_models"

# activate environment
eval_root="/data/share/project/psjin"
results_output_folder="$eval_root/result/evaluation"

cd "$eval_root/code/evaluation/code" || exit 1

tasks=(
  "AILAStatutes"
  # "ArguAna"
  # "BelebeleRetrieval"
  # "CovidRetrieval"
  # "SCIDOCS"
  # "SpartQA"
  # "TRECCOVID"
  # "WinoGrande"
  # "StatcanDialogueDatasetRetrieval"
  # "TwitterHjerneRetrieval"
)

model_path="/data/share/project/psjin/model/geevec-check/ailastatutes/merged_model"

# 源文件路径列表
required_files=(
  "/data/share/project/psjin/model/Qwen3-8B/merged_model/1_Pooling/config.json"
  "/data/share/project/psjin/model/Qwen3-8B/merged_model/config_sentence_transformers.json"
  "/data/share/project/psjin/model/Qwen3-8B/merged_model/modules.json"
  "/data/share/project/psjin/model/Qwen3-8B/merged_model/sentence_bert_config.json"
)

copy_required_files() {
  # 遍历每个目标路径
  for required_file in "${required_files[@]}"; do
    # 提取文件名
    file_name=$(basename "$required_file")
    # 如果文件是 config.json 等，确保目标路径下有 1_Pooling 目录
    if [[ "$file_name" == "config.json" ]]; then
      target_dir="${model_path}/1_Pooling"
  
      # 如果 1_Pooling 子目录不存在，则创建
      if [[ ! -d "$target_dir" ]]; then
        echo "创建目录 $target_dir"
        mkdir -p "$target_dir"
      fi
      target_path="${target_dir}/${file_name}"
    else
      target_path="${model_path}/${file_name}"
    fi
  
    # 复制文件到目标路径
    echo "复制文件 $required_file 到 $target_path"
    cp "$required_file" "$target_path"
  done
}
# 调用函数来复制文件
copy_required_files

for task in "${tasks[@]}"; do
    out_dir="$results_output_folder/geevec-check/$task"
    mkdir -p "$out_dir"
    cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
        --benchmark_name \"$task\" \
        --results_output_folder \"$out_dir\" \
        --model_name_or_path \"$model_path\" \
        --use_fp16 False \
        --use_bf16 True \
        --prompt_template 'Instruct: {}\nQuery: ' \
        --assert_prompts_exist True \
        --normalize_embeddings True \
        --batch_size 8 \
        --max_length 512 \
        "
    echo "$cmd"
    eval "$cmd"
done

python "$eval_root/code/evaluation/code/summary.py" \
    "$results_output_folder/geevec-check" \