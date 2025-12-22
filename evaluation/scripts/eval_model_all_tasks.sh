#!/bin/bash
# 1. 基础路径设置
eval_root="/data/share/project/psjin"
results_output_folder="$eval_root/result/evaluation"

# 2. 进入代码目录
cd "$eval_root/code/evaluation/code" || exit 1

tasks=(
  # "AILAStatutes"
  # "ArguAna"
  # "BelebeleRetrieval"
  "CovidRetrieval"
  # "SCIDOCS"
  # "SpartQA"
  # "TRECCOVID"
  # "WinoGrande"
  "StatcanDialogueDatasetRetrieval"
  # "TwitterHjerneRetrieval"
)

model_path="/data/share/project/shared_models/nvidia__llama-embed-nemotron-8b"

# 源文件路径列表
# required_files=(
#   "/data/share/project/psjin/model/Qwen3-8B/merged_model/1_Pooling/config.json"
#   "/data/share/project/psjin/model/Qwen3-8B/merged_model/config_sentence_transformers.json"
#   "/data/share/project/psjin/model/Qwen3-8B/merged_model/modules.json"
#   "/data/share/project/psjin/model/Qwen3-8B/merged_model/sentence_bert_config.json"
# )
required_files=(
  "/data/share/project/shared_models/nvidia__llama-embed-nemotron-8b/1_Pooling/config.json"
  "/data/share/project/shared_models/nvidia__llama-embed-nemotron-8b/config_sentence_transformers.json"
  "/data/share/project/shared_models/nvidia__llama-embed-nemotron-8b/modules.json"
  "/data/share/project/shared_models/nvidia__llama-embed-nemotron-8b/sentence_bert_config.json"

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
# copy_required_files

for task in "${tasks[@]}"; do
    out_dir="$results_output_folder/geevec-nemotron-8b-v0/$task"
    mkdir -p "$out_dir"
    
    # 修改点：
    # 1. 修正 prompt_template 匹配官方文档
    # 2. 增加 trust_remote_code
    # 3. 建议使用 BF16 代替 FP16
    cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
        --benchmark_name \"$task\" \
        --results_output_folder \"$out_dir\" \
        --model_name_or_path \"$model_path\" \
        --use_bf16 True \
        --use_fp16 False \
        --trust_remote_code True \
        --prompt_template 'Instruct: {}\nQuery: ' \
        --assert_prompts_exist True \
        --normalize_embeddings True \
        --batch_size 4 \
        --max_length 1024"
    
    echo "$cmd"
    eval "$cmd"
done

# 3. 修正 Summary 路径：指向刚刚跑完的模型文件夹
echo "Generating Summary..."
python "$eval_root/code/evaluation/code/summary.py" \
    "$results_output_folder/geevec-nemotron-8b-v0"