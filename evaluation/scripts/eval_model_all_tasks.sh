#!/bin/bash
# 1. 基础路径设置
eval_root="/data/share/project/psjin"
results_output_folder="$eval_root/result/evaluation"
model_dir="/data/share/project/shared_models"

# 2. 进入代码目录
cd "$eval_root/code/evaluation/code" || exit 1

tasks=(
  "AILAStatutes"
  "ArguAna"
  # # "BelebeleRetrieval"
  "CovidRetrieval"
  "SCIDOCS"
  # "SpartQA"
  # "TRECCOVID"
  # "WinoGrande"
  # "StatcanDialogueDatasetRetrieval"
  # "TwitterHjerneRetrieval"
)

# 定义模型列表
models=(
  "Qwen3-Embedding-8B"
  "bge-multilingual-gemma2"
)

# 定义与模型一一对应的 Prompt 模板
# 索引 0 对应 Qwen3，索引 1 对应 BGE
prompt_templates=(
  'Instruct: {}\nQuery: '
  '<instruct>{}\n<query>'
)

# --- 开始循环 ---
# 使用下标遍历，以便同时获取模型名和对应的 Prompt
for i in "${!models[@]}"; do
    model="${models[$i]}"
    current_prompt="${prompt_templates[$i]}"
    model_path="$model_dir/$model"
    
    echo "========================================================="
    echo "正在评测模型: $model"
    echo "对应模板: $current_prompt"
    echo "========================================================="

    for task in "${tasks[@]}"; do
        out_dir="$results_output_folder/$model/$task"
        mkdir -p "$out_dir"

        echo "-------------------- 正在运行任务: $task --------------------"
        
        # 注意：这里使用了变量 $current_prompt
        cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
            --benchmark_name \"$task\" \
            --results_output_folder \"$out_dir\" \
            --model_name_or_path \"$model_path\" \
            --use_bf16 True \
            --use_fp16 False \
            --trust_remote_code True \
            --prompt_template \"$current_prompt\" \
            --assert_prompts_exist True \
            --normalize_embeddings True \
            --batch_size 4 \
            --max_length 512"

        echo "执行指令: $cmd"
        eval "$cmd"
    done 
done

for i in "${!models[@]}"; do
    model="${models[$i]}"
    echo "正在生成模型 $model 的汇总报告..."
    python "$eval_root/code/evaluation/code/summary.py" \
        "$results_output_folder/$model"
done  
echo "所有模型评测完成！"