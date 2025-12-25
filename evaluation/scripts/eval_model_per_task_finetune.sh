#!/bin/bash
# 1. 基础路径设置
eval_root="/data/share/project/psjin"
results_output_folder="$eval_root/result/evaluation"
repo_model_dir="/data/share/project/shared_models"
model_dir="/data/share/project/psjin/model" # 基础模型路径

# 2. 进入代码目录
cd "$eval_root/code/evaluation/code" || exit 1

# 定义任务列表
tasks=(
  "AILAStatutes"
  "ArguAna"
  "CovidRetrieval"
  "SCIDOCS"
)

# 定义模型列表 (对应你路径中的文件夹名)
models=(
  "Qwen3-Embedding-8B"
  "bge-multilingual-gemma2"
)

# 定义 Prompt 模板
prompt_templates=(
  'Instruct: {}\nQuery: '
  '<instruct>{}\n<query>'
)

# 复制配置文件的函数
copy_required_files() {
  local src_model_path="$1"
  local target_model_path="$2"
  local files_to_copy=(
    "1_Pooling/config.json"
    "config_sentence_transformers.json"
    "modules.json"
    "sentence_bert_config.json"
  )

  echo "---------------------------------------------------------"
  echo "Syncing configs from: $src_model_path"
  echo "To target: $target_model_path"

  for rel_path in "${files_to_copy[@]}"; do
    src_file="${src_model_path}/${rel_path}"
    target_file="${target_model_path}/${rel_path}"

    if [[ -f "$src_file" ]]; then
      mkdir -p "$(dirname "$target_file")"
      cp "$src_file" "$target_file"
      echo "  [OK] Copied $rel_path"
    else
      echo "  [SKIP] Not found: $src_file"
    fi
  done
}

# --- 开始模型大循环 ---
for i in "${!models[@]}"; do
    current_model="${models[$i]}"
    current_prompt="${prompt_templates[$i]}"
    
    # 基础配置文件来源 (共享目录)
    config_source_path="${repo_model_dir}/${current_model}"

    echo "#########################################################"
    echo "模型阶段: $current_model"
    echo "#########################################################"

    for task in "${tasks[@]}"; do
        lower_task=$(echo "$task" | tr '[:upper:]' '[:lower:]')
        
        # 【关键修改】：指向 /v1 目录下的任务子文件夹
        # 路径格式：/data/share/project/psjin/model/模型名/v1/任务名
        eval_model_path="${model_dir}/${current_model}/v1/${lower_task}/merged_model"

        if [[ ! -d "$eval_model_path" ]]; then
            echo ">> 警告: 跳过任务 $task, 目录不存在: $eval_model_path"
            continue
        fi

        # 1. 同步配置文件到 /v1 对应的任务目录
        copy_required_files "$config_source_path" "$eval_model_path"

        # 2. 设置输出目录 (为了区分版本，建议在路径中加入 v1)
        out_dir="$results_output_folder/${current_model}/v1/$task"
        mkdir -p "$out_dir"

        # 3. 运行评测
        echo ">> Running Evaluation: $task"
        cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
            --benchmark_name \"$task\" \
            --results_output_folder \"$out_dir\" \
            --model_name_or_path \"$eval_model_path\" \
            --use_bf16 True \
            --use_fp16 False \
            --trust_remote_code True \
            --prompt_template \"$current_prompt\" \
            --assert_prompts_exist True \
            --normalize_embeddings True \
            --batch_size 4 \
            --max_length 512"
        
        eval "$cmd"
    done
done
for i in "${!models[@]}"; do
    current_model="${models[$i]}"
    # 4. 每个模型跑完所有任务后，生成该模型的 Summary
    echo ">> 生成模型 $current_model (v1) 的汇总报告..."
    python "$eval_root/code/evaluation/code/summary.py" \
        "$results_output_folder/${current_model}/v1"
done

echo "所有模型评测任务已完成！"