#!/bin/bash
export WANDB_MODE=disabled

# 1. 基础路径配置
model_base_dir="/data/share/project/shared_models"
output_root="/data/share/project/psjin/model"

# 2. 定义任务配置表 (格式: "任务名 语言 训练轮数")
task_table=(
    "ailastatutes en 1"
    "arguana en 3"
    "covidretrieval zh 5"
    "scidocs en 5"
)

# 3. 定义任务与具体数据路径的映射 (精准对应您提供的路径)
declare -A task_data_map
task_data_map["ailastatutes"]="/data/share/project/psjin/data/generated_data/ailastatutes/generation_results/hn_mine_data_uk_lex/en/ailastatutes/en-triplets.jsonl"
task_data_map["arguana"]="/data/share/project/psjin/data/generated_data/arguana/generation_results/hn_mine_data/en/arguana/en-triplets.jsonl"
task_data_map["covidretrieval"]="/data/share/project/psjin/data/generated_data/covidretrieval/generation_results/hn_mine_data_scored/zh/covidretrieval/zh-triplets_scored.jsonl"
task_data_map["scidocs"]="/data/share/project/psjin/data/generated_data/scidocs/generation_results/hn_mine_data/en/scidocs/en-triplets.jsonl"

# 4. 定义模型及其对应的 Prompt 模板
models=(
    # "Qwen3-Embedding-8B"
    "bge-multilingual-gemma2"
)

prompt_templates=(
    # 'Instruct: {}\nQuery: {}'
    '<instruct>{}\n<query>{}'
)

# --- 开始双重循环 ---
for i in "${!models[@]}"; do
    base_model_name="${models[$i]}"
    current_prompt="${prompt_templates[$i]}"
    base_model_path="${model_base_dir}/${base_model_name}"

    for entry in "${task_table[@]}"; do
        # 解析表格行
        read -r task lang epoch <<< "$entry"
        
        # 获取该任务对应的数据路径
        current_train_data=${task_data_map[$task]}
        
        # 动态设置输出目录，包含模型名和任务名
        current_output_dir="${output_root}/${base_model_name}/v1/${task}"

        echo "=========================================================="
        echo "模型: $base_model_name"
        echo "任务: $task"
        echo "数据: $current_train_data"
        echo "轮数: $epoch"
        echo "模板: $current_prompt"
        echo "=========================================================="

        # 检查训练数据文件是否存在
        if [ ! -f "$current_train_data" ]; then
            echo "错误: 找不到训练数据 $current_train_data，跳过此任务。"
            continue
        fi

        # 5. 配置参数
        num_gpus=8
        
        model_args="--model_name_or_path $base_model_path \
            --trust_remote_code True \
            --use_lora True \
            --lora_rank 32 \
            --lora_alpha 64 \
            --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
            --save_merged_lora_model True"

        data_args="--train_data $current_train_data \
            --train_group_size 8 \
            --query_max_len 512 \
            --passage_max_len 512 \
            --pad_to_multiple_of 8 \
            --query_instruction_format '$current_prompt' \
            --knowledge_distillation True \
            --same_dataset_within_batch True \
            --small_threshold 0 \
            --drop_threshold 0"

        training_args="--output_dir $current_output_dir \
            --learning_rate 5e-5 \
            --fp16 False \
            --bf16 True \
            --num_train_epochs $epoch \
            --per_device_train_batch_size 8 \
            --sub_batch_size 4 \
            --dataloader_drop_last True \
            --warmup_ratio 0.1 \
            --gradient_checkpointing \
            --deepspeed /data/share/project/public_envs/FlagEmbedding/examples/finetune/ds_stage1.json \
            --logging_steps 1 \
            --save_steps 1000 \
            --negatives_cross_device \
            --temperature 0.02 \
            --sentence_pooling_method last_token \
            --normalize_embeddings True \
            --kd_loss_type kl_div \
            --max_example_num_per_dataset 20000 \
            --overwrite_output_dir"

        # 6. 执行训练
        cmd="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node $num_gpus \
            -m FlagEmbedding.finetune.embedder.decoder_only.base \
            $model_args \
            $data_args \
            $training_args"

        echo "执行指令: $cmd"
        eval "$cmd"

        echo "任务 $task 完成。结果保存至: $current_output_dir"
        echo "----------------------------------------------------------"
    done
done

echo "所有模型及任务训练完成！"