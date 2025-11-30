#!/bin/bash

task_type="covidretrieval"
languages=("zh")
num_examples=5
num_samples=100000

cache_dir="/data/share/project/psjin/data/.cache"
examples_dir="/data/share/project/psjin/data/examples/data_cleaning/covidretrieval"

# ✅ 已增广数据所在目录
input_root="/data/share/project/psjin/data/generated_data/${task_type}/generation_results/prod_augmentation"

# ✅ 打分结果与筛选后语料的保存目录
save_dir="/data/share/project/psjin/data/data_cleaning/score/${task_type}/generation_results/augmented_and_scored"

mkdir -p "$save_dir"

for language in "${languages[@]}"; do
    echo "Scoring augmented corpus for language: $language"

    # 如果你的增广结果是单文件，比如 zh-synth_corpus.jsonl，就改成具体文件名：
    # input_path="${input_root}/${language}/${task_type}/${language}_synth_corpus.jsonl"
    # 目前按“目录下所有 jsonl/arrow/parquet 都加载”的方式写：
    input_path="${input_root}/${language}/${task_type}"

    cmd="python /data/share/project/psjin/code/data_cleaning/code_for_CovidRetrieval/data_synthesis/run_generation.py \
        --task_type ${task_type} \
        --language ${language} \
        --input_path ${input_path} \
        --save_dir ${save_dir} \
        --examples_dir ${examples_dir} \
        --num_examples ${num_examples} \
        --cache_dir ${cache_dir} \
        --num_samples ${num_samples} \
        --model Qwen2-5-72B-Instruct \
        --model_type open-source \
        --port 8000 \
        --num_processes 8 \
        --min_score_to_keep 4 \
        --overwrite"

    echo "$cmd"
    eval "$cmd" 2>&1 | tee "${save_dir}/log_${language}_${task_type}.txt"
done
