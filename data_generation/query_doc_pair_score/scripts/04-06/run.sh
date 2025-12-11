#!/bin/bash
# source /root/miniconda3/bin/activate /root/miniconda3/envs/embedder

task_type="treccovid"
languages=("en")
num_samples=2000

cache_dir="/data/share/project/tr/mmteb/code/datasets/.cache"
examples_dir="/data/share/project/tr/mmteb/code/datasets/teccovid_generation_results/treccovid/formatted/11-27/"

save_dir="/data/share/project/tr/mmteb/code/datasets/teccovid_generation_results/$task_type/11-31-score-generation/"

mkdir -p $save_dir

for language in "${languages[@]}"; do
    echo "Generating for language: $language"

    cmd="python /data/share/project/tr/mmteb/code/code_for_treccovid/query_doc_pair_score/run_generation.py \
    --task_type $task_type \
    --examples_dir $examples_dir \
    --save_dir $save_dir \
    --cache_dir $cache_dir \
    --language $language \
    --num_samples $num_samples \
    --model Qwen2-5-72B-Instruct \
    --model_type open-source \
    --port 8000 \
    --num_processes 32 "

    echo $cmd
    eval $cmd 2>&1 | tee $save_dir/log_${language}_${task_type}.txt
done
