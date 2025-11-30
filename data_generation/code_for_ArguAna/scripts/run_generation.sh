source /root/miniconda3/bin/activate /share/project/psjin/envs/psjin_embedder

task_type="arguana"
languages=("en")
num_examples=10
num_samples=50000

cache_dir="/share/project/psjin/dataset/.cache"
examples_dir="/share/project/psjin/dataset/examples"

save_dir="/share/project/psjin/dataset/data/$task_type/generation_results/prod"

mkdir -p $save_dir

for language in "${languages[@]}"; do
    echo "Generating for language: $language"

    cmd="python /share/project/psjin/code/code_for_ArguAna/data_synthesis/run_generation.py \
    --task_type $task_type \
    --save_dir $save_dir \
    --examples_dir $examples_dir \
    --num_examples $num_examples \
    --cache_dir $cache_dir \
    --language $language \
    --num_samples $num_samples \
    --model Qwen2-5-72B-Instruct \
    --model_type open-source \
    --port 8000 \
    --num_processes 8
    "

    echo $cmd
    eval $cmd 2>&1 | tee $save_dir/log_${language}_${task_type}.txt
done
