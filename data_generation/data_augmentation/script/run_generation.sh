cd /data/share/project/psjin/code/data_generation/code_for_CovidRetrieval/data_augmentation
python run_corpus_generation.py \
  --task_type covidretrieval \
  --language zh \
  --cache_dir /data/share/project/psjin/data/.cache \
  --model Qwen2-5-72B-Instruct \
  --model_type open-source \
  --port 8000 \
  --num_variants_per_seed 50 \
  --num_threads 8 \
  --num_seed_samples -1 \
  --overwrite

task_type="covidretrieval"
languages=("zh")
num_examples=10
num_samples=10000
num_variants_per_doc=1
num_rounds=5

cache_dir="/data/share/project/psjin/data/.cache"
examples_dir="/data/share/project/psjin/data/examples/data_generation"

save_dir="/data/share/project/psjin/data/generated_data/$task_type/generation_results/prod_augmentation"

mkdir -p $save_dir
OVERWRITE=1

for language in "${languages[@]}"; do
    echo "Generating for language: $language"

    extra_args=""
    if [ "${OVERWRITE}" -eq 1 ]; then
        extra_args="--overwrite"
    fi

    cmd="python run_generation.py \
    --task_type $task_type \
    --save_dir $save_dir \
    --examples_dir $examples_dir \
    --num_examples $num_examples \
    --cache_dir $cache_dir \
    --language $language \
    --num_samples $num_samples \
    --num_variants_per_doc $num_variants_per_doc \
    --model Qwen2-5-72B-Instruct \
    --model_type open-source \
    --port 8000 \
    --num_processes 8 \
    --num_rounds $num_rounds \
    ${extra_args}
    "

    echo $cmd
    eval $cmd 2>&1 | tee $save_dir/log_${language}_${task_type}.txt
done
