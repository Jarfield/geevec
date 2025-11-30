#!/bin/bash
source /root/miniconda3/bin/activate /root/miniconda3/envs/train_embedder

task_type="covidretrieval"
data_dir="psjin/dataset/data/$task_type/generation_results/prod"
corpus_dir="shared_datasets/MMTEB/C-MTEB___covid_retrieval/default/0.0.0/1271c7809071a13532e05f25fb53511ffce77117"

save_dir="psjin/dataset/data/$task_type/generation_results/hn_mine_data"
index_save_dir="psjin/dataset/data/$task_type/generation_results/hn_mine_index"

query_instruction_for_retrieval="Given a question on COVID-19, retrieve news articles that answer the question."

languages=("zh")

for language in "${languages[@]}"; do
    echo "Processing language: $language"

    python /share/project/psjin/code/Data_generation/code_for_CovidRetrieval/hn_mine/mine_v2_modified.py \
    --embedder_name_or_path /share/project/shared_models/Qwen3-Embedding-8B \
    --embedder_model_class decoder-only-base \
    --pooling_method last_token \
    --cache_dir /share/project/shared_models \
    --query_instruction_for_retrieval "$query_instruction_for_retrieval" \
    --query_instruction_format_for_retrieval 'Instruct: {}\nQuery:{}' \
    --trust_remote_code False \
    --batch_size 128 \
    --embedder_query_max_length 512 \
    --embedder_passage_max_length 512 \
    --input_file $data_dir/$language/$task_type/$language-triplets.jsonl \
    --output_file $save_dir/$language/$task_type/$language-triplets.jsonl \
    --candidate_pool $corpus_dir/covid_retrieval-corpus.arrow \
    --index_save_dir $index_save_dir/Qwen3-Embedding-8B/$language \
    --search_top_k 1000 \
    --negative_number 63 \
    --use_gpu_for_searching True 

    echo $cmd
    eval $cmd
    echo "Finished processing language: $language"
done
