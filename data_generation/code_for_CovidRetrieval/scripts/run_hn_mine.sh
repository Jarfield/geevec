#!/bin/bash
task_type="covidretrieval"
data_dir="/data/share/project/psjin/data/generated_data/$task_type/generation_results/prod_augmentation"
corpus_dir="/data/share/project/shared_datasets/MMTEB/C-MTEB___covid_retrieval/default/0.0.0/1271c7809071a13532e05f25fb53511ffce77117"

save_dir="/data/share/project/psjin/data/generated_data/$task_type/generation_results/hn_mine_data_augmentation"
index_save_dir="/data/share/project/psjin/data/generated_data/$task_type/generation_results/hn_mine_index_augmentation"

query_instruction_for_retrieval="Given a question on COVID-19, retrieve news articles that answer the question."

languages=("zh")

for language in "${languages[@]}"; do
    echo "Processing language: $language"

    for round in {1..5}; do
        input_file="$data_dir/$language/$task_type/${language}-triplets_round${round}.jsonl"
        output_file="$save_dir/$language/$task_type/${language}-triplets_round${round}.jsonl"
        round_index_dir="$index_save_dir/Qwen3-Embedding-8B/$language/round${round}"

        echo "  Round $round: input  = $input_file"
        echo "  Round $round: output = $output_file"
        echo "  Round $round: index  = $round_index_dir"

        python /data/share/project/psjin/code/data_generation/code_for_CovidRetrieval/hn_mine/mine_v2_modified.py \
            --embedder_name_or_path /data/share/project/shared_models/Qwen3-Embedding-8B \
            --embedder_model_class decoder-only-base \
            --pooling_method last_token \
            --cache_dir /data/share/project/shared_models \
            --query_instruction_for_retrieval "$query_instruction_for_retrieval" \
            --query_instruction_format_for_retrieval 'Instruct: {}\nQuery:{}' \
            --trust_remote_code False \
            --batch_size 128 \
            --embedder_query_max_length 512 \
            --embedder_passage_max_length 512 \
            --input_file "$input_file" \
            --output_file "$output_file" \
            --candidate_pool "$corpus_dir/covid_retrieval-corpus.arrow" \
            --index_save_dir "$round_index_dir" \
            --search_top_k 1000 \
            --negative_number 63 \
            --use_gpu_for_searching True
    done

    echo "Finished processing language: $language"
done

