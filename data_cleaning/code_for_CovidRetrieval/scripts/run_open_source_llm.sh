python /data/share/project/psjin/code/data_cleaning/code_for_CovidRetrieval/vllm_deploy/run_open_source_llm.py \
--model_path /share/project/shared_models/Qwen2-5-72B-Instruct \
--serve_name Qwen2-5-72B-Instruct \
--max_length 32768 \
--parallel_size 8 \
--gpu_memory_utilization 0.9