# source /root/miniconda3/bin/activate /root/miniconda3/envs/vllm

python /data/share/project/psjin/code/data_generation/code_for_AILAStatutes/vllm_deploy/run_open_source_llm.py \
--model_path /data/share/project/shared_models/Qwen2-5-72B-Instruct \
--serve_name Qwen2-5-72B-Instruct \
--max_length 32768 \
--parallel_size 8 \
--gpu_memory_utilization 0.9