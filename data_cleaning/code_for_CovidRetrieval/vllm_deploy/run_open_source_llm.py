"""
python run_open_source_llm.py \
--model_path /share/chaofan/models/Qwen2.5-72B-Instruct \
--serve_name Qwen2-5-72B-Instruct \
--max_length 32768 \
--parallel_size 8 \
--gpu_memory_utilization 0.9
"""

import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/share/chaofan/models/Qwen2.5-72B-Instruct")
    parser.add_argument("--serve_name", type=str, default="Qwen2-5-72B-Instruct")
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = get_args()
    
    model_path = args.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_path}' not found")
    
    serve_name = args.serve_name
    max_length = args.max_length
    parallel_size = args.parallel_size
    gpu_memory_utilization = args.gpu_memory_utilization
    port = args.port
    
    cmd = f"vllm serve {model_path} --port {port} --served-model-name {serve_name} --max-model-len {max_length} --tensor-parallel-size {parallel_size} --gpu-memory-utilization {gpu_memory_utilization}"
    print(f"Start serving model '{model_path}' with name '{serve_name}' on port {port}...")
    os.system(cmd)


if __name__ == "__main__":
    main()
