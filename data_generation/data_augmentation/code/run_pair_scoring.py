import argparse
import json
import os
import sys
from typing import List

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_generation.shared.constants import Language, TaskType
from query_doc_pair_scorer import QueryDocPairScorer


def load_data(input_path: str, num_samples: int = -1) -> List[dict]:
    data: List[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
            if 0 < num_samples == len(data):
                break
    return data


def save_data(data: List[dict], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="Score query-doc pairs to filter mined negatives.")
    parser.add_argument("--task", "--task_type", dest="task_type", type=str, required=True, choices=[t.name for t in TaskType])
    parser.add_argument("--language", type=str, default="en", choices=[l.name for l in Language])
    parser.add_argument("--input_path", type=str, required=True, help="Input JSONL with query/pos/neg fields.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save scored results.")
    parser.add_argument("--model", type=str, default="Qwen2-5-72B-Instruct")
    parser.add_argument("--model_type", type=str, default="open-source")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_processes", type=int, default=1, help="Thread count for scoring.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Subset size for quick experiments.")
    parser.add_argument("--pos_threshold", type=float, default=4.0)
    parser.add_argument("--neg_threshold", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    return parser.parse_args()


def main():
    args = get_args()

    if os.path.exists(args.output_path) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_path}. Use --overwrite to replace.")

    data = load_data(args.input_path, num_samples=args.num_samples)

    scorer = QueryDocPairScorer(
        model=args.model,
        model_type=args.model_type,
        port=args.port,
        cache_dir=args.cache_dir,
    )

    results = scorer.run(
        data,
        task_type=args.task_type,
        language=args.language,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        thread_count=args.num_processes,
    )

    save_data(results, args.output_path)
    print(f"Saved scored pairs to {args.output_path}")


if __name__ == "__main__":
    main()
