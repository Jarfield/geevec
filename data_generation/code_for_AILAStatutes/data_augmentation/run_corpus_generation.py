"""
run_corpus_generation.py

Entry script for synthesizing a new statute corpus for AILAStatutes.

Usage example (from project root):

    python data_augmentation/gen_corpus/run_corpus_generation.py \
        --task_type ailastatutes \
        --language en \
        --save_path /path/to/uklex_ailastatutes_synth_v1.jsonl \
        --cache_dir /path/to/.cache \
        --model Qwen2-5-72B-Instruct \
        --model_type open-source \
        --port 8000 \
        --num_variants_per_seed 3 \
        --num_threads 4

The script:
1. Loads the (all_corpus, relevant_corpus) via CorpusGenerator.run(...)
2. Chooses relevant_corpus as seeds (fallback to all_corpus if empty)
3. Calls StatuteSynthesisGenerator to generate synthetic statutes
4. Writes them into a JSONL corpus at save_path
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import List, Dict, Any

from tqdm import tqdm

# ---- Make project root & this folder importable ----
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
for _p in (ROOT_DIR, THIS_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from constant import TaskType, Language  # type: ignore
from corpus_generator import CorpusGenerator  # type: ignore

from statute_synthesis_generator import StatuteSynthesisGenerator  # type: ignore

DEFAULT_SYNTH_CORPUS_ROOT = "/data/share/project/psjin/data/generated_data/ailastatutes/generation_results/generated_corpus"

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_type",
        type=str,
        default="ailastatutes",
        choices=[t.name for t in TaskType],
        help="Task type. For this script it should be 'ailastatutes'.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=[l.name for l in Language],
        help="Language (ISO 639-1 code in enum name form). Default: en",
    )
    parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help=(
        "Output JSONL file path for the synthetic corpus. "
        "If not provided, a default file will be created under "
        f"{DEFAULT_SYNTH_CORPUS_ROOT}."
    ),
)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache dir for loading original corpus (datasets cache).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2-5-72B-Instruct",
        help="LLM model name to use for synthesis.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="open-source",
        help="Model type, e.g. 'open-source'.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for vLLM / OpenAI-compatible server.",
    )
    parser.add_argument(
        "--num_variants_per_seed",
        type=int,
        default=1,
        help="How many synthetic statutes to generate per seed statute.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads (ThreadPoolExecutor workers) for synthesis.",
    )
    parser.add_argument(
        "--num_seed_samples",
        type=int,
        default=-1,
        help=(
            "How many seed statutes to use. -1 means use all available "
            "seed statutes."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing save_path.",
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    task_type = args.task_type
    language = args.language

    # 1) 如果用户没显式给 --save_path，就用默认目录 + 自动文件名
    if args.save_path is None:
        os.makedirs(DEFAULT_SYNTH_CORPUS_ROOT, exist_ok=True)
        file_name = f"{language}_synth_corpus.jsonl"
        save_path = os.path.join(DEFAULT_SYNTH_CORPUS_ROOT, file_name)
        print(f"[INFO] No --save_path provided, use default: {save_path}")
    else:
        save_path = args.save_path

    # 2) 正常的 overwrite 检查
    if os.path.exists(save_path) and not args.overwrite:
        print(f"[WARN] save_path already exists: {save_path}")
        print("       Use --overwrite to overwrite it.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Load original corpus (all_corpus, relevant_corpus)
    corpus_generator = CorpusGenerator(cache_dir=args.cache_dir)

    print("[INFO] Loading original corpus...")
    all_corpus, relevant_corpus = corpus_generator.run(
        language=language,
        num_samples=args.num_seed_samples,
    )

    if len(relevant_corpus) > 0:
        seed_docs: List[Dict[str, Any]] = relevant_corpus
        print(f"[INFO] Using relevant_corpus as seeds: {len(seed_docs)} docs.")
    else:
        seed_docs = all_corpus
        print(
            "[INFO] No relevant_corpus available, fallback to all_corpus "
            f"as seeds: {len(seed_docs)} docs."
        )

    if len(seed_docs) == 0:
        print("[WARN] No seed docs found. Nothing to do.")
        return

    # 2. Initialize synthesis generator
    synth_gen = StatuteSynthesisGenerator(
        model=args.model,
        model_type=args.model_type,
        port=args.port,
    )

    print("[INFO] Start synthesizing new statutes...")
    synthetic_docs = synth_gen.run(
        seed_docs=seed_docs,
        task_type=task_type,
        language=language,
        num_variants_per_seed=args.num_variants_per_seed,
        thread_count=args.num_threads,
        tqdm_desc="Synthesizing statutes",
        # you can pass extra decoding kwargs here if needed:
        # temperature=0.9,
        # top_p=0.95,
    )

    print(f"[INFO] Generated {len(synthetic_docs)} synthetic statutes.")
    print(f"[INFO] Writing to {save_path} ...")

    with open(save_path, "w", encoding="utf-8") as f:
        for doc in synthetic_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    args = get_args()
    main(args)
