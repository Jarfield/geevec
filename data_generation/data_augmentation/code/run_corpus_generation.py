from __future__ import annotations

import os
import sys
import json
import argparse
from typing import List, Dict, Any

from tqdm import tqdm

# ---- Make project root & this folder importable ----
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
for _p in (PROJECT_ROOT, THIS_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from data_generation.shared.constants import Language, TaskType  # type: ignore
from data_generation.data_preparation.code.task_configs import DEFAULT_GENERATED_ROOT
from corpus_generator import CorpusGenerator  # type: ignore
from doc_synthesis_generator import DocSynthesisGenerator


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        "--task_type",
        dest="task_type",
        type=str,
        default="covidretrieval",
        choices=[t.name for t in TaskType],
        help="Task type to synthesise.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=[l.name for l in Language],
        help="Language (ISO 639-1 code in enum name form). Example: en / zh",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help=(
            "Output JSONL file path for the synthetic corpus. "
            "If not provided, a default file will be created under a generated-data root."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache dir for loading original corpus (datasets cache).",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help="Override corpus path instead of using task_configs.py default.",
    )
    parser.add_argument(
        "--qrels_path",
        type=str,
        default=None,
        help="Optional qrels path for filtering positives.",
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
        help="How many synthetic variants to generate per seed document.",
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
            "How many seed docs to use. -1 means use all available "
            "seed docs after filtering."
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

    if args.save_path is None:
        default_dir = os.path.join(DEFAULT_GENERATED_ROOT, task_type, "generation_results", "generated_corpus")
        os.makedirs(default_dir, exist_ok=True)
        file_name = f"{language}_synth_corpus.jsonl"
        save_path = os.path.join(default_dir, file_name)
        print(f"[INFO] No --save_path provided, use default: {save_path}")
    else:
        save_path = args.save_path

    if os.path.exists(save_path) and not args.overwrite:
        print(f"[WARN] save_path already exists: {save_path}")
        print("       Use --overwrite to overwrite it.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    corpus_generator = CorpusGenerator(cache_dir=args.cache_dir)

    print("[INFO] Loading original corpus as seeds...")
    seed_docs: List[Dict[str, Any]] = corpus_generator.run(
        task_type=task_type,
        language=language,
        corpus_path=args.corpus_path,
        qrels_path=args.qrels_path,
        num_samples=args.num_seed_samples,
    )
    print(f"[INFO] Using {len(seed_docs)} seed docs for synthesis.")

    if len(seed_docs) == 0:
        print("[WARN] No seed docs found after filtering. Nothing to do.")
        return

    synth_gen = DocSynthesisGenerator(
        model=args.model,
        model_type=args.model_type,
        port=args.port,
    )

    print("[INFO] Start synthesizing new docs...")
    synthetic_docs = synth_gen.run(
        seed_docs=seed_docs,
        task_type=task_type,
        language=language,
        num_variants_per_seed=args.num_variants_per_seed,
        thread_count=args.num_threads,
        tqdm_desc="Synthesizing docs",
    )

    print(f"[INFO] Generated {len(synthetic_docs)} synthetic docs.")
    print(f"[INFO] Writing to {save_path} ...")

    with open(save_path, "w", encoding="utf-8") as f:
        for doc in synthetic_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    args = get_args()
    main(args)
