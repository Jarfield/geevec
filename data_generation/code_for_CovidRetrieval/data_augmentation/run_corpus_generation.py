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

from constant import TaskType, Language, Task  # type: ignore
from corpus_generator import CorpusGenerator  # type: ignore

from covid_synthesis_generator import CovidDocSynthesisGenerator

DEFAULT_SYNTH_CORPUS_ROOT = "/data/share/project/psjin/data/generated_data/covidretrieval/generation_results/generated_corpus"

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_type",
        type=str,
        default="ailastatutes",
        choices=[t.name for t in TaskType],
        help="Task type. For this script it is usually 'ailastatutes' or 'covidretrieval'.",
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
        help="How many synthetic statutes/docs to generate per seed.",
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

    # ================== 关键改动 1：构造 CorpusGenerator 参数（含 score 过滤） ==================
    # corpus_kwargs = {"cache_dir": args.cache_dir}

    # # 只在 zh 的 CovidRetrieval 上启用 score 过滤
    # if task_type == "covidretrieval" and language == "zh":
    #     score_path = (
    #         "/data/share/project/psjin/data/data_cleaning/score/"
    #         "covidretrieval/generation_results/test/zh/covidretrieval/zh-scores.jsonl"
    #     )
    #     print(f"[INFO] 使用 score 过滤：{score_path}，阈值 >= 3")
    #     corpus_kwargs.update(
    #         score_path=score_path,
    #         score_threshold=3,
    #         score_id_key="docid",
    #         score_score_key="score",
    #     )
    
    # else:
    #     print("[INFO] 未配置 score 过滤，使用原始 CorpusGenerator 行为。")

    # corpus_generator = CorpusGenerator(**corpus_kwargs)
    corpus_generator = CorpusGenerator(cache_dir=args.cache_dir)

    # =======================================================================

    # 3) 加载 seed corpus
    print("[INFO] Loading original corpus as seeds...")

    if task_type == "covidretrieval" and language == "zh":
        # CovidRetrieval: CorpusGenerator.run 返回一个已过滤好的列表
        seed_docs: List[Dict[str, Any]] = corpus_generator.run(
            language=language,
            num_samples=args.num_seed_samples,
        )
        print(f"[INFO] Using CovidRetrieval (zh) score-filtered docs as seeds: {len(seed_docs)} docs.")
    else:
        all_corpus, relevant_corpus = corpus_generator.run(
            language=language,
            num_samples=args.num_seed_samples,
        )
        if len(relevant_corpus) > 0:
            seed_docs = relevant_corpus
            print(f"[INFO] Using relevant_corpus as seeds: {len(seed_docs)} docs.")
        else:
            seed_docs = all_corpus
            print(
                "[INFO] No relevant_corpus available, fallback to all_corpus "
                f"as seeds: {len(seed_docs)} docs."
            )

    if len(seed_docs) == 0:
        print("[WARN] No seed docs found after filtering. Nothing to do.")
        return

    # 4. Initialize synthesis generator
    synth_gen = CovidDocSynthesisGenerator(
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
        # you can pass extra decoding kwargs here if needed:
        # temperature=0.9,
        # top_p=0.95,
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