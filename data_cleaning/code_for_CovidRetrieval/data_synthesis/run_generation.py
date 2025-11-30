from __future__ import annotations

import os
import json
import time
import argparse
import random
import multiprocessing as mp
from typing import List, Optional

from constant import TaskType, Language
from corpus_generator import CorpusGenerator
from triplet_generator import TripletGenerator


def get_args():
    parser = argparse.ArgumentParser(
        description="Score augmented CovidRetrieval corpus with LLM and keep only high-quality docs."
    )
    parser.add_argument(
        '--task_type',
        type=str,
        required=False,
        default='covidretrieval',
        choices=[t.name for t in TaskType],
        help="Task type. For this script it should be 'covidretrieval'.",
    )
    parser.add_argument(
        '--language',
        type=str,
        default='zh',
        choices=[l.name for l in Language],
        help='Language to score. Default: zh (Simplified Chinese).',
    )
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help=(
            "Path to augmented corpus. "
            "Can be a single *.jsonl file or a directory containing multiple *.jsonl / *.arrow / *.parquet."
        ),
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Base directory to save scoring results and filtered corpus.',
    )
    parser.add_argument(
        '--examples_dir',
        type=str,
        default=None,
        help='Optional few-shot examples directory: {examples_dir}/{task_type}/{language}_sample_examples.json',
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=3,
        help='Number of few-shot examples used in the scoring prompt.',
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Cache directory (also used by CorpusGenerator if needed).',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen2-5-72B-Instruct',
        help='LLM model name for scoring.',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='open-source',
        help="Model type, e.g. 'open-source'.",
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for vLLM / OpenAI-compatible server.',
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=1,
        help='Number of worker threads used inside TripletGenerator.',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='If >0, randomly sample this many docs from the input corpus before scoring.',
    )
    parser.add_argument(
        '--min_score_to_keep',
        type=int,
        default=4,
        help='Only keep documents whose score is >= this value (default: 4).',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='If set, existing output files will be overwritten.',
    )
    return parser.parse_args()


def gen_scores(
    model: str,
    model_type: str,
    port: int,
    positives: List[dict],
    task_type: str,
    language: str,
    examples_pool: Optional[List[dict]] = None,
    num_examples: int = 3,
    tqdm_desc: str = "Scoring documents",
    thread_count: int = 1,
    gen_cache_dir: Optional[str] = None,
):
    """
    调用 TripletGenerator.run，对每篇文档打 1–5 分。
    """
    scorer = TripletGenerator(model, model_type, port, cache_dir=gen_cache_dir)
    scores = scorer.run(
        positives=positives,
        task_type=task_type,
        language=language,
        examples_pool=examples_pool,
        num_examples=num_examples,
        tqdm_desc=tqdm_desc,
        thread_count=thread_count,
    )
    return scores


def _get_output_paths(
    save_dir: str,
    task_type: str,
    language: str,
    min_score: int,
):
    base_dir = os.path.join(save_dir, language, task_type)
    os.makedirs(base_dir, exist_ok=True)

    scores_all_path = os.path.join(base_dir, f"{language}-scores_all.jsonl")
    filtered_path = os.path.join(base_dir, f"{language}-filtered_ge{min_score}.jsonl")
    return scores_all_path, filtered_path


def save_scores_and_filtered(
    scores: List[dict],
    save_dir: str,
    task_type: str,
    language: str,
    min_score: int = 4,
    overwrite: bool = True,
):
    """
    写两个文件：
      1) {language}-scores_all.jsonl    —— 所有文档的打分结果；
      2) {language}-filtered_ge{min_score}.jsonl —— 只保留 score >= min_score 的文档（含原字段 + score）。
    """
    scores_all_path, filtered_path = _get_output_paths(
        save_dir=save_dir,
        task_type=task_type,
        language=language,
        min_score=min_score,
    )

    if (not overwrite) and (os.path.exists(scores_all_path) or os.path.exists(filtered_path)):
        print(f"[WARN] Output already exists and --overwrite is not set. Skip saving.")
        return

    print(f"[INFO] Writing all scores to: {scores_all_path}")
    with open(scores_all_path, "w", encoding="utf-8") as f_all:
        for item in scores:
            f_all.write(json.dumps(item, ensure_ascii=False) + "\n")

    kept = [it for it in scores if int(it.get("score", 0)) >= min_score]
    print(f"[INFO] Kept {len(kept)} / {len(scores)} docs with score >= {min_score}")

    print(f"[INFO] Writing filtered docs to: {filtered_path}")
    with open(filtered_path, "w", encoding="utf-8") as f_f:
        for item in kept:
            f_f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(args):
    task_type = args.task_type
    language = args.language

    # 1. 加载 augmented corpus
    corpus_generator = CorpusGenerator(
        input_path=args.input_path,
        cache_dir=args.cache_dir,
        min_len=0,  # 如有需要可以设置最小长度过滤
    )
    positives = corpus_generator.run(
        language=language,
        num_samples=args.num_samples,
    )
    print(f"[INFO] Num docs to score: {len(positives)}")

    # 2. few-shot 示例（可选）
    examples_dir = args.examples_dir
    num_examples = args.num_examples
    if examples_dir is not None:
        examples_path = os.path.join(
            examples_dir, task_type, f"{language}_sample_examples.json"
        )
        try:
            with open(examples_path, "r", encoding="utf-8") as f:
                examples_pool = json.load(f)
            examples_pool = random.sample(
                examples_pool,
                min(30, len(examples_pool)),
            )
            print(f"[INFO] Loaded {len(examples_pool)} few-shot examples from {examples_path}")
        except Exception as e:
            print(f"[WARN] Error loading examples from {examples_path}: {e}")
            examples_pool = None
    else:
        examples_pool = None

    print("=================== Score augmented corpus ===================")
    print(f"Task Type: {task_type} | Language: {language}")
    start_time = time.time()

    num_threads = max(1, min(args.num_processes, int(mp.cpu_count() * 0.8)))
    scores = gen_scores(
        model=args.model,
        model_type=args.model_type,
        port=args.port,
        positives=positives,
        task_type=task_type,
        language=language,
        examples_pool=examples_pool,
        num_examples=num_examples,
        thread_count=num_threads,
        gen_cache_dir=os.path.join(args.save_dir, language, task_type, "gen_cache_dir"),
    )

    save_scores_and_filtered(
        scores=scores,
        save_dir=args.save_dir,
        task_type=task_type,
        language=language,
        min_score=args.min_score_to_keep,
        overwrite=args.overwrite,
    )

    end_time = time.time()
    print("====================================================")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("====================================================")
    print("DONE!")


if __name__ == "__main__":
    _args = get_args()
    main(_args)
