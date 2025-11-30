import os
import json
import time
import gc
import torch
import argparse
import random
from hashlib import md5
import multiprocessing as mp
from typing import List, Optional

from constant import TaskType, Language
from corpus_generator import CorpusGenerator
from triplet_generator import TripletGenerator


def compute_md5(text: str):
    return md5(text.encode()).hexdigest()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_type',
        type=str,
        required=True,
        help='The task type to generate data for',
        choices=[t.name for t in TaskType]
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='The path to save the generated data'
    )
    parser.add_argument(
        '--examples_dir',
        type=str,
        default=None,
        help='The path to the examples directory. If not None, the examples will be used for few-shot generation.'
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=3,
        help='The number of examples to use for few-shot generation. Default: 3'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='The cache directory'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='zh',  # 对 covidretrieval 你现在主要跑 zh，就直接默认 zh
        help='The language to generate for. ISO 639-1 code. Default: zh',
        choices=[l.name for l in Language]
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='The number of examples to use for generation. Default: -1. Use all available examples.'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen2-5-72B-Instruct',
        help='The model to use for generation. Default: Qwen2-5-72B-Instruct'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='open-source',
        help='The type of model to use for generation. Default: open-source',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='The port for vllm.'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=1,
        help='The number of processes to use for generation. Default: 1'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Whether to overwrite the existing data.'
    )
    args = parser.parse_args()
    return args


def gen_triplets(
    model: str,
    model_type: str,
    port: int,
    positives: List[dict],
    task_type: str,
    language: str,
    examples_pool: Optional[List[dict]] = None,
    num_examples: int = 3,
    tqdm_desc: str = "Generating triplets",
    thread_count: int = 1,
    gen_cache_dir: Optional[str] = None,
):
    triplet_generator = TripletGenerator(model, model_type, port, cache_dir=gen_cache_dir)
    triplets = triplet_generator.run(
        positives=positives,
        task_type=task_type,
        language=language,
        examples_pool=examples_pool,
        num_examples=num_examples,
        tqdm_desc=tqdm_desc,
        thread_count=thread_count,
    )
    return triplets


def get_save_path(
    save_dir: str,
    task_type: str,
    language: str,
):
    save_dir = os.path.join(save_dir, language, task_type)
    file_name = f"{language}-triplets.jsonl"
    save_path = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_path


def save_triplets(
    triplets: list,
    save_dir: str,
    task_type: str,
    language: str,
):
    if len(triplets) == 0:
        print(f"No triplets to save: {task_type} | {language}")
        return

    save_path = get_save_path(save_dir, task_type, language)
    query_md5s = set()
    pos_md5s = set()
    old_triplets = []
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                triplet = json.loads(line)
                old_triplets.append(triplet)
                query_md5s.add(compute_md5(triplet['query']))
                pos_md5s.add(compute_md5(triplet['pos'][0]))

    new_count = 0
    with open(save_path, 'w', encoding='utf-8') as f:
        # 先写旧的
        for triplet in old_triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')

        # 再写新的，按 query+pos 去重
        for triplet in triplets:
            _query_md5 = compute_md5(triplet['query'])
            _pos_md5 = compute_md5(triplet['pos'][0])
            if _query_md5 in query_md5s or _pos_md5 in pos_md5s:
                continue
            query_md5s.add(_query_md5)
            pos_md5s.add(_pos_md5)
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
            new_count += 1

    print(
        f"Triplets saved to {save_path}，"
        f"原有样本 {len(old_triplets)} 条，新增加样本 {new_count} 条。"
    )


def main(args):
    model = args.model
    model_type = args.model_type
    port = args.port

    num_samples = args.num_samples

    task_type = args.task_type
    language = args.language

    save_dir = args.save_dir
    cache_dir = args.cache_dir
    num_processes = min(args.num_processes, int(mp.cpu_count() * 0.8))
    overwrite = args.overwrite

    save_path = get_save_path(save_dir, task_type, language)
    if num_samples > 0 and os.path.exists(save_path) and not overwrite:
        data = []
        with open(save_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        if len(data) >= num_samples * 0.8:
            print(f"Triplets already exist at {save_path}. Skipping generation.")
            return
        else:
            print(f"Triplets already exist at {save_path}. But samples is really small, continue generation.")
            # 考虑已经存在的样本被过滤，这里稍微放大一点目标数量
            num_samples = int((num_samples - len(data)) * 1.25)

    # ================== 这里是关键改动：启用 score 过滤 ==================
    corpus_kwargs = {"cache_dir": cache_dir}

    # 目前只对 covidretrieval + zh 启用 score 文件过滤
    if task_type == "covidretrieval" and language == "zh":
        score_path = "/share/project/psjin/data/data_cleaning/score/covidretrieval/generation_results/test/zh/covidretrieval/zh-scores.jsonl"
        print(f"[INFO] 使用 score 过滤：{score_path}，阈值 >= 3")
        corpus_kwargs.update(
            score_path=score_path,
            score_threshold=3,
            score_id_key="docid",
            score_score_key="score",
        )
    else:
        print("[INFO] 未配置 score 过滤，使用原始 CorpusGenerator 行为。")

    corpus_generator = CorpusGenerator(**corpus_kwargs)
    # ===============================================================

    examples_dir = args.examples_dir
    num_examples = args.num_examples
    if examples_dir is not None:
        examples_path = os.path.join(examples_dir, task_type, f"{language}_sample_examples.json")
        try:
            with open(examples_path, 'r', encoding='utf-8') as f:
                examples_pool = json.load(f)
                # few-shot 池中最多抽 30 条
                examples_pool = random.sample(examples_pool,
                                              min(30, len(examples_pool)))
        except Exception as e:
            print(f'Error loading examples from {examples_path}: {e}')
            examples_pool = None
    else:
        examples_pool = None

    # 这里拿到的 positives，已经在 CorpusGenerator 里按：
    # 1）相关性（qrels）过滤
    # 2）长度过滤
    # 3）score >= 3 过滤（如果配置了 score_path）
    positives = corpus_generator.run(
        language=language,
        num_samples=num_samples,
    )

    print("=================== Generate training data ===================")
    print(f'Task Type: {task_type} | Language: {language}')
    start_time = time.time()
    triplets = gen_triplets(
        model=model,
        model_type=model_type,
        port=port,
        positives=positives,
        task_type=task_type,
        language=language,
        examples_pool=examples_pool,
        num_examples=num_examples,
        thread_count=num_processes,
        gen_cache_dir=os.path.join(save_dir, language, task_type, "gen_cache_dir")
    )
    save_triplets(
        triplets=triplets,
        save_dir=save_dir,
        task_type=task_type,
        language=language,
    )
    end_time = time.time()
    print("=============================================================")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("=============================================================")
    print("DONE!")


if __name__ == "__main__":
    args = get_args()
    main(args)
