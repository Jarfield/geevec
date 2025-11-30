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
        default='en',
        help='The language to generate for. ISO 639-1 code. Default: en',
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
        help='The model to use for generation. Default: Qwen2.5-72B-Instruct'
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
    parser.add_argument(
        '--num_variants_per_doc',
        type=int,
        default=1,
        help='The number of variants to generate per document. Default: 1'
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=1,
        help='How many rounds of query generation to run. '
             'Each round will be saved into a separate file.'
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
    num_variants_per_doc: int = 1,
    narrative_focus: Optional[str] = None,
):
    triplet_generator = TripletGenerator(model, model_type, port, cache_dir=gen_cache_dir)
    triplets = triplet_generator.run(
        positives=positives,
        task_type=task_type,
        language=language,
        examples_pool=examples_pool,
        num_examples=num_examples,
        num_variants_per_doc=num_variants_per_doc,
        tqdm_desc=tqdm_desc,
        thread_count=thread_count,
        narrative_focus=narrative_focus,
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

    with open(save_path, 'w', encoding='utf-8') as f:
        for triplet in old_triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')

        for triplet in triplets:
            _query_md5 = compute_md5(triplet['query'])
            _pos_md5 = compute_md5(triplet['pos'][0])
            if _query_md5 in query_md5s or _pos_md5 in pos_md5s:
                continue
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
    print(f"Triplets saved to {save_path}")

def get_save_path_for_round(
    save_dir: str,
    task_type: str,
    language: str,
    round_idx: int,
):
    """
    每一轮单独一个文件，比如：
    {save_dir}/{language}/{task_type}/en-triplets_round1.jsonl
    """
    base_dir = os.path.join(save_dir, language, task_type)
    os.makedirs(base_dir, exist_ok=True)
    file_name = f"{language}-triplets_round{round_idx}.jsonl"
    return os.path.join(base_dir, file_name)


def save_triplets_for_round(
    triplets: list,
    save_dir: str,
    task_type: str,
    language: str,
    round_idx: int,
):
    """
    保存当前这一轮的 triplets。这里我们简单点：这一轮只写这一轮的结果，
    不和旧文件混合（有 overwrite 逻辑在 main 里控制是否跳过）。
    """
    if len(triplets) == 0:
        print(f"[Round {round_idx}] No triplets to save: {task_type} | {language}")
        return

    save_path = get_save_path_for_round(save_dir, task_type, language, round_idx)

    # 你要是想在一轮内部去重，也可以加 md5；简单起见，先直接写。
    with open(save_path, "w", encoding="utf-8") as f:
        for triplet in triplets:
            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

    print(f"[Round {round_idx}] Triplets saved to {save_path}")


def main(args):
    model = args.model
    model_type = args.model_type
    port = args.port

    num_samples = args.num_samples
    num_rounds = args.num_rounds

    task_type = args.task_type
    language = args.language

    save_dir = args.save_dir
    cache_dir = args.cache_dir
    num_processes = min(args.num_processes, int(mp.cpu_count() * 0.8))
    overwrite = args.overwrite

    # ===== Single-round old behavior: keep backward compatible =====
    if num_rounds == 1:
        save_path = get_save_path(save_dir, task_type, language)

        # 只有在 num_samples > 0 时才做“已有就跳过”的判断
        if num_samples > 0 and os.path.exists(save_path) and not overwrite:
            data = []
            with open(save_path) as f:
                for line in f:
                    data.append(json.loads(line))
            if len(data) >= num_samples * 0.8:
                print(f"Triplets already exist at {save_path}. Skipping generation.")
                return
            else:
                print(f"Triplets already exist at {save_path}. But samples is really small, continue generation.")
                num_samples = int((num_samples - len(data)) * 1.25)  # consider the filtered samples
    else:
        # 多轮模式下一般不做这个 skip，只是把 num_samples 当作“用于生成的 doc 数量”
        if num_samples <= 0:
            print("[INFO] num_samples <= 0 in multi-round mode: use all positives.")

    corpus_generator = CorpusGenerator(cache_dir)

    examples_dir = args.examples_dir
    num_examples = args.num_examples
    if examples_dir is not None:
        examples_path = os.path.join(examples_dir, task_type, f"{language}_sample_examples.json")
        try:
            with open(examples_path, 'r', encoding='utf-8') as f:
                examples_pool = json.load(f)
                examples_pool = random.sample(
                    examples_pool,
                    min(30, len(examples_pool))
                )  # sample 30 examples for few-shot generation
        except Exception as e:
            print(f'Error for loading examples from {examples_path}: {e}')
            examples_pool = None
    else:
        examples_pool = None

    positives = corpus_generator.run(
        language=language,
        num_samples=num_samples,
    )
    print(f"[INFO] Num positives used for generation: {len(positives)}")
    print("=================== Generate training data ===================")
    print(f'Task Type: {task_type} | Language: {language}')
    start_time = time.time()

    if num_rounds == 1:
        # ===== 单轮模式：保持原逻辑，方便兼容以前的代码 =====
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
            gen_cache_dir=os.path.join(save_dir, language, task_type, "gen_cache_dir"),
            num_variants_per_doc=getattr(args, "num_variants_per_doc", 1),
        )
        save_triplets(
            triplets=triplets,
            save_dir=save_dir,
            task_type=task_type,
            language=language,
        )
    else:
        # ===== 多轮模式：每一轮每个 doc 只生成 1 条，分文件保存 =====

        if task_type == "covidretrieval":
            # CovidRetrieval：按“问题类型”来区分每一轮
            focus_sequence = [
                "covid_fact_detail",        # 细节事实型问题
                "covid_policy_measure",     # 政策 / 措施型问题
                "covid_vaccine_treatment",  # 疫苗 / 药物 / 临床型问题
                "covid_risk_protection",    # 风险评估 / 防护建议型问题
                "covid_social_impact",      # 社会影响 / 经济教育影响型问题
            ]
        elif task_type == "ailastatutes":
            # AILAStatutes 原来的叙事视角
            focus_sequence = [
                "victim_focus",
                "investigation_focus",
                "judgment_focus",
                "social_impact_focus",
                "neutral_brief",
            ]
        else:
            # 其他任务：不使用 narrative_focus
            focus_sequence = [None] * max(num_rounds, 1)

        for round_idx in range(1, num_rounds + 1):
            print(f"\n********** Round {round_idx}/{num_rounds} **********")
            round_save_path = get_save_path_for_round(
                save_dir=save_dir,
                task_type=task_type,
                language=language,
                round_idx=round_idx,
            )

            if os.path.exists(round_save_path) and not overwrite:
                print(f"[Round {round_idx}] File already exists at {round_save_path}, skip this round.")
                continue

            # 每一轮选择一个 focus；>len 时自动循环
            narrative_focus = focus_sequence[(round_idx - 1) % len(focus_sequence)]

            round_cache_dir = os.path.join(
                save_dir, language, task_type, f"gen_cache_dir_round{round_idx}"
            )

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
                gen_cache_dir=round_cache_dir,
                num_variants_per_doc=1,        # 每轮每 doc 一条
                narrative_focus=narrative_focus,
            )
            save_triplets_for_round(
                triplets=triplets,
                save_dir=save_dir,
                task_type=task_type,
                language=language,
                round_idx=round_idx,
            )

    end_time = time.time()
    print("=============================================================")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("=============================================================")
    print("DONE!")

if __name__ == "__main__":
    args = get_args()
    main(args)
