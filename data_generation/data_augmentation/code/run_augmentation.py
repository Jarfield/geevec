import os
import sys
import json
import time
import argparse
import random
import multiprocessing as mp
from typing import List, Optional

from datasets import Dataset

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_generation.shared.constants import Language, TaskType  # type: ignore
from data_generation.shared.utils import compute_md5  # type: ignore
from data_generation.data_preparation.code.task_configs import (  # type: ignore
    DEFAULT_GENERATED_ROOT,
    default_generated_corpus_path,
    get_task_config,
)
from triplet_generator import TripletGenerator


def get_args():
    """解析命令行参数。

    - 输入：命令行传入的各项开关。
    - 输出：`argparse.Namespace`，后续直接在 `main` 中使用。
    主要逻辑：定义任务、模型、进程数、数据路径等参数，并提供默认值。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', '--task_type',
        dest='task',
        type=str,
        required=True,
        help='The task to generate data for',
        choices=[t.name for t in TaskType],
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
        '--corpus_path',
        type=str,
        default=None,
        help='Optional override for corpus path. Defaults to the synthesized corpus under DATA_AUG_GENERATED_ROOT.'
    )
    parser.add_argument(
        '--qrels_path',
        type=str,
        default=None,
        help='Optional override for qrels path. Defaults to task_configs.TASK_DATASETS.'
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
        help='The type of model to use for generation. Default: open-source'
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


def load_generated_corpus(corpus_path: str, num_samples: int = -1) -> List[dict]:
    """直接读取合成语料文件，逐条返回用于生成查询的样本。

    输入：
    - corpus_path：`run_corpus_generation.py` 产出的合成语料路径，支持 `.jsonl` 或 Arrow。
    - num_samples：若 >0，则按原始顺序截取前 N 条；-1 代表使用全部。

    输出：包含 text 等字段的字典列表；只要 `text` 不为空就会被保留，不再做 qrels 过滤。
    主要逻辑：根据扩展名选择读取方式，保持语料顺序，确保生成阶段“一条语料一条生成”。
    """

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus path does not exist: {corpus_path}")

    records: List[dict] = []

    if corpus_path.endswith(".jsonl"):
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = data.get("text") or data.get("desc")
                if text:
                    records.append(data)
    else:
        ds = Dataset.from_file(corpus_path)
        for item in ds:
            text = item.get("text") or item.get("desc")
            if text:
                records.append(dict(item))

    if num_samples > 0:
        records = records[:num_samples]

    return records


def load_examples_pool(examples_path: str, sample_size: int = 30) -> Optional[List[dict]]:
    """加载 few-shot 示例池，自动兼容 JSON / JSONL，并容忍字段差异。

    输入：
    - examples_path：示例文件路径，既可能是一个 JSON list，也可能是逐行 JSON（JSONL）。
    - sample_size：最多抽取多少个示例进入内存，默认 30。

    输出：若读取成功返回**字段已对齐的**字典列表，否则返回 None。
    主要逻辑：
    1) 优先尝试 `json.load`（支持单个 list / dict）；
    2) 如果解析失败，再按逐行 JSON 解析并忽略空行；
    3) 将读取到的对象统一映射为 `{input, output}` 结构，自动兼容 `context`、`content`、`target`、`query` 等字段名，缺失时跳过；
    4) 最后对有效列表进行随机抽样，保证与原逻辑一致。
    """

    if not os.path.exists(examples_path):
        print(f"[WARN] examples_path not found: {examples_path}")
        return None

    def _normalize_to_list(obj):
        # 将单个对象或字典包装成列表，保持后续处理一致
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        return [obj]

    examples: List[dict] = []

    try:
        # 先尝试常规 JSON 读取
        with open(examples_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
            examples = _normalize_to_list(raw)
    except json.JSONDecodeError:
        # 如果文件包含多个 JSON 对象（如 JSONL），逐行解析
        with open(examples_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    examples.append(obj)
                except json.JSONDecodeError:
                    print(f"[WARN] Skip invalid example line: {line[:50]}...")
    except Exception as e:
        print(f"[WARN] Failed to load examples from {examples_path}: {e}")
        return None

    def _normalize_example(example: dict) -> Optional[dict]:
        """把不同字段名的示例映射成统一的 input/output。

        - input 备选字段：`input`、`context`、`content`、`text`、`doc`、`document`、`pos`、`positive`、`passage`。
        - output 备选字段：`output`、`target`、`query`、`question`、`label`。

        缺失任意一侧都会返回 None 并跳过。
        """

        input_candidates = [
            "input",
            "context",
            "content",
            "text",
            "doc",
            "document",
            "pos",
            "positive",
            "passage",
        ]
        output_candidates = [
            "output",
            "target",
            "query",
            "question",
            "label",
        ]

        def _pick_first(keys):
            for k in keys:
                if k in example and example[k]:
                    return example[k]
            return None

        normalized_input = _pick_first(input_candidates)
        normalized_output = _pick_first(output_candidates)

        # 某些示例的 `pos` 可能是列表（例如多个正样本），这里兼容并拼成单一字符串
        if isinstance(normalized_input, list):
            normalized_input = "\n".join(map(str, normalized_input))

        if isinstance(normalized_output, list):
            normalized_output = "\n".join(map(str, normalized_output))

        if normalized_input is None or normalized_output is None:
            print(
                f"[WARN] Skip example lacking input/output fields: "
                f"available keys = {list(example.keys())}"
            )
            return None

        return {"input": normalized_input, "output": normalized_output}

    normalized_examples: List[dict] = []
    for ex in examples:
        if not isinstance(ex, dict):
            print(f"[WARN] Skip non-dict example: {ex}")
            continue
        normalized = _normalize_example(ex)
        if normalized:
            normalized_examples.append(normalized)

    if len(normalized_examples) == 0:
        print(f"[WARN] No valid examples parsed from {examples_path}")
        return None

    if len(normalized_examples) > sample_size:
        normalized_examples = random.sample(normalized_examples, sample_size)

    return normalized_examples


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
    """调用 `TripletGenerator` 生成查询-正例对。

    输入：模型相关配置、待生成的正例列表（每条含 text）、任务与语言信息、few-shot 示例池等。
    输出：包含 query/pos 的三元组列表；内部会做多线程并行，遵循传入的变体数和叙事焦点。
    """

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
    """返回单轮模式下的输出路径并确保目录存在。"""

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
    """将生成的三元组落盘；自动与旧文件去重并追加。"""

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
    """入口函数：按指定语料逐条生成查询并写入输出文件。"""

    model = args.model
    model_type = args.model_type
    port = args.port

    num_samples = args.num_samples
    num_rounds = args.num_rounds

    task_type = args.task
    language = args.language

    save_dir = args.save_dir
    cache_dir = args.cache_dir
    num_processes = min(args.num_processes, int(mp.cpu_count() * 0.8))
    overwrite = args.overwrite
    gen_cache_root = cache_dir or save_dir

    # Always default to the synthesized corpus produced by run_corpus_generation.
    if args.corpus_path:
        corpus_path = args.corpus_path
        print(f"[INFO] Using user-provided corpus_path: {corpus_path}")
    else:
        prepared_corpus = os.path.join(
            DEFAULT_GENERATED_ROOT,
            task_type,
            "preparation",
            f"{language}_corpus_filtered.jsonl",
        )
        if os.path.exists(prepared_corpus):
            corpus_path = prepared_corpus
            print(f"[INFO] Using prepared corpus from: {corpus_path}")
        else:
            corpus_path = default_generated_corpus_path(task_type, language)
            if not os.path.exists(corpus_path):
                raise FileNotFoundError(
                    f"No synthesized corpus found at {corpus_path}. "
                    "Please run script/run_corpus.sh first, run_preparation.sh, or pass --corpus_path explicitly."
                )
            print(f"[INFO] Using synthesized corpus from: {corpus_path}")

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

    examples_dir = args.examples_dir or get_task_config(task_type).examples_dir
    num_examples = args.num_examples
    if examples_dir is not None:
        examples_path = os.path.join(examples_dir, task_type, f"{language}_sample_examples.json")
        examples_pool = load_examples_pool(examples_path, sample_size=30)
    else:
        examples_pool = None

    if args.qrels_path:
        print("[INFO] 当前生成逻辑直接遍历合成语料，不再使用 qrels 做过滤，忽略传入的 qrels_path。")

    positives = load_generated_corpus(
        corpus_path=corpus_path,
        num_samples=num_samples,
    )
    print(f"[INFO] Num positives used for generation: {len(positives)}")
    print("=================== Generate training data ===================")
    print(f'Task: {task_type} | Language: {language}')
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
            gen_cache_dir=os.path.join(gen_cache_root, language, task_type, "gen_cache_dir"),
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
                gen_cache_root, language, task_type, f"gen_cache_dir_round{round_idx}"
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
