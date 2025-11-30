import os
import json
import random
from tqdm import tqdm
from hashlib import md5
from warnings import warn
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from llm import LLM
from utils import clean_content
from constant import Task, get_task, get_generation_prompt, get_quality_control_prompt


def compute_md5(text: str):
    return md5(text.encode()).hexdigest()


class TripletGenerator(LLM):
    def __init__(
        self,
        model: str = "Qwen2-5-Coder-32B-Instruct",
        model_type: str = "open-source",
        port: int = 8000,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(model, model_type, port)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def generate_triplets(
        self,
        data: dict,
        task: Task,
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        num_variants_per_doc: int = 1,
        narrative_focus: Optional[str] = None,
        debug_mode: bool = False,
        **kwargs,
    ):
        """
        对单个 doc 生成若干 (query, pos) 样本。

        narrative_focus:
            - 只对 AILAStatutes 有意义，用来控制叙事视角。
            - 典型值: "victim_focus", "investigation_focus", "judgment_focus",
                     "social_impact_focus", "neutral_brief"
        """
        result_list: List[dict] = []

        text = data["text"]

        for _ in range(num_variants_per_doc):
            examples = None
            if examples_pool is not None and len(examples_pool) > 0:
                examples = random.sample(
                    examples_pool,
                    min(num_examples, len(examples_pool)),
                )

            try:
                gen_prompt = get_generation_prompt(
                    task=task,
                    text=text,
                    examples=examples,
                    narrative_focus=narrative_focus,
                )
                response = self.chat(gen_prompt, **kwargs)[0]

                query = clean_content(response)
                pos = text

                if debug_mode:
                    result = {
                        "generation_prompt": gen_prompt,
                        "prompt": task.task_instruction,
                        "query": query,
                        "pos": [pos],
                        "neg": [],
                    }
                else:
                    result = {
                        "prompt": task.task_instruction,
                        "query": query,
                        "pos": [pos],
                        "neg": [],
                    }

                result_list.append(result)

                # 如果你以后想重新启用 QC，可以在这里加 get_quality_control_prompt(...)

            except Exception as e:
                warn(f"Error: {e}")

        return result_list

    def run_single(
        self,
        data: dict,
        task: Task,
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        num_variants_per_doc: int = 1,
        narrative_focus: Optional[str] = None,
        debug_mode: bool = False,
        **kwargs,
    ):
        """
        对单个 doc 进行带 cache 的生成。
        """
        result_list: List[dict] = []

        docid = compute_md5(data["text"])
        if self.cache_dir is not None:
            gen_data_cache_path = os.path.join(self.cache_dir, f"{docid}.json")
            if os.path.exists(gen_data_cache_path):
                with open(gen_data_cache_path, "r", encoding="utf-8") as f:
                    result_list = json.load(f)

                if len(result_list) > 0:
                    return result_list

        triplets = self.generate_triplets(
            data=data,
            task=task,
            examples_pool=examples_pool,
            num_examples=num_examples,
            num_variants_per_doc=num_variants_per_doc,
            narrative_focus=narrative_focus,
            debug_mode=debug_mode,
            **kwargs,
        )
        if len(triplets) == 0:
            return result_list

        if debug_mode:
            # debug 模式下，把所有生成都包进一个结构里
            result = {
                "docid": docid,
                "task_type": task.task_type.value,
                "language": task.language.value,
                "triplets": triplets,
            }
            result_list.append(result)
        else:
            # 训练模式下，默认保留其中一个样本
            result_list.append(random.choice(triplets))

        if self.cache_dir is not None:
            gen_data_cache_path = os.path.join(self.cache_dir, f"{docid}.json")
            with open(gen_data_cache_path, "w", encoding="utf-8") as f:
                json.dump(result_list, f, indent=4, ensure_ascii=False)

        return result_list

    def run(
        self,
        positives: List[dict],
        task_type: str,
        language: str = "en",
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        num_variants_per_doc: int = 1,
        narrative_focus: Optional[str] = None,
        tqdm_desc: str = "Generating triplets",
        debug_mode: bool = False,
        thread_count: int = 1,
        **kwargs,
    ):
        """
        对一批 positives 并行生成 triplets。
        """
        task = get_task(
            task_type=task_type,
            language=language,
        )

        result_list: List[dict] = []

        def process_positive(positive: dict):
            return self.run_single(
                data=positive,
                task=task,
                examples_pool=examples_pool,
                num_examples=num_examples,
                num_variants_per_doc=num_variants_per_doc,
                narrative_focus=narrative_focus,
                debug_mode=debug_mode,
                **kwargs,
            )

        # 多线程 + tqdm 进度条
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            results = list(
                tqdm(
                    executor.map(process_positive, positives),
                    total=len(positives),
                    desc=tqdm_desc,
                )
            )

        # 展平结果
        for res in results:
            if isinstance(res, list):
                result_list.extend(res)
            elif res is not None:
                result_list.append(res)

        return result_list
