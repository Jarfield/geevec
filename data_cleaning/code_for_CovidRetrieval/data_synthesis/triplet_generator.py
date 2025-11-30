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
from constant import Task, get_task, get_generation_prompt  # 不再需要 get_quality_control_prompt


def compute_md5(text: str):
    return md5(text.encode()).hexdigest()


def _extract_doc_text(data: dict) -> str:
    """
    从 item 里抽取要打分的正文：
    优先使用 data["text"]；
    如果没有，则从 data["pos"] 里取（支持 list 或 str）。
    """
    # 1) 优先用 text 字段（兼容之前的实现）
    if "text" in data and isinstance(data["text"], str) and data["text"].strip():
        return data["text"]

    # 2) 其次用 pos 字段
    if "pos" in data:
        pos = data["pos"]
        # 常见格式：pos 是一个 list，每个元素是一个文档
        if isinstance(pos, list):
            if not pos:
                return ""
            # 这里默认取第一个正例文档
            return pos[0]
        # 也兼容直接是字符串的情况
        if isinstance(pos, str):
            return pos

    # 3) 都没有就返回空字符串（上层会直接跳过）
    return ""


class TripletGenerator(LLM):
    """
    现在不再真正生成 triplets，而是对每篇文档打一个 1~5 的相关性分数。
    为了减少改动，类名先保持不变，外层调用只需要改存储逻辑即可。
    """

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

    def _score_single_doc(
        self,
        data: dict,
        task: Task,
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        debug_mode: bool = False,
        **kwargs,
    ) -> Optional[dict]:
        """
        对单条 data 打分，data 至少要能提供正文（text 或 pos）。
        返回一个 dict:
        {
            "docid": ...,
            "task_type": ...,
            "language": ...,
            "text": ...,
            "score": 1~5,
            (debug 时还会有 raw_response / generation_prompt)
        }
        """
        examples = None
        if examples_pool is not None and len(examples_pool) > 0 and num_examples > 0:
            examples = random.sample(examples_pool, min(num_examples, len(examples_pool)))

        try:
            text = _extract_doc_text(data)
            if not text or not isinstance(text, str) or not text.strip():
                warn("Empty text when scoring doc, skip this item.")
                return None

            docid = compute_md5(text)

            # 构造打分 prompt（constant.get_generation_prompt 里已经改成 1~5 打分逻辑）
            gen_prompt = get_generation_prompt(
                task=task,
                text=text,
                examples=examples,
            )

            # 用 LLM 生成分数：只需要很少 token & 低温度，保证稳定
            default_llm_kwargs = dict(
                max_tokens=4,
                temperature=0.0,
                top_p=1.0,
                stop=["\n"],
            )
            merged_kwargs = {**default_llm_kwargs, **kwargs}

            response = self.chat(gen_prompt, **merged_kwargs)[0]
            if response is None:
                warn("LLM returned None response, skip this doc.")
                return None

            raw = clean_content(response)

            # 只保留第一个数字字符，保证是 1~5 之间
            digits = [ch for ch in raw if ch.isdigit()]
            if not digits:
                warn(f"Cannot parse score from response: {raw!r}")
                return None

            score = int(digits[0])
            if score < 1 or score > 5:
                warn(f"Parsed score out of range [1,5]: {score} (raw: {raw!r})")
                return None

            result = {
                "docid": docid,
                "task_type": task.task_type.name,
                "language": task.language.name,
                "text": text,       # 这里统一写回 text，方便后续过滤 & 训练
                "score": score,
            }

            if debug_mode:
                result["raw_response"] = raw
                result["generation_prompt"] = gen_prompt

            return result

        except Exception as e:
            warn(f"Error when scoring doc: {e}")
            return None

    def run_single(
        self,
        data: dict,
        task: Task,
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        debug_mode: bool = False,
        **kwargs,
    ) -> Optional[dict]:
        """
        带 cache 的单条打分。
        """
        text = _extract_doc_text(data)
        if not text or not isinstance(text, str) or not text.strip():
            warn("Empty text when scoring doc in run_single, skip.")
            return None

        docid = compute_md5(text)

        # 命中 cache 直接读
        if self.cache_dir is not None:
            gen_data_cache_path = os.path.join(self.cache_dir, f"{docid}.json")
            if os.path.exists(gen_data_cache_path):
                try:
                    with open(gen_data_cache_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    return cached
                except Exception as e:
                    warn(f"Error loading cache for {docid}: {e}")

        result = self._score_single_doc(
            data=data,
            task=task,
            examples_pool=examples_pool,
            num_examples=num_examples,
            debug_mode=debug_mode,
            **kwargs,
        )

        # 写 cache
        if result is not None and self.cache_dir is not None:
            gen_data_cache_path = os.path.join(self.cache_dir, f"{docid}.json")
            try:
                with open(gen_data_cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                warn(f"Error saving cache for {docid}: {e}")

        return result

    def run(
        self,
        positives: List[dict],
        task_type: str,
        language: str = "zh",
        examples_pool: Optional[List[dict]] = None,
        num_examples: int = 3,
        tqdm_desc: str = "Scoring documents",
        debug_mode: bool = False,
        thread_count: int = 1,
        **kwargs,
    ) -> List[dict]:
        """
        对 positives 列表做并行打分。
        兼容两种输入格式：
          1) {"text": "..."}               # 直接是文档
          2) {"pos": ["doc1", "doc2", ...]} # 使用第一个正例文档
        """
        task = get_task(
            task_type=task_type,
            language=language,
        )

        results: List[dict] = []

        def process_positive(positive: dict) -> Optional[dict]:
            return self.run_single(
                data=positive,
                task=task,
                examples_pool=examples_pool,
                num_examples=num_examples,
                debug_mode=debug_mode,
                **kwargs,
            )

        # 多线程 + tqdm 进度条
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            for res in tqdm(
                executor.map(process_positive, positives),
                total=len(positives),
                desc=tqdm_desc,
            ):
                if isinstance(res, dict):
                    results.append(res)

        return results
