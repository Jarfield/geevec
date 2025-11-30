from __future__ import annotations

import os
import sys
import random
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# ---- Make project root & this folder importable ----
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
for _p in (ROOT_DIR, THIS_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from constant import (  # type: ignore
    TaskType,
    Language,
    get_task,
    Task,
    get_covid_doc_synthesis_prompt,
)
from llm import LLM  # type: ignore
from utils import clean_content  # type: ignore

from covid_attributes_config import (  # type: ignore
    CovidArticleAttributeBundle,
    sample_covid_attributes,
    attributes_to_hint_text,
)


def _trim_trailing_your_output(prompt: str) -> str:
    """
    如果 base prompt 以 'Your output:' 结尾，就去掉这一段，
    方便后面追加属性说明和新的 'Your output:'。
    """
    marker = "Your output:"
    idx = prompt.rfind(marker)
    if idx == -1:
        return prompt.rstrip()
    return prompt[:idx].rstrip()


def _split_title_desc(raw_text: str) -> (str, str):
    """
    将 LLM 输出粗略切分为 (title, desc)。

    预期格式为：
        Title: ...
        Desc: ...

    如果没找到前缀，则退化为：
        第一行作为 title，其余作为 desc。
    """
    text = raw_text.strip()
    if not text:
        return "", ""

    lines = text.splitlines()
    title = ""
    desc_lines: List[str] = []

    for line in lines:
        lower = line.strip().lower()
        if lower.startswith("title:"):
            # everything after "Title:" as title
            title = line.split(":", 1)[1].strip()
        elif lower.startswith("desc:"):
            # everything after "Desc:" as part of description
            desc_lines.append(line.split(":", 1)[1].strip())
        else:
            desc_lines.append(line)

    if not title:
        # fallback: first line is title, rest is desc
        title = lines[0].strip()
        desc_lines = lines[1:] if len(lines) > 1 else []

    desc = "\n".join(desc_lines).strip()
    return title, desc


class CovidDocSynthesisGenerator(LLM):
    """
    LLM wrapper that synthesizes new Covid-related Chinese documents
    from reference (seed) documents, guided by high-level attributes.
    """

    def __init__(
        self,
        model: str = "Qwen2-5-72B-Instruct",
        model_type: str = "open-source",
        port: int = 8000,
    ):
        super().__init__(model, model_type, port)

    def _build_prompt(
        self,
        task: Task,
        seed_text: str,
        attr_bundle: CovidArticleAttributeBundle,
    ) -> str:
        """
        构建完整的合成 prompt：
        - 基础指令（如何仿写 Covid 文稿）来自 get_covid_doc_synthesis_prompt(...)
        - + 属性约束（文章类型 / 来源 / 主题 / 语气等）
        - + 输出格式（Title / Desc）
        """
        base_prompt = get_covid_doc_synthesis_prompt(
            task=task,
            seed_text=seed_text,
            extra_style_examples=None,
        )
        base_prompt = _trim_trailing_your_output(base_prompt)

        attr_hint = attributes_to_hint_text(attr_bundle)

        format_hint = """
在输出时，请严格使用以下格式（注意大小写）：

Title: <一句话中文标题>
Desc: <多段中文正文，可以包含若干自然段，每段之间可留空行>
"""

        full_prompt = f"""{base_prompt}

[高层次属性约束（必须在整体上满足）]
{attr_hint}

[输出格式要求]
{format_hint}

现在请在综合考虑以上所有要求的基础上，给出最终输出。

Your output:
"""
        return full_prompt

    def generate_single_doc(
        self,
        seed_doc: Dict[str, Any],
        task: Task,
        global_index: int,
        rng: Optional[random.Random] = None,
        debug_mode: bool = False,
        **llm_kwargs,
    ) -> Dict[str, Any]:
        """
        从一个 seed_doc 生成 ONE synthetic Covid 文章。

        Args:
            seed_doc: 必须包含 "text" 字段（原始中文文章）。
            task: Task 对象（一般是 TaskType.covidretrieval + zh）。
            global_index: 用于构造合成文档的唯一 _id。
            rng: 可选 random.Random，用于可复现的属性采样。
        """
        if rng is None:
            rng = random

        seed_text = seed_doc.get("text") or ""
        if not seed_text.strip():
            raise ValueError("seed_doc must contain a non-empty 'text' field.")

        # 为该 variant 采样一套 Covid 属性
        attr_bundle = sample_covid_attributes(rng=rng)

        # 默认给一点随机性
        llm_kwargs.setdefault("temperature", 0.9)
        llm_kwargs.setdefault("top_p", 0.95)

        prompt = self._build_prompt(task, seed_text, attr_bundle)
        raw_output = self.chat(prompt, **llm_kwargs)[0]
        cleaned = clean_content(raw_output)

        title, desc = _split_title_desc(cleaned)

        synth_doc: Dict[str, Any] = {
            "_id": f"synth-{global_index:08d}",
            "title": title,
            "text": desc,
            "seed_id": seed_doc.get("_id"),
            "attributes": attr_bundle.to_dict(),
        }

        if debug_mode:
            synth_doc["raw_output"] = raw_output
            synth_doc["prompt"] = prompt

        return synth_doc

    def run(
        self,
        seed_docs: List[Dict[str, Any]],
        task_type: str,
        language: str = "zh",
        num_variants_per_seed: int = 1,
        thread_count: int = 1,
        tqdm_desc: str = "Synthesizing Covid docs",
        debug_mode: bool = False,
        **llm_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        对多个 seed_docs 做批量合成。

        Args:
            seed_docs: 每个元素至少包含 'text' 字段。
            task_type: 一般为 "covidretrieval"。
            language: 一般为 "zh"。
            num_variants_per_seed: 每个 seed 生成多少条新 doc。
            thread_count: ThreadPoolExecutor worker 数。
        """
        if len(seed_docs) == 0:
            return []

        task = get_task(task_type=task_type, language=language)

        # 展平成任务队列 (seed_doc, global_index)
        tasks: List[Dict[str, Any]] = []
        idx = 0
        for seed in seed_docs:
            for _ in range(num_variants_per_seed):
                tasks.append({"seed": seed, "index": idx})
                idx += 1

        def _worker(item: Dict[str, Any]) -> Dict[str, Any]:
            seed = item["seed"]
            global_index = item["index"]
            # 每个任务各自一个 rng，保证属性采样可复现
            rng = random.Random(global_index + 1337)
            return self.generate_single_doc(
                seed_doc=seed,
                task=task,
                global_index=global_index,
                rng=rng,
                debug_mode=debug_mode,
                **llm_kwargs,
            )

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            for doc in tqdm(
                executor.map(_worker, tasks),
                total=len(tasks),
                desc=tqdm_desc,
            ):
                results.append(doc)

        return results
