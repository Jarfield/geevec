"""基于 LLM 的通用文档改写器。"""

"""面向多任务的数据增强：将种子文档改写为风格多样的新文档。"""

import os
import sys
import random
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# ---- 确保当前目录可被 import ----
THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
for _p in (ROOT_DIR, THIS_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from constant import TaskType, Language, get_task, Task  # type: ignore
from llm import LLM  # type: ignore
from utils import clean_content  # type: ignore
from attributes_config import (
    sample_attributes_for_task,
    attributes_to_hint_text,
    get_base_synthesis_prompt,
)


def _trim_trailing_your_output(prompt: str) -> str:
    marker = "Your output:"
    idx = prompt.rfind(marker)
    if idx == -1:
        return prompt.rstrip()
    return prompt[:idx].rstrip()


def _split_title_desc(raw_text: str) -> (str, str):
    text = raw_text.strip()
    if not text:
        return "", ""

    lines = text.splitlines()
    title = ""
    desc_lines: List[str] = []

    for line in lines:
        lower = line.strip().lower()
        if lower.startswith("title:"):
            title = line.split(":", 1)[1].strip()
        elif lower.startswith("desc:"):
            desc_lines.append(line.split(":", 1)[1].strip())
        else:
            desc_lines.append(line)

    if not title:
        title = lines[0].strip()
        desc_lines = lines[1:] if len(lines) > 1 else []

    desc = "\n".join(desc_lines).strip()
    return title, desc


class DocSynthesisGenerator(LLM):
    """用于多任务文档改写的 LLM 封装。"""

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
        attr_bundle,
    ) -> str:
        """组装完整的改写提示。"""

        attr_hint = attributes_to_hint_text(attr_bundle)
        base_prompt = get_base_synthesis_prompt(
            task=task,
            seed_text=seed_text,
            narrative_hint=attr_hint,
        )
        return _trim_trailing_your_output(base_prompt)

    def generate_single_doc(
        self,
        seed_doc: Dict[str, Any],
        task: Task,
        global_index: int,
        rng: Optional[random.Random] = None,
        debug_mode: bool = False,
        **llm_kwargs,
    ) -> Dict[str, Any]:
        if rng is None:
            rng = random

        seed_text = seed_doc.get("text") or ""
        if not seed_text.strip():
            raise ValueError("seed_doc must contain a non-empty 'text' field.")

        attr_bundle = sample_attributes_for_task(task.task_type, rng=rng)

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
            "attributes": attr_bundle.to_dict() if attr_bundle else {},
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
        tqdm_desc: str = "Synthesizing docs",
        debug_mode: bool = False,
        **llm_kwargs,
    ) -> List[Dict[str, Any]]:
        if len(seed_docs) == 0:
            return []

        task = get_task(task_type=task_type, language=language)

        tasks: List[Dict[str, Any]] = []
        idx = 0
        for seed in seed_docs:
            for _ in range(num_variants_per_seed):
                tasks.append({"seed": seed, "index": idx})
                idx += 1

        def _worker(item: Dict[str, Any]) -> Dict[str, Any]:
            seed = item["seed"]
            global_index = item["index"]
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


__all__ = ["DocSynthesisGenerator"]
