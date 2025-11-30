"""
statute_synthesis_generator.py

Use an LLM to synthesize NEW statute-like documents based on existing
reference statutes (seed docs) and a randomly sampled set of high-level
attributes (see attributes_config.py).

Each synthetic statute is returned as a dict that can be written into a
JSONL corpus, e.g.:

{
  "_id": "synth-00000001",
  "title": "...",
  "text": "...",
  "seed_id": "...",          # optional, from original corpus
  "attributes": {...}        # the sampled attribute bundle
}
"""

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

from constant import TaskType, Language, get_task, Task, get_statute_synthesis_prompt  # type: ignore
from llm import LLM  # type: ignore
from utils import clean_content  # type: ignore

from attributes_config import (  # type: ignore
    StatuteAttributeBundle,
    sample_statute_attributes,
    attributes_to_hint_text,
)


def _trim_trailing_your_output(prompt: str) -> str:
    """
    If the base prompt ends with 'Your output:' (as our helper does),
    strip that part so we can append extra guidance and then a fresh
    'Your output:' at the very end.
    """
    marker = "Your output:"
    idx = prompt.rfind(marker)
    if idx == -1:
        return prompt.rstrip()
    return prompt[:idx].rstrip()


def _split_title_desc(raw_text: str) -> (str, str):
    """
    Heuristically split the LLM output into (title, desc).

    We expect (and encourage) the model to output either:

        Title: ...
        Desc: ...

    or else we fall back to:
        title = first line
        desc  = remaining lines
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


class StatuteSynthesisGenerator(LLM):
    """
    LLM wrapper that synthesizes new statute-like documents
    from reference (seed) statutes, guided by high-level attributes.
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
        attr_bundle: StatuteAttributeBundle,
    ) -> str:
        """
        Build the full synthesis prompt:
        - base instruction from get_statute_synthesis_prompt(...)
        - + attribute constraints
        - + explicit output format (Title / Desc)
        """
        base_prompt = get_statute_synthesis_prompt(
            task=task,
            seed_text=seed_text,
            extra_style_examples=None,
        )
        base_prompt = _trim_trailing_your_output(base_prompt)

        attr_hint = attributes_to_hint_text(attr_bundle)

        format_hint = """
In addition, the new statute MUST be formatted as:

Title: <a concise heading describing the offence or rule>
Desc: <the full text of the legal provision, in one or more sentences or clauses>
"""

        full_prompt = f"""{base_prompt}

[High-level attributes for the new statute]
{attr_hint}

[Output format requirement]
{format_hint}

Now follow ALL of the above constraints carefully.

Your output:
"""
        return full_prompt

    def generate_single_statute(
        self,
        seed_doc: Dict[str, Any],
        task: Task,
        global_index: int,
        rng: Optional[random.Random] = None,
        debug_mode: bool = False,
        **llm_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate ONE synthetic statute from a single seed_doc.

        Args:
            seed_doc: a dict that MUST contain a "text" field.
            task: Task object for AILAStatutes.
            global_index: unique index used to build synthetic _id.
            rng: optional random.Random for reproducibility.
        """
        if rng is None:
            rng = random

        seed_text = seed_doc.get("text") or ""
        if not seed_text.strip():
            raise ValueError("seed_doc must contain a non-empty 'text' field.")

        # sample a fresh attribute bundle for this variant
        attr_bundle = sample_statute_attributes(rng=rng)

        # default decoding params to encourage diversity, if not provided
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
        language: str = "en",
        num_variants_per_seed: int = 1,
        thread_count: int = 1,
        tqdm_desc: str = "Synthesizing statutes",
        debug_mode: bool = False,
        **llm_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Batch synthesis over multiple seed docs.

        Args:
            seed_docs: list of seed statutes, each with at least 'text' field.
            task_type: e.g. "ailastatutes"
            language: e.g. "en"
            num_variants_per_seed: how many synthetic statutes per seed.
            thread_count: ThreadPoolExecutor worker数
        """
        if len(seed_docs) == 0:
            return []

        task = get_task(task_type=task_type, language=language)

        # Prepare flat list of tasks = (seed_doc, global_index)
        tasks: List[Dict[str, Any]] = []
        idx = 0
        for seed in seed_docs:
            for _ in range(num_variants_per_seed):
                tasks.append({"seed": seed, "index": idx})
                idx += 1

        def _worker(item: Dict[str, Any]) -> Dict[str, Any]:
            seed = item["seed"]
            global_index = item["index"]
            # Per-task rng to get reproducible attribute sampling if desired
            rng = random.Random(global_index + 1337)
            return self.generate_single_statute(
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
