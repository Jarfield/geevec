import os
import sys
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_generation.shared.constants import Task, get_pair_scoring_prompt, get_task  # type: ignore
from data_generation.shared.llm import LLM  # type: ignore
from data_generation.shared.utils import clean_content, compute_md5  # type: ignore


class QueryDocPairScorer(LLM):
    """Score query-document pairs with an LLM to filter mined negatives."""

    def __init__(
        self,
        model: str = "Qwen2-5-72B-Instruct",
        model_type: str = "open-source",
        port: int = 8000,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(model, model_type, port)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, query: str, doc: str) -> Optional[str]:
        if self.cache_dir is None:
            return None
        query_md5 = compute_md5(query)
        doc_md5 = compute_md5(doc)
        return os.path.join(self.cache_dir, query_md5, f"{doc_md5}.json")

    def score_pair(
        self,
        query: str,
        doc: str,
        task: Task,
        **kwargs,
    ) -> Optional[float]:
        cache_path = self._cache_path(query, doc)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return cached.get("score")

        prompt = get_pair_scoring_prompt(task, query, doc)
        response = self.chat(prompt, **kwargs)[0]
        cleaned = clean_content(response)

        match = re.search(r"<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>", cleaned)
        if match is None:
            return None
        score = float(match.group(1))

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"query": query, "doc": doc, "score": score}, f, ensure_ascii=False, indent=2)

        return score

    def score_item(
        self,
        item: Dict,
        task: Task,
        pos_threshold: float = 4.0,
        neg_threshold: float = 2.0,
        **kwargs,
    ) -> Optional[Dict]:
        query = item.get("query")
        if not query:
            return None

        candidates: List[str] = []
        for field in ("pos", "neg", "topk"):
            for doc in item.get(field, []) or []:
                candidates.append(doc)

        pos_docs: List[str] = []
        neg_docs: List[str] = []
        score_details: List[Dict] = []

        for doc in candidates:
            score = self.score_pair(query, doc, task, **kwargs)
            score_details.append({"doc": doc, "score": score})
            if score is None:
                continue
            if score >= pos_threshold:
                pos_docs.append(doc)
            elif score <= neg_threshold:
                neg_docs.append(doc)

        scored = dict(item)
        scored["pos"] = pos_docs
        scored["neg"] = neg_docs
        scored["score_details"] = score_details

        return scored

    def run(
        self,
        data: List[Dict],
        task_type: str,
        language: str = "en",
        pos_threshold: float = 4.0,
        neg_threshold: float = 2.0,
        thread_count: int = 1,
        **kwargs,
    ) -> List[Dict]:
        task = get_task(task_type=task_type, language=language)

        def process(item: Dict) -> Optional[Dict]:
            return self.score_item(
                item,
                task=task,
                pos_threshold=pos_threshold,
                neg_threshold=neg_threshold,
                **kwargs,
            )

        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            for res in executor.map(process, data):
                if res is not None:
                    results.append(res)

        return results
