from __future__ import annotations

import os
import json
import random
from typing import List, Dict, Any, Optional

from datasets import Dataset
from tqdm import tqdm


class CorpusGenerator:
    """
    简化版：只负责从“已经 augmented 的数据”中加载文档。
    不再依赖 C-MTEB 原始 corpus / qrels。

    支持：
      - input_path 指向单个 *.jsonl 文件；
      - input_path 指向一个目录，目录下所有 *.jsonl / *.arrow / *.parquet 文件会被加载。

    每条返回的样本是一个 dict，至少包含 "text" 字段（原有字段会保留）。
    """

    def __init__(self, input_path: str, cache_dir: Optional[str] = None, min_len: int = 0):
        self.input_path = input_path
        self.cache_dir = cache_dir
        self.min_len = min_len

    def _iter_input_files(self) -> List[str]:
        """遍历需要加载的所有文件路径。"""
        if os.path.isdir(self.input_path):
            files = []
            for name in sorted(os.listdir(self.input_path)):
                if name.endswith(".jsonl") or name.endswith(".arrow") or name.endswith(".parquet"):
                    files.append(os.path.join(self.input_path, name))
            if not files:
                raise FileNotFoundError(
                    f"No *.jsonl / *.arrow / *.parquet found under dir: {self.input_path}"
                )
            return files
        else:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"input_path not found: {self.input_path}")
            return [self.input_path]

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                text = data.get("text", "")
                if not isinstance(text, str):
                    continue
                if self.min_len > 0 and len(text) < self.min_len:
                    continue
                docs.append(data)
        print(f"[INFO] Loaded {len(docs)} docs from jsonl: {path}")
        return docs

    def _load_arrow_like(self, path: str) -> List[Dict[str, Any]]:
        ds = Dataset.from_file(path)
        docs: List[Dict[str, Any]] = []
        for row in tqdm(ds, desc=f"Loading {os.path.basename(path)}"):
            data = dict(row)
            text = data.get("text", "")
            if not isinstance(text, str):
                continue
            if self.min_len > 0 and len(text) < self.min_len:
                continue
            docs.append(data)
        print(f"[INFO] Loaded {len(docs)} docs from arrow/parquet: {path}")
        return docs

    def run(
        self,
        language: str,   # 为兼容旧接口保留，这里不会实际使用
        num_samples: int = -1,
    ) -> List[Dict[str, Any]]:
        all_docs: List[Dict[str, Any]] = []
        for path in self._iter_input_files():
            if path.endswith(".jsonl"):
                docs = self._load_jsonl(path)
            else:
                docs = self._load_arrow_like(path)
            all_docs.extend(docs)

        print(f"[INFO] Total loaded docs before sampling: {len(all_docs)}")

        if num_samples > 0 and num_samples < len(all_docs):
            all_docs = random.sample(all_docs, num_samples)
            print(f"[INFO] Sampled {len(all_docs)} docs.")

        return all_docs
