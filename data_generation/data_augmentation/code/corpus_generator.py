"""Task-aware corpus loader used by query and corpus generation scripts."""

import os
import sys
import random
from typing import List

from datasets import Dataset, load_dataset
from tqdm import tqdm

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_generation.data_preparation.code.task_configs import TaskDatasetConfig, get_task_config


def _load_qrels_ids(qrels_path: str, cfg: TaskDatasetConfig) -> set:
    qrels_ds = Dataset.from_file(qrels_path)
    pos_docids = set()
    for d in qrels_ds:
        score = int(d.get(cfg.qrels_score_key, 0))
        if score > 0:
            pos_docids.add(d.get(cfg.qrels_pid_key))
    return pos_docids


def _load_corpus_records(corpus_path: str, cfg: TaskDatasetConfig) -> Dataset:
    if corpus_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=corpus_path)["train"]
    else:
        ds = Dataset.from_file(corpus_path)
    return ds


def _filter_corpus(
    ds: Dataset,
    cfg: TaskDatasetConfig,
    allowed_ids: set | None,
) -> List[dict]:
    corpus_list = []
    for data in tqdm(ds, desc="Loading corpus"):
        cid = data.get(cfg.id_key)
        text = data.get(cfg.text_key)
        if text is None:
            continue
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        if len(text) < cfg.min_len:
            continue
        corpus_list.append({
            "_id": cid,
            "text": text,
        })
    return corpus_list


class CorpusGenerator:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir

    def run(
        self,
        task_type: str,
        language: str,
        corpus_path: str | None = None,
        qrels_path: str | None = None,
        num_samples: int = -1,
    ) -> List[dict]:
        cfg = get_task_config(task_type)

        corpus_path = corpus_path or cfg.corpus_path
        qrels_path = qrels_path or cfg.qrels_path

        if corpus_path is None:
            raise ValueError(
                f"No corpus_path provided for task {task_type}. Update task_configs.py or pass --corpus_path."
            )

        corpus_ds = _load_corpus_records(corpus_path, cfg)
        allowed_ids = None
        if qrels_path:
            allowed_ids = _load_qrels_ids(qrels_path, cfg)
            print(f"[INFO] Filtered positive doc ids from qrels: {len(allowed_ids)}")

        corpus_list = _filter_corpus(corpus_ds, cfg, allowed_ids)

        if num_samples > 0 and num_samples < len(corpus_list):
            corpus_list = random.sample(corpus_list, num_samples)

        return corpus_list


__all__ = ["CorpusGenerator"]
