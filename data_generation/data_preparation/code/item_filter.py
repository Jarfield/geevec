"""Filter out evaluation positives from the full corpus for data synthesis."""

import argparse
import json
import os
import sys
from typing import Iterable, Set

from datasets import Dataset, load_dataset
from tqdm import tqdm

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_generation.shared.constants import Language, TaskType
from data_generation.shared.utils import compute_md5
from data_generation.data_preparation.code.task_configs import (
    DEFAULT_GENERATED_ROOT,
    TaskDatasetConfig,
    get_task_config,
)


def _load_jsonl_or_arrow(path: str) -> Iterable[dict]:
    """通用加载函数，支持 JSONL 和 Arrow"""
    if path.endswith(".jsonl"):
        return (json.loads(line) for line in open(path, "r", encoding="utf-8"))
    return Dataset.from_file(path)

def _load_queries(queries_path: str, cfg: TaskDatasetConfig) -> dict[str, str]:
    """加载查询 ID 到查询文本的映射"""
    queries = {}
    for row in _load_jsonl_or_arrow(queries_path):
        qid = row.get(cfg.query_id_key)
        text = row.get(cfg.query_text_key)
        if qid is not None and text is not None:
            queries[str(qid)] = text
    return queries

def process_data(
    args: argparse.Namespace,
    cfg: TaskDatasetConfig,
    corpus_path: str,
    qrels_path: str,
    queries_path: str = None,
) -> list[dict]:
    # 1. 预加载 qrels (只看 score=1)
    # 这里的 qrels 需要保存 qid 和 pid 的关系
    qrels_records = []
    pos_pids = set()
    for row in _load_jsonl_or_arrow(qrels_path):
        try:
            if float(row.get(cfg.qrels_score_key, 0)) == 1.0:
                qid = str(row.get(cfg.qrels_qid_key))
                pid = str(row.get(cfg.qrels_pid_key))
                qrels_records.append((qid, pid))
                pos_pids.add(pid)
        except (ValueError, TypeError):
            continue
    
    # 2. 加载 Corpus 文本映射
    # 无论哪个模式，都需要根据 pid 找 corpus text
    corpus_map = {}
    for row in tqdm(_load_jsonl_or_arrow(corpus_path), desc="Loading Corpus"):
        cid = str(row.get(cfg.id_key))
        if cid in pos_pids:
            corpus_map[cid] = row.get(cfg.text_key)

    # 3. 根据模式输出
    results = []
    if args.item == "corpus":
        # 模式 A: 只输出 score=1 的去重文档
        seen_md5 = set()
        for cid, text in corpus_map.items():
            if not text: continue
            text_hash = compute_md5(text)
            if text_hash not in seen_md5:
                results.append({"_id": cid, "text": text})
                seen_md5.add(text_hash)
        print(f"[INFO] Extracted {len(results)} unique positive documents.")

    else:
        # 模式 B: 输出 (query, text) 对
        if not queries_path:
            raise ValueError("Queries path is required for 'pair' mode.")
        
        query_map = _load_queries(queries_path, cfg)
        for qid, pid in qrels_records:
            q_text = query_map.get(qid)
            c_text = corpus_map.get(pid)
            if q_text and c_text:
                results.append({"query": q_text, "text": c_text})
        print(f"[INFO] Extracted {len(results)} query-passage pairs.")

    return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, choices=[t.name for t in TaskType])

    parser.add_argument("--language", type=str, default="en", choices=[l.name for l in Language])

    parser.add_argument("--item", type=str, default="corpus", choices=["corpus", "pair"], 
                        help="corpus: output score=1 docs; pair: output query-doc pairs")

    parser.add_argument("--corpus_path", type=str, default=None, help="Override default corpus path.")

    parser.add_argument("--qrels_path", type=str, default=None, help="Override default qrels path.")

    parser.add_argument("--queries_path", type=str, default=None, help="Override default queries path (required for 'pair' mode).")

    parser.add_argument("--output_path", type=str, default=None, help="Where to write the filtered corpus JSONL.")

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str, str, TaskDatasetConfig]:
    cfg = get_task_config(args.task)

    corpus_path = args.corpus_path or cfg.corpus_path
    qrels_path = args.qrels_path or cfg.qrels_path
    queries_path = args.queries_path or cfg.queries_path
    item = args.item
    if corpus_path is None or qrels_path is None:
        raise ValueError("Both corpus_path and qrels_path are required.")

    if args.output_path:
        output_path = args.output_path
    else:
        default_dir = os.path.join(
            DEFAULT_GENERATED_ROOT,
            args.task,
            "preparation",
        )
        os.makedirs(default_dir, exist_ok=True)
        if item == "corpus":
            output_path = os.path.join(default_dir, f"{args.language}_corpus_filtered.jsonl")
        else:
            output_path = os.path.join(default_dir, f"{args.language}_pair_filtered.jsonl")

    return corpus_path, qrels_path, queries_path, output_path, cfg


def main() -> None:
    args = parse_args()
    corpus_path, qrels_path, queries_path, output_path, cfg = resolve_paths(args)

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    filtered = process_data(
        args,
        cfg,
        corpus_path,    
        qrels_path,
        queries_path,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
