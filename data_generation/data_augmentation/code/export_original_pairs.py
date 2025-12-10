"""从原始元数据直接构建训练三元组文件。

输入：原始 corpus、queries 与 qrels（score=1）文件，字段名可通过参数覆盖。
输出：形如 ``{"prompt": str, "query": str, "pos": [str], "neg": []}`` 的 JSONL，
      默认写入 ``DATA_AUG_ORIGINAL_ROOT/<task>/original_pairs/<language>_original.jsonl``。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Set

from datasets import Dataset, load_dataset
from tqdm import tqdm

from constant import Language, TaskType
from task_configs import DEFAULT_ORIGINAL_ROOT, TaskDatasetConfig, get_task_config


def _load_dataset(path: str) -> Dataset:
    """按照扩展名加载数据文件，支持 jsonl 与 Arrow。"""

    if path.endswith(".jsonl"):
        return load_dataset("json", data_files=path)["train"]
    return Dataset.from_file(path)


def _to_lookup(
    ds: Iterable[dict],
    id_key: str,
    text_key: str,
    title_key: Optional[str],
    min_len: int,
) -> Dict[str, str]:
    """将 Dataset 转成 id -> 文本 的查找表。"""

    lookup: Dict[str, str] = {}
    for row in tqdm(ds, desc="Loading records"):
        rid = row.get(id_key)
        text = row.get(text_key)
        title = row.get(title_key) if title_key else None

        if text is None:
            continue
        merged_text = f"{title}\n{text}" if title else text
        if len(merged_text.strip()) < min_len:
            continue

        lookup[str(rid)] = merged_text
    return lookup


def _load_positive_map(
    ds: Iterable[dict],
    qid_key: str,
    pid_key: str,
    score_key: str,
    min_positive: int,
) -> Dict[str, Set[str]]:
    """读取 qrels，收集满足分数阈值的 query->doc 映射。"""

    pos_map: Dict[str, Set[str]] = {}
    for row in tqdm(ds, desc="Loading qrels"):
        try:
            score = int(row.get(score_key, 0))
        except Exception:
            continue
        if score < min_positive:
            continue

        qid = str(row.get(qid_key))
        pid = str(row.get(pid_key))
        if qid not in pos_map:
            pos_map[qid] = set()
        pos_map[qid].add(pid)
    return pos_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert metadata into training-ready triplets")

    parser.add_argument("--task_type", required=True, choices=[t.name for t in TaskType], help="任务名")
    parser.add_argument("--language", default="en", choices=[l.name for l in Language], help="语言代码")

    parser.add_argument("--corpus_path", default=None, help="自定义 corpus 路径，默认读取 task_configs")
    parser.add_argument("--queries_path", default=None, help="自定义 queries 路径，默认读取 task_configs")
    parser.add_argument("--qrels_path", default=None, help="自定义 qrels 路径，默认读取 task_configs")

    parser.add_argument("--corpus_id_key", default=None, help="corpus 主键字段名，默认取 task_configs.id_key")
    parser.add_argument("--corpus_text_key", default=None, help="corpus 文本字段名，默认取 task_configs.text_key")
    parser.add_argument("--corpus_title_key", default=None, help="corpus 标题字段名，可选，存在时会与正文拼接")
    parser.add_argument("--query_id_key", default=None, help="queries 主键字段名，默认取 task_configs.query_id_key")
    parser.add_argument("--query_text_key", default=None, help="queries 文本字段名，默认取 task_configs.query_text_key")
    parser.add_argument("--qrels_qid_key", default=None, help="qrels 中 query id 字段名，默认取 task_configs.qrels_qid_key")
    parser.add_argument("--qrels_pid_key", default=None, help="qrels 中 doc id 字段名，默认取 task_configs.qrels_pid_key")
    parser.add_argument("--qrels_score_key", default=None, help="qrels 中分数字段名，默认取 task_configs.qrels_score_key")

    parser.add_argument("--positive_score", type=int, default=1, help="认为是正例的最小 score 阈值，默认 1")
    parser.add_argument("--min_len", type=int, default=None, help="过滤过短文本的最小长度，默认取 task_configs.min_len")
    parser.add_argument("--max_queries", type=int, default=-1, help="最多保留多少条 query，-1 表示不过滤")

    parser.add_argument("--save_root", default=None, help="输出根目录，默认 DATA_AUG_ORIGINAL_ROOT")
    parser.add_argument(
        "--output_path",
        default=None,
        help="完整输出路径；若提供则忽略 save_root/language 组合文件名",
    )
    parser.add_argument("--overwrite", action="store_true", help="若文件存在是否覆盖")
    parser.add_argument("--prompt", default="", help="输出中的 prompt 字段值，默认空字符串")

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace, cfg: TaskDatasetConfig) -> tuple[str, str, str]:
    corpus_path = args.corpus_path or cfg.corpus_path
    queries_path = args.queries_path or cfg.queries_path
    qrels_path = args.qrels_path or cfg.qrels_path

    if corpus_path is None:
        raise ValueError("corpus_path 未提供，请通过参数或 task_configs 配置")
    if queries_path is None:
        raise ValueError("queries_path 未提供，请通过参数或 task_configs 配置")
    if qrels_path is None:
        raise ValueError("qrels_path 未提供，请通过参数或 task_configs 配置")
    return corpus_path, queries_path, qrels_path


def main():
    args = parse_args()

    cfg = get_task_config(args.task_type)
    corpus_path, queries_path, qrels_path = resolve_paths(args, cfg)

    corpus_id_key = args.corpus_id_key or cfg.id_key
    corpus_text_key = args.corpus_text_key or cfg.text_key
    corpus_title_key = args.corpus_title_key or cfg.title_key
    query_id_key = args.query_id_key or cfg.query_id_key
    query_text_key = args.query_text_key or cfg.query_text_key
    qrels_qid_key = args.qrels_qid_key or cfg.qrels_qid_key
    qrels_pid_key = args.qrels_pid_key or cfg.qrels_pid_key
    qrels_score_key = args.qrels_score_key or cfg.qrels_score_key
    min_len = args.min_len if args.min_len is not None else cfg.min_len

    # ---------- 加载数据 ----------
    corpus_ds = _load_dataset(corpus_path)
    queries_ds = _load_dataset(queries_path)
    qrels_ds = _load_dataset(qrels_path)

    corpus_lookup = _to_lookup(
        corpus_ds,
        id_key=corpus_id_key,
        text_key=corpus_text_key,
        title_key=corpus_title_key,
        min_len=min_len,
    )
    query_lookup = _to_lookup(
        queries_ds,
        id_key=query_id_key,
        text_key=query_text_key,
        title_key=None,
        min_len=0,
    )
    pos_map = _load_positive_map(
        qrels_ds,
        qid_key=qrels_qid_key,
        pid_key=qrels_pid_key,
        score_key=qrels_score_key,
        min_positive=args.positive_score,
    )

    print(f"[INFO] Loaded corpus docs: {len(corpus_lookup)} | queries: {len(query_lookup)} | qrels pairs: {len(pos_map)}")

    # ---------- 组装输出 ----------
    results: List[dict] = []
    for idx, (qid, doc_ids) in enumerate(pos_map.items()):
        if 0 <= args.max_queries <= idx:
            break
        query_text = query_lookup.get(qid)
        if not query_text:
            continue
        pos_texts = [corpus_lookup[pid] for pid in doc_ids if pid in corpus_lookup]
        if not pos_texts:
            continue
        results.append(
            {
                "prompt": args.prompt,
                "query": query_text,
                "pos": sorted(set(pos_texts)),
                "neg": [],
            }
        )

    save_root = args.save_root or DEFAULT_ORIGINAL_ROOT
    if args.output_path:
        save_path = args.output_path
    else:
        save_dir = os.path.join(save_root, args.task_type, "original_pairs")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{args.language}_original.jsonl")

    if os.path.exists(save_path) and not args.overwrite:
        raise FileExistsError(f"文件已存在且未指定 --overwrite: {save_path}")

    with open(save_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved {len(results)} records to {save_path}")


if __name__ == "__main__":
    main()
