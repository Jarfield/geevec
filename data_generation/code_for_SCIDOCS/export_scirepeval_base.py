"""
Stage A (Part 1): Export base training triples from allenai/scirepeval:cite_prediction_new

你要求的 base 规则（逐条样本，不聚合，不扩充）：
- query = row["query"]["title"]
- pos   = [row["pos"]["abstract"]]
- neg   = [row["neg"]["abstract"]]

输出 JSONL 每行结构：
{
  "query": <str>,
  "pos": [<str>],
  "neg": [<str>],
  "source_ids": {"query_id": <str>, "pos_id": <str>, "neg_id": <str>},
  "doc_pool": {<id>: <text>, ...}
}

注意：
- 如果某条缺 title/abstract/id 或 pos/neg abstract 太短，会被丢弃（可调 min_abs_len）。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm


# -----------------------------
# 1) 工具函数 (保持不变)
# -----------------------------
def _norm(x: Optional[str]) -> str:
    return (x or "").strip()


def _get_title_abs_id(struct_obj: Any) -> Tuple[str, str, Optional[str]]:
    if not isinstance(struct_obj, dict):
        return "", "", None

    title = _norm(struct_obj.get("title"))
    abstract = _norm(struct_obj.get("abstract"))

    cid = struct_obj.get("corpus_id")
    if cid is None:
        for k in ("paper_id", "id", "mag_id", "pid"):
            if struct_obj.get(k) is not None:
                cid = struct_obj.get(k)
                break

    return title, abstract, (str(cid) if cid is not None else None)


# -----------------------------
# 2) 参数配置 (保持不变)
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export base SCIDOCS-like JSONL from allenai/scirepeval")
    p.add_argument("--save_root", default=os.environ.get("DATA_AUG_ORIGINAL_ROOT", "./original_data"))
    p.add_argument("--output_path", default=None)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--min_abs_len", type=int, default=200)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    return Path(args.save_root) / "scidocs" / "scirep_citation_train" / "en_scirep.base.jsonl"


# -----------------------------
# 3) 主流程：完全流式处理
# -----------------------------
def main() -> None:
    args = parse_args()
    out_path = resolve_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite to replace)")

    print(f"Loading dataset 'allenai/scirepeval' (split='train')...")
    # 这里加载并不会读取数据，只是读取了元数据，内存占用极小
    ds = load_dataset("allenai/scirepeval", "cite_prediction_new", split="train")

    print("Shuffling dataset (Lazy operation)...")
    # 【关键修改1】使用 HF 原生的 shuffle
    # 这不会打乱内存中的数据，而是创建一个打乱的索引映射，速度极快且不占内存
    ds = ds.shuffle(seed=args.seed)

    # 【关键修改2】如果指定了条数，直接取前 N 条
    # select 也是懒执行的，不会触发读取
    if args.num_samples > 0:
        print(f"Selecting top {args.num_samples} samples after shuffle...")
        limit = min(len(ds), args.num_samples)
        ds = ds.select(range(limit))

    kept = 0
    dropped_missing = 0
    dropped_short = 0

    print(f"Start processing {len(ds)} rows...")
    print(f" -> Output: {out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        # 【关键修改3】直接迭代 ds 对象，不要转 list(ds)
        # 这样就是读一行、处理一行、写一行、扔一行。内存永远平稳。
        with tqdm(ds, desc="Exporting") as pbar:
            for row in pbar:
                q_title, q_abs, qid = _get_title_abs_id(row.get("query"))
                p_title, p_abs, pid = _get_title_abs_id(row.get("pos"))
                n_title, n_abs, nid = _get_title_abs_id(row.get("neg"))

                # 1. 检查缺字段
                if not qid or not pid or not nid:
                    dropped_missing += 1
                    continue
                if not q_title or not p_abs or not n_abs:
                    dropped_missing += 1
                    continue

                # 2. 检查长度
                if len(p_abs) < args.min_abs_len or len(n_abs) < args.min_abs_len:
                    dropped_short += 1
                    continue

                rec: Dict[str, Any] = {
                    "query": q_title,
                    "pos": [p_abs],
                    "neg": [n_abs],
                    "source_ids": {
                        "query_id": qid,
                        "pos_id": pid,
                        "neg_id": nid,
                    },
                    "doc_pool": {
                        qid: q_title,
                        pid: p_abs,
                        nid: n_abs,
                    },
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

                if kept % 100 == 0:
                    pbar.set_postfix({
                        "kept": kept, 
                        "drop_miss": dropped_missing, 
                        "drop_short": dropped_short
                    })
                
                # 注意：由于我们已经在外面做了 ds.select，这里理论上不需要 break，
                # 但如果你的逻辑是“要保留够N条有效数据”而不是“只处理N条原始数据”，
                # 可以在这里加逻辑。现在的逻辑是处理完 select 的那批数据。

    print(f"\n[DONE] Saved {kept} samples to: {out_path}")
    print(f"       Stats: kept={kept}, dropped_missing={dropped_missing}, dropped_short={dropped_short}")


if __name__ == "__main__":
    main()