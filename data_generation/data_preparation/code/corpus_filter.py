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


def _load_qrels_ids(qrels_path: str, cfg: TaskDatasetConfig) -> Set[str]:
    """Read positive doc ids from qrels."""

    if qrels_path.endswith(".jsonl"):
        records = (json.loads(line) for line in open(qrels_path, "r", encoding="utf-8"))
    else:
        records = Dataset.from_file(qrels_path)

    pos_ids: Set[str] = set()
    for row in records:
        try:
            score = int(row.get(cfg.qrels_score_key, 0))
        except Exception:
            continue
        if score <= 0:
            continue
        pid = row.get(cfg.qrels_pid_key)
        if pid is not None:
            pos_ids.add(str(pid))
    return pos_ids


def _load_corpus(corpus_path: str, cfg: TaskDatasetConfig) -> Iterable[dict]:
    if corpus_path.endswith(".jsonl"):
        yield from (json.loads(line) for line in open(corpus_path, "r", encoding="utf-8"))
        return
    ds = Dataset.from_file(corpus_path)
    for row in ds:
        yield row


def filter_corpus(
    corpus_path: str,
    qrels_path: str,
    cfg: TaskDatasetConfig,
) -> list[dict]:
    pos_ids = _load_qrels_ids(qrels_path, cfg)
    print(f"[INFO] Loaded {len(pos_ids)} positive doc ids from qrels.")

    kept = []
    seen_md5 = set()
    for row in tqdm(_load_corpus(corpus_path, cfg), desc="Filtering corpus"):
        cid = row.get(cfg.id_key)
        if cid is None or str(cid) in pos_ids:
            continue

        text = row.get(cfg.text_key)
        if text is None:
            continue

        text_hash = compute_md5(text)
        if text_hash in seen_md5:
            continue
        seen_md5.add(text_hash)

        kept.append({"_id": cid, "text": text})

    print(f"[INFO] Kept {len(kept)} documents after removing positives and duplicates.")
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolate synthesis corpus by removing evaluation positives.")
    parser.add_argument("--task", type=str, required=True, choices=[t.name for t in TaskType])
    parser.add_argument("--language", type=str, default="en", choices=[l.name for l in Language])
    parser.add_argument("--corpus_path", type=str, default=None, help="Override default corpus path.")
    parser.add_argument("--qrels_path", type=str, default=None, help="Override default qrels path.")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to write the filtered corpus JSONL.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, str, str, TaskDatasetConfig]:
    cfg = get_task_config(args.task)

    corpus_path = args.corpus_path or cfg.corpus_path
    qrels_path = args.qrels_path or cfg.qrels_path
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
        output_path = os.path.join(default_dir, f"{args.language}_corpus_filtered.jsonl")

    return corpus_path, qrels_path, output_path, cfg


def main() -> None:
    args = parse_args()
    corpus_path, qrels_path, output_path, cfg = resolve_paths(args)

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    filtered = filter_corpus(
        corpus_path=corpus_path,
        qrels_path=qrels_path,
        cfg=cfg,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote filtered corpus to {output_path}")


if __name__ == "__main__":
    main()
