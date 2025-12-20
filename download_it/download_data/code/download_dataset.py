"""Download Hugging Face datasets with a single command.

Standardizes dataset storage using BASE_DATA_DIR and HF_DATASETS_CACHE.
Supports automatic fallbacks and directory management.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from datasets import load_dataset

# 默认数据集存放根目录
DEFAULT_DATASET_ROOT = Path("/data/share/project/shared_datasets")


def _as_list(items: Optional[Sequence[str]]) -> List[str]:
    if not items:
        return []
    return list(items)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g., 'mteb/covid_retrieval' or 'glue'.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional subset/config name (e.g., 'wikitext-2-raw-v1').",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="One or more splits to fetch (e.g., train validation test).",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_DATASETS_CACHE"),
        help="Defaults to $HF_DATASETS_CACHE or BASE_DATA_DIR/hf_datasets.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision (branch, tag, or commit hash).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Authentication token for gated datasets.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        # 默认设为 True 可能更方便，但为了安全性，这里保留参数显式调用
        help="Allow executing dataset scripts that require trust.",
    )
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help="Optional custom data files for datasets that support them.",
    )
    return parser.parse_args(argv)


def download_dataset(args: argparse.Namespace) -> None:
    # 1. 路径优先级逻辑：参数指定 > 环境变量 > 默认硬编码
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        # 尝试从之前 bashrc 定义的环境变量中获取
        base_data_root = os.environ.get("BASE_DATA_DIR")
        if base_data_root:
            cache_dir = Path(base_data_root) / "hf_datasets"
        else:
            cache_dir = DEFAULT_DATASET_ROOT / "hf_datasets"

    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 确定需要下载的 split
    # 如果用户没指定，默认下载全部（load_dataset 不传 split 会下载全部）
    # 但为了明确反馈，我们通过循环来处理用户指定的特定 splits
    target_splits = _as_list(args.splits)
    
    try:
        if target_splits:
            for split in target_splits:
                print(f"🚀 Downloading split: '{split}' for dataset '{args.dataset}'...")
                load_dataset(
                    path=args.dataset,
                    name=args.subset,
                    split=split,
                    cache_dir=str(cache_dir),
                    revision=args.revision,
                    token=args.token,
                    trust_remote_code=args.trust_remote_code,
                    data_files=args.data_files,
                )
        else:
            print(f"🚀 Downloading all splits for dataset '{args.dataset}'...")
            load_dataset(
                path=args.dataset,
                name=args.subset,
                cache_dir=str(cache_dir),
                revision=args.revision,
                token=args.token,
                trust_remote_code=args.trust_remote_code,
                data_files=args.data_files,
            )
        
        print(f"\n✅ Dataset successfully cached at: {cache_dir.resolve()}")

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        if "trust_remote_code" in str(e).lower():
            print("\n💡 HINT: This dataset requires trusting remote code. Try adding '--trust-remote-code'.")
        sys.exit(1)


def main() -> None:
    args = _parse_args()
    download_dataset(args)


if __name__ == "__main__":
    main()