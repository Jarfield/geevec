"""Download Hugging Face datasets with a single command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from datasets import load_dataset


def _as_list(items: Optional[Sequence[str]]) -> List[str]:
    if not items:
        return []
    return list(items)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g., 'wikitext' or 'glue'.",
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
        help="One or more splits to fetch (defaults to 'train').",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_DATASETS_CACHE"),
        help="Cache directory for datasets; defaults to $HF_DATASETS_CACHE when set.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision (branch, tag, or commit hash).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Authentication token; falls back to the HF_TOKEN env variable.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
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
    splits = _as_list(args.splits) or ["train"]
    for split in splits:
        print(f"Downloading split '{split}' for dataset '{args.dataset}'...")
        load_dataset(
            path=args.dataset,
            name=args.subset,
            split=split,
            cache_dir=args.cache_dir,
            revision=args.revision,
            token=args.token,
            trust_remote_code=args.trust_remote_code,
            data_files=args.data_files,
        )
    resolved_cache = Path(args.cache_dir) if args.cache_dir else Path.home()
    print(f"Datasets cached under: {resolved_cache.resolve()}")


def main() -> None:
    args = _parse_args()
    download_dataset(args)


if __name__ == "__main__":
    main()
