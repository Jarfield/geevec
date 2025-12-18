"""Download Hugging Face models with a single command.

This script leans on the environment variables that are commonly used for
mirrors and cache locations (HF_ENDPOINT, HUGGINGFACE_HUB_CACHE, BASE_MODEL_DIR
etc.). Only the repository ID is required for a basic download.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

from huggingface_hub import snapshot_download

DEFAULT_MODEL_ROOT = Path("/data/share/project/shared_models")


def _as_sequence(value: Optional[Sequence[str]]) -> Optional[Sequence[str]]:
    if not value:
        return None
    return value


def _default_local_dir(repo_id: str, override: Optional[str]) -> Path:
    if override:
        base_dir = Path(override)
    else:
        base_dir = DEFAULT_MODEL_ROOT / repo_id.replace("/", "__")
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model repository.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Repository ID, e.g., 'Qwen/Qwen2.5-3B'.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision (branch, tag, or commit hash).",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Override the directory to place the resolved files (default uses BASE_MODEL_DIR/hf_models/<repo>).",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HUGGINGFACE_HUB_CACHE"),
        help="Custom cache dir (defaults to $HUGGINGFACE_HUB_CACHE when set).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Authentication token; falls back to the HF_TOKEN env variable.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="Only download files that match these patterns.",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="+",
        default=None,
        help="Skip files that match these patterns.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallelism for downloads.",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="Resume from partial downloads when possible.",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Disable symlinked local dirs (forces full file copies).",
    )
    return parser.parse_args(argv)


def download_model(args: argparse.Namespace) -> Path:
    local_dir = _default_local_dir(args.repo_id, args.local_dir)
    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=not args.no_symlinks,
        allow_patterns=_as_sequence(args.allow_patterns),
        ignore_patterns=_as_sequence(args.ignore_patterns),
        resume_download=args.resume_download,
        max_workers=args.max_workers,
        token=args.token,
    )
    return Path(downloaded_path)


def main() -> None:
    args = _parse_args()
    downloaded_path = download_model(args)
    print(f"Model cached at: {downloaded_path}")
    print(f"Local directory: {downloaded_path.resolve()}")


if __name__ == "__main__":
    main()
