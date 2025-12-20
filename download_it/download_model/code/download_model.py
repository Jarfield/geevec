"""Download Hugging Face models with a single command.

Target directory: /data/share/project/shared_models/author__model_name
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

from huggingface_hub import snapshot_download

# 1. 修正后的确切根目录
DEFAULT_MODEL_ROOT = Path("/data/share/project/shared_models")


def _as_sequence(value: Optional[Sequence[str]]) -> Optional[Sequence[str]]:
    if not value:
        return None
    return value


def _default_local_dir(repo_id: str, override: Optional[str]) -> Path:
    """确定模型下载的具体物理路径：/shared_models/author__model"""
    if override:
        base_dir = Path(override)
    else:
        # 优先从环境变量读取根目录，如果没有则使用 DEFAULT_MODEL_ROOT
        # 确保不会再出现 /path/to/ 这种占位符
        base_root = Path(os.environ.get("BASE_MODEL_DIR", DEFAULT_MODEL_ROOT))
        
        # 保持你喜欢的冗长格式：nvidia/llama-8b -> nvidia__llama-8b
        folder_name = repo_id.replace("/", "__")
        base_dir = base_root / folder_name
    
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _check_hf_transfer():
    """检查 hf_transfer 是否可用，并根据需要自动降级"""
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
        try:
            import hf_transfer
        except ImportError:
            print("\n" + "!"*60)
            print("注意: 检测到 HF_HUB_ENABLE_HF_TRANSFER=1 但未安装 hf_transfer 包。")
            print("正在自动切换回标准下载模式。")
            print("!"*60 + "\n")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model repository.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Repository ID, e.g., 'nvidia/llama-embed-nemotron-8b'.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision (branch, tag, or commit hash).",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Override the directory to place the files.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HUGGINGFACE_HUB_CACHE"),
        help="Custom cache dir (defaults to $HUGGINGFACE_HUB_CACHE).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Authentication token for gated models.",
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
        default=8,
        help="Parallelism for downloads.",
    )
    return parser.parse_args(argv)


def download_model(args: argparse.Namespace) -> Path:
    _check_hf_transfer()
    
    # 路径计算
    local_dir = _default_local_dir(args.repo_id, args.local_dir)
    
    print(f"🚀 开始下载至: {local_dir}")
    
    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=local_dir,
        allow_patterns=_as_sequence(args.allow_patterns),
        ignore_patterns=_as_sequence(args.ignore_patterns),
        max_workers=args.max_workers,
        token=args.token,
    )
    return Path(downloaded_path)


def main() -> None:
    args = _parse_args()
    try:
        downloaded_path = download_model(args)
        print(f"\n✅ 下载成功!")
        print(f"绝对路径: {downloaded_path.resolve()}")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()