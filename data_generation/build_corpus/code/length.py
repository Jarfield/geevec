#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import jsonlines
from tqdm import tqdm
import matplotlib.pyplot as plt


def iter_jsonl_files(input_file: str | None, input_dir: str | None) -> list[Path]:
    """根据参数返回要处理的 jsonl 文件列表。"""
    if input_file:
        return [Path(input_file)]
    else:
        d = Path(input_dir)
        files = sorted(d.glob("*.jsonl"))
        if not files:
            raise ValueError(f"No .jsonl files found in directory: {d}")
        return files


def collect_body_lengths(jsonl_files: list[Path], metric: str = "chars") -> list[int]:
    """
    遍历多个 jsonl 文件，收集每条 doc 的 body 长度。

    metric:
      - "chars": 按字符数计算 len(body)
      - "words": 按空格切分后的词数
    """
    lengths: list[int] = []

    for f in tqdm(jsonl_files, desc="Reading jsonl files"):
        with jsonlines.open(f, "r") as reader:
            for doc in reader:
                body = doc.get("text", "") or ""
                if metric == "chars":
                    lengths.append(len(body))
                elif metric == "words":
                    lengths.append(len(body.split()))
                else:
                    raise ValueError(f"Unknown metric: {metric}")

    return lengths


def plot_lengths(
    lengths: list[int],
    output_image: str,
    metric: str = "chars",
    downsample: int | None = None,
) -> None:
    """
    画出长度曲线，并保存为图片。

    downsample: 若指定为 N，则每隔 N 个点取一个，避免 3万+ 点太挤。
    """
    if not lengths:
        raise ValueError("No lengths collected.")

    if downsample and downsample > 1:
        xs = list(range(0, len(lengths), downsample))
        ys = [lengths[i] for i in xs]
    else:
        xs = list(range(len(lengths)))
        ys = lengths

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, linewidth=0.8)
    plt.xlabel("Document index")
    if metric == "chars":
        plt.ylabel("Body length (chars)")
    else:
        plt.ylabel("Body length (words)")
    plt.title("Body length curve")

    plt.tight_layout()
    plt.savefig(output_image, dpi=200)
    plt.close()
    print(f"Saved plot to {output_image}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot body length curve from UK-LEX jsonl(s)."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_file",
        type=str,
        help="单个 jsonl 文件路径",
    )
    group.add_argument(
        "--input_dir",
        type=str,
        help="包含多个 jsonl 的目录（处理其中所有 *.jsonl）",
    )

    parser.add_argument(
        "--output_image",
        type=str,
        required=True,
        help="输出图片路径，例如 body_lengths.png",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["chars", "words"],
        default="chars",
        help="长度度量方式：chars=按字符数，words=按词数（默认 chars）",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=0,
        help="下采样间隔（>1 时启用，如 10 表示每 10 条取一个点）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    files = iter_jsonl_files(args.input_file, args.input_dir)
    lengths = collect_body_lengths(files, metric=args.metric)
    down = args.downsample if args.downsample > 1 else None

    plot_lengths(
        lengths=lengths,
        output_image=args.output_image,
        metric=args.metric,
        downsample=down,
    )
