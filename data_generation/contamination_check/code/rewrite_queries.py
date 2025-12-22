"""Rewrite queries for contamination checks using a vLLM endpoint."""

import argparse
import json
import os
import sys
from typing import Dict, Iterable, Iterator, Optional

from tqdm import tqdm

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_generation.shared.llm import LLM  # type: ignore

from .prompts import build_rewrite_prompt


def iter_jsonl(path: str) -> Iterator[Dict]:
    """Yield dictionaries from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite queries via vLLM.")
    parser.add_argument(
        "--input_path",
        type=str,
        default="/data/share/project/psjin/data/generated_data/task/preparation/en_pair_filtered.jsonl",
        help="Path to the input JSONL containing queries under the 'query' key.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/share/project/psjin/data/generated_data/task/contamination_check/rewrite_results.jsonl",
        help="Where to save the rewritten queries JSONL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2-5-72B-Instruct",
        help="Model name served by vLLM.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="open-source",
        choices=["open-source", "azure", "openai"],
        help="Model provider type.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the vLLM/OpenAI-compatible server.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate for each rewrite.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional cap on how many queries to process. -1 uses all.",
    )
    return parser.parse_args()


def rewrite_single_query(
    llm: LLM,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Optional[str]:
    outputs = llm.chat(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    rewritten = outputs[0] if outputs else None
    if rewritten is None:
        return None
    return rewritten.strip()


def rewrite_dataset(
    llm: LLM,
    samples: Iterable[Dict],
    output_path: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    limit: int,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed = 0
    saved = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc="Rewriting queries"):
            if limit > 0 and processed >= limit:
                break

            processed += 1
            query = sample.get("query")
            if not query:
                continue

            prompt = build_rewrite_prompt(query)
            rewritten = rewrite_single_query(
                llm=llm,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if not rewritten:
                continue

            enriched = dict(sample)
            enriched["rewrite_prompt"] = prompt
            enriched["rewritten_query"] = rewritten

            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            saved += 1

    print(f"Processed {processed} queries; saved {saved} rewrites to {output_path}.")


def main():
    args = parse_args()
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")

    llm = LLM(model=args.model, model_type=args.model_type, port=args.port)
    samples = iter_jsonl(args.input_path)

    rewrite_dataset(
        llm,
        samples,
        args.output_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
