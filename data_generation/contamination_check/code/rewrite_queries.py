import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, Iterator, Optional
from tqdm import tqdm

# 环境路径配置
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_generation.shared.llm import LLM
from data_generation.shared.constants import Language, TaskType
from .prompts import build_rewrite_prompt

def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if s := line.strip():
                yield json.loads(s)

def postprocess_rewrite(text: str) -> str:
    """清理输出：取首行、剥离引号。"""
    # 取第一行非空内容
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    t = lines[0] if lines else ""
    
    # 剥离常见的包裹引号
    return t.strip("'\"“”` ")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite queries via vLLM.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--model", type=str, default="Qwen2-5-72B-Instruct")
    parser.add_argument("--model_type", type=str, default="open-source", choices=["open-source", "azure", "openai"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task_type", type=str, required=True, choices=[t.name for t in TaskType])
    parser.add_argument("--language", type=str, required=True, choices=[l.name for l in Language])
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--passthrough", action="store_true", help="Keep original fields.")
    parser.add_argument("--drop_suspicious", action="store_true")
    return parser.parse_args()

def rewrite_dataset(
    llm: LLM,
    samples: Iterable[Dict[str, Any]],
    output_path: str,
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    task_type = TaskType[args.task_type]
    language = Language[args.language]

    stats = {"total": 0, "saved": 0, "dropped": 0}

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(samples, desc="Rewriting"):
            if 0 < args.limit <= stats["total"]: break
            stats["total"] += 1

            # 直接提取内容
            orig_q = sample.get("query", "").strip()
            orig_c = sample.get("text", "").strip() or None
            if not orig_q: continue

            # LLM 推理
            prompt = build_rewrite_prompt(task_type, language, orig_q)
            raw_outputs = llm.chat(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
            
            rewritten = postprocess_rewrite(raw_outputs[0]) if raw_outputs else ""
            
            # 核心检查逻辑
            is_same = rewritten == orig_q
            is_echo = any(kw in rewritten for kw in ["Original query", "Rewritten query"])
            
            if args.drop_suspicious and (not rewritten or is_same or is_echo):
                stats["dropped"] += 1
                continue

            # 组装输出
            out = dict(sample) if args.passthrough else {}
            out.update({
                "rewritten_query": rewritten,
                "original_query": orig_q,
                "original_corpus": orig_c,
                "rewrite_checks": {
                    "is_same_as_original": is_same,
                    "contains_prompt_echo": is_echo
                }
            })
            
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["saved"] += 1

    print(f"Done. Saved {stats['saved']}/{stats['total']} (Dropped: {stats['dropped']})")

def main():
    args = parse_args()
    if args.output_path is None:
        args.output_path = os.path.splitext(args.input_path)[0] + ".rewritten.jsonl"

    llm = LLM(model=args.model, model_type=args.model_type, port=args.port)
    rewrite_dataset(llm, iter_jsonl(args.input_path), args.output_path, args)

if __name__ == "__main__":
    main()