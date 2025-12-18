#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage A (Part 2): Topic-anchor filtering using vLLM

输入：上一步导出的 base JSONL
输出：过滤后的 JSONL（同结构）

核心思想：
1) 从 scirepeval 的 SCIDOCS eval 配置里抽样若干篇 title+abstract
2) 调 vLLM 总结关键词（anchor vocab）
3) 对每条样本：如果 query(title) 与 anchor vocab 的 token overlap >= min_anchor_overlap，则保留；否则丢弃
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, List, Set

from tqdm import tqdm

# -----------------------------
# 1) 文本处理工具
# -----------------------------
def _norm(x: str) -> str:
    return (x or "").strip()

def _tokenize_en(text: str, min_len: int = 3) -> List[str]:
    """
    简单分词：保留字母，转小写。
    """
    return [t for t in re.findall(r"[A-Za-z]+", text.lower()) if len(t) >= min_len]

def _passes_anchor(query_text: str, anchor_vocab: Set[str], min_overlap: int, min_token_len: int) -> bool:
    """宽松匹配：只要 query 里有 min_overlap 个词在 anchor_vocab 里就算过"""
    if min_overlap <= 0 or not anchor_vocab:
        return True
    
    # 将 Query 分词
    q_toks = set(_tokenize_en(query_text, min_len=min_token_len))
    
    # 求交集
    intersection = q_toks.intersection(anchor_vocab)
    return len(intersection) >= min_overlap

# -----------------------------
# 2) 读取测试集 (Anchor Source)
# -----------------------------
def sample_from_jsonl_pos(test_path: str, max_docs: int, seed: int) -> List[str]:
    """
    从测试集 JSONL 中读取 'pos' 字段的内容。
    结构: {"pos": ["Abstract 1...", "Abstract 2..."], ...}
    """
    print(f"Loading test anchors from: {test_path} ...")
    samples = []
    
    # 蓄水池采样 (防止文件太大撑爆内存)
    rng = random.Random(seed)
    
    with open(test_path, 'r', encoding='utf-8') as f:
        # 这里的 desc 会显示在屏幕上，让你知道正在读文件
        lines = [line for line in tqdm(f, desc="Reading Test File")]
        
    # 如果文件太大，可以改为流式读取；这里假设测试集一般只有几千条，readlines 没问题
    # 随机打乱行顺序
    print("Shuffling test lines...")
    rng.shuffle(lines)
    
    for line in lines:
        if len(samples) >= max_docs:
            break
        try:
            row = json.loads(line)
            # 提取 pos 列表里的内容
            pos_list = row.get("pos", [])
            if not pos_list:
                continue
            
            # 拼接该样本下所有的 pos abstract，形成一个大的 context
            # 或者随机选一个 pos。为了丰富性，我们把它们拼在一起处理，或者分开。
            # 这里策略：把 pos 里的每一个文本都当作一个独立的 sample 加进去
            for p_text in pos_list:
                if len(samples) >= max_docs: 
                    break
                if len(p_text) > 50: # 过滤太短的
                    samples.append(p_text[:1200]) # 截断一下防止 token 溢出
                    
        except Exception:
            continue
            
    print(f"Sampled {len(samples)} text snippets from test set.")
    return samples

# -----------------------------
# 3) vLLM 总结 (Macro/Broad Mode)
# -----------------------------
def _ensure_openai():
    # -------------------------------------------------------
    import importlib.util  # <--- 必须加上这一行！
    # -------------------------------------------------------
    if importlib.util.find_spec("openai") is None:
        raise RuntimeError("Need `openai>=1.0`. Install: pip install openai>=1.0")
    import openai
    return openai

def summarize_broad_topics(
    text_samples: List[str],
    model: str,
    endpoint: str,
    keywords_per_chunk: int,
    min_token_len: int,
) -> Set[str]:
    if not text_samples:
        return set()
    
    openai = _ensure_openai()
    client = openai.OpenAI(base_url=endpoint, api_key="EMPTY")
    
    anchor: Set[str] = set()
    # 稍微加大 chunk size，让模型看更多文档来总结共性
    chunk_size = 5 

    print(f"Requesting vLLM ({model}) for BROAD topic summary...")

    for i in tqdm(range(0, len(text_samples), chunk_size), desc="vLLM Generating Anchors"):
        chunk_text = "\n---\n".join(text_samples[i : i + chunk_size])
        
        # -------------------------------------------------------------
        # 【关键修改】Prompt Engineering: 要求宏大、丰富、高层级
        # -------------------------------------------------------------
        system_prompt = (
            "You are a senior scientific editor with a broad perspective on research fields. "
            "Your task is to identify the **broad research domains**, **general methodologies**, "
            "and **high-level themes** covered in these abstracts.\n\n"
            "Do NOT limit yourself to narrow, specific entities. "
            "Instead, include broader categories (e.g., if you see 'LSTM', add 'Deep Learning', 'Neural Networks', 'Time Series'). "
            "Output a rich, comprehensive list of keywords and phrases."
        )
        
        user_prompt = (
            f"Abstracts:\n{chunk_text}\n\n"
            f"Return a comma-separated list of {keywords_per_chunk} keywords/phrases (lower case)."
        )
        
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4, # 稍微调高温度，增加丰富度
                max_tokens=300,
            )
            content = (resp.choices[0].message.content or "").lower()
            
            # 解析结果
            for phrase in re.split(r"[,\n]+", content):
                phrase = phrase.strip()
                if not phrase: continue
                
                # 1. 加入短语本身 (允许包含空格，如 "deep learning")
                # 只有当短语本身不太长时才加入，防止模型吐出句子
                if len(phrase.split()) <= 4: 
                    # 再次清洗一下，去掉非字母字符
                    clean_phrase = " ".join(_tokenize_en(phrase, min_len=1))
                    if len(clean_phrase) >= min_token_len:
                        anchor.add(clean_phrase)
                
                # 2. 同时也把短语拆碎了加进去 (增加 recall)
                # 例如 "deep learning" -> add "deep", add "learning"
                for tok in _tokenize_en(phrase, min_len=min_token_len):
                    anchor.add(tok)
                    
        except Exception as e:
            print(f"vLLM Error: {e}")
            continue

    return anchor

# -----------------------------
# 4) 主流程
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter Train Set using Macro Topics form Test Set")

    # 路径
    p.add_argument("--train_path", required=True, help="输入: 训练集 JSONL")
    p.add_argument("--test_path", required=True, help="输入: 测试集 JSONL (提取 anchor)")
    p.add_argument("--output_path", required=True, help="输出: 过滤后的 JSONL")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # vLLM
    p.add_argument("--model", required=True, help="vLLM 模型名称")
    p.add_argument("--endpoint", default=os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1"))
    p.add_argument("--max_test_samples", type=int, default=100, help="抽样多少条 pos 用于总结")
    p.add_argument("--keywords_per_chunk", type=int, default=60, help="每批次让 LLM 生成多少词 (越多越丰富)")

    # 过滤
    p.add_argument("--min_overlap", type=int, default=1, help="Query 命中多少个 anchor 词才保留")
    p.add_argument("--min_token_len", type=int, default=3)

    return p.parse_args()

def main():
    args = parse_args()
    
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path}")

    # 1. 提取 Anchor Vocab
    print(f"=== Step 1: Macro Topic Extraction ===")
    test_samples = sample_from_jsonl_pos(args.test_path, args.max_test_samples, args.seed)
    
    anchor_vocab = summarize_broad_topics(
        test_samples,
        args.model,
        args.endpoint,
        keywords_per_chunk=args.keywords_per_chunk,
        min_token_len=args.min_token_len
    )
    
    if not anchor_vocab:
        print("Error: Anchor vocab is empty. Check vLLM connection.")
        return

    print(f"\n[Anchor Stats] Size: {len(anchor_vocab)} unique tokens/phrases")
    print(f"[Preview] {list(anchor_vocab)[:15]} ...\n")

    # 2. 过滤
    print(f"=== Step 2: Filtering Training Data ===")
    train_path = Path(args.train_path)
    kept = 0
    dropped = 0
    total_lines = sum(1 for _ in open(train_path, 'r', encoding='utf-8'))

    with open(train_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, total=total_lines, desc="Filtering"):
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
            except:
                continue

            # 获取 Query 文本
            # 你的训练集 query 字段如果是字符串直接用，如果是 dict 则取 title
            q_obj = rec.get("query")
            q_text = ""
            if isinstance(q_obj, str):
                q_text = q_obj
            elif isinstance(q_obj, dict):
                q_text = q_obj.get("title", "") or q_obj.get("text", "")
            
            # 核心判断
            if _passes_anchor(q_text, anchor_vocab, args.min_overlap, args.min_token_len):
                fout.write(line + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"\n[DONE]")
    print(f"Output: {out_path}")
    print(f"Kept: {kept} / Total: {kept+dropped} ({kept/(kept+dropped+1e-5):.2%})")

if __name__ == "__main__":
    main()