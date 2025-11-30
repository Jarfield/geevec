#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
切分逻辑概要：
1. 对 body 识别 PART / SCHEDULE / 大写行等为章节标题，插入 "## " 形成伪 Markdown。
2. 用 MarkdownHeaderTextSplitter 按 section 切分。
3. 对每个 section：清洗条款编号和 a)/b)/c) 小项，保留换行，再按 max_part_chars 逐行切块。
4. 若完全切不出 section：将全文清洗后，短文档作为一条，长文档按 max_part_chars 切块。
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional

import jsonlines
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 行首类似 "3 1 ..." 或 "12 2 ..."
RE_LINE_STARTS_WITH_NUM = re.compile(r'^\s*\d+(\s+\d+)?\s+')
# 行首类似 "a)", "b.", "c " 这种小项
RE_LINE_STARTS_WITH_LETTER_ITEM = re.compile(r'^\s*([a-z])[\).\s]+')

# PART I / PART 1 / PART II ...
RE_PART_HEADER = re.compile(r'^\s*PART\s+([IVXLC]+|\d+)\b', re.IGNORECASE)
# Schedule 1 / SCHEDULE 2 ...
RE_SCHED_HEADER = re.compile(r'^\s*SCHEDULE\s+([IVXLC]+|\d+)\b', re.IGNORECASE)


def is_potential_section_header(line: str, next_nonempty_line: Optional[str]) -> bool:
    line = line.strip()
    if not line:
        return False

    words = line.split()

    # 单字母/两字母的行（如 a, b, c, I, II），一定不是标题
    if len(words) == 1 and len(words[0]) <= 2:
        return False

    # PART / SCHEDULE 这种强结构标题
    if RE_PART_HEADER.match(line) or RE_SCHED_HEADER.match(line):
        return True

    # 全大写且词数不多（如 "AMENDMENT OF ..."）
    if line.isupper() and 1 <= len(words) <= 10:
        return True

    # 数字开头通常是条款内容（如 "3 1 Where ...")
    if line[0].isdigit():
        return False

    if len(line) > 200:
        return False

    if next_nonempty_line is None:
        return False

    # 一般标题：3–25 个单词，下一行是数字条款
    if not (3 <= len(words) <= 25):
        return False

    if RE_LINE_STARTS_WITH_NUM.match(next_nonempty_line):
        return True

    return False


def body_to_markdown_with_headers(body: str) -> str:
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    lines = body.split("\n")
    n = len(lines)

    markdown_lines: List[str] = []
    idx = 0

    while idx < n:
        line = lines[idx]
        stripped = line.strip()

        next_nonempty = None
        j = idx + 1
        while j < n:
            s2 = lines[j].strip()
            if s2:
                next_nonempty = s2
                break
            j += 1

        if is_potential_section_header(stripped, next_nonempty):
            header_text = stripped

            # SCHEDULE 开头时，括号后内容作为正文
            if RE_SCHED_HEADER.match(header_text):
                paren_pos = header_text.find("(")
                if paren_pos > 0:
                    core = header_text[:paren_pos].rstrip()
                    rest = header_text[paren_pos:].lstrip()
                    if core:
                        markdown_lines.append(f"## {core}")
                    else:
                        markdown_lines.append(f"## {header_text}")
                    if rest:
                        markdown_lines.append(rest)
                    idx += 1
                    continue

            markdown_lines.append(f"## {header_text}")
        else:
            markdown_lines.append(line)

        idx += 1

    return "\n".join(markdown_lines)


def clean_desc_text(text: str, keep_line_breaks: bool = True) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        line = RE_LINE_STARTS_WITH_NUM.sub("", line)

        m = RE_LINE_STARTS_WITH_LETTER_ITEM.match(line)
        if m:
            item = m.group(1)
            line = RE_LINE_STARTS_WITH_LETTER_ITEM.sub(f"({item}) ", line, count=1)

        cleaned_lines.append(line)

    if not cleaned_lines:
        return ""

    if keep_line_breaks:
        return "\n".join(cleaned_lines)
    else:
        return " ".join(cleaned_lines)


def _split_long_line_by_tokens(line: str, max_chars: int) -> List[str]:
    line = line.strip()
    if not line:
        return []
    if len(line) <= max_chars:
        return [line]

    parts = []
    current_tokens: List[str] = []
    current_len = 0

    for tok in line.split():
        tok_len = len(tok)
        add_len = tok_len + (1 if current_tokens else 0)
        if current_len + add_len > max_chars:
            parts.append(" ".join(current_tokens))
            current_tokens = [tok]
            current_len = tok_len
        else:
            current_tokens.append(tok)
            current_len += add_len

    if current_tokens:
        parts.append(" ".join(current_tokens))

    return parts


def split_text_by_length(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    lines = text.split("\n")
    parts: List[str] = []
    current_lines: List[str] = []
    current_len = 0

    for raw_line in lines:
        line = raw_line.rstrip()

        if not line:
            add_len = 1 if current_lines else 0
            if current_len + add_len <= max_chars:
                current_lines.append("")
                current_len += add_len
            else:
                parts.append("\n".join(current_lines))
                current_lines = [""]
                current_len = 1
            continue

        line_len = len(line)

        # 单行太长，先 flush 当前块，再对这一行按 token 切分
        if line_len > max_chars:
            if current_lines:
                parts.append("\n".join(current_lines))
                current_lines = []
                current_len = 0
            long_parts = _split_long_line_by_tokens(line, max_chars)
            parts.extend(long_parts)
            continue

        add_len = line_len + (1 if current_lines else 0)
        if current_len + add_len <= max_chars:
            current_lines.append(line)
            current_len += add_len
        else:
            parts.append("\n".join(current_lines))
            current_lines = [line]
            current_len = line_len

    if current_lines:
        parts.append("\n".join(current_lines))

    return parts


def process_single_jsonl(
    input_file: Path,
    writer: jsonlines.Writer,
    markdown_splitter: MarkdownHeaderTextSplitter,
    min_desc_chars: int,
    max_part_chars: int,
) -> None:
    with jsonlines.open(input_file, "r") as reader:
        for doc in reader:
            doc_id = doc.get("id") or doc.get("_id") or ""
            year = doc.get("year", "")
            full_title = doc.get("title", "")
            body = doc.get("body", "") or ""

            # 1. 先统一做“插入 markdown 标题 + 按章节切”
            markdown_text = body_to_markdown_with_headers(body)
            md_docs = markdown_splitter.split_text(markdown_text)

            # 是否真的存在至少一个非空 section_title
            has_real_section = any(
                ((d.metadata or {}).get("section_title") or "").strip()
                for d in md_docs
            )

            if has_real_section:
                # 有章节标题：按章节切 + 按长度二次切
                sec_idx = 0
                for md_doc in md_docs:
                    meta = md_doc.metadata or {}
                    sec_title = (meta.get("section_title") or "").strip()
                    section_body = md_doc.page_content

                    # 前言或无标题块，用全文标题或 "Preamble" 兜底
                    if not sec_title:
                        sec_title = full_title or "Preamble"

                    desc = clean_desc_text(section_body, keep_line_breaks=True)
                    if not desc:
                        continue

                    parts = split_text_by_length(desc, max_chars=max_part_chars)
                    if not parts:
                        continue

                    for part_idx, part in enumerate(parts):
                        if len(part) < min_desc_chars:
                            continue

                        if len(parts) == 1:
                            corpus_id = f"{doc_id}__sec{sec_idx}"
                        else:
                            corpus_id = f"{doc_id}__sec{sec_idx}_part{part_idx}"

                        text_field = f"Title: {sec_title}\nDesc: {part}"
                        out_item = {
                            "_id": corpus_id,
                            "title": "",
                            "text": text_field,
                            "_source_id": doc_id,
                            "_source_year": year,
                            "_source_title": full_title,
                            "_source_file": str(input_file.name),
                        }
                        writer.write(out_item)

                    sec_idx += 1

                # 这一篇 doc 处理完，继续下一篇
                continue

            # 2. 完全没有章节标题：走“整篇清洗 + 长度切”逻辑
            desc_full = clean_desc_text(body, keep_line_breaks=True)
            if not desc_full:
                # 这一篇没法用，跳过
                continue

            if len(desc_full) <= max_part_chars:
                # 短文档：当成一条 __full
                if len(desc_full) >= min_desc_chars:
                    corpus_id = f"{doc_id}__full"
                    text_field = f"Title: {full_title}\nDesc: {desc_full}"
                    out_item = {
                        "_id": corpus_id,
                        "title": "",
                        "text": text_field,
                        "_source_id": doc_id,
                        "_source_year": year,
                        "_source_title": full_title,
                        "_source_file": str(input_file.name),
                    }
                    writer.write(out_item)
                continue

            # 长文档且无章节：整篇按长度拆为 part
            parts = split_text_by_length(desc_full, max_chars=max_part_chars)
            part_idx = 0
            for part in parts:
                if len(part) < min_desc_chars:
                    continue
                corpus_id = f"{doc_id}__part{part_idx}"
                part_idx += 1
                text_field = f"Title: {full_title}\nDesc: {part}"
                out_item = {
                    "_id": corpus_id,
                    "title": "",
                    "text": text_field,
                    "_source_id": doc_id,
                    "_source_year": year,
                    "_source_title": full_title,
                    "_source_file": str(input_file.name),
                }
                writer.write(out_item)


def build_ailastatutes_corpus_with_langchain(
    input_files: List[Path],
    output_path: str,
    min_desc_chars: int = 50,
    max_part_chars: int = 10000,
) -> None:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "section_title")]
    )

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_path, "w") as writer:
        for input_file in tqdm(input_files, desc="Building corpus (multi-jsonl)"):
            process_single_jsonl(
                input_file=input_file,
                writer=writer,
                markdown_splitter=markdown_splitter,
                min_desc_chars=min_desc_chars,
                max_part_chars=max_part_chars,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build AILA-style statute corpus from UK-LEX jsonl(s) using LangChain text splitters."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_file",
        type=str,
        help="单个 UK-LEX jsonl 文件路径",
    )
    group.add_argument(
        "--input_dir",
        type=str,
        help="包含多个 UK-LEX jsonl 的目录（会处理目录下所有 *.jsonl）",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出的 corpus jsonl 文件路径",
    )
    parser.add_argument(
        "--min_desc_chars",
        type=int,
        default=50,
        help="过滤过短条文的最小字符数（默认 50）",
    )
    parser.add_argument(
        "--max_part_chars",
        type=int,
        default=10000,
        help="每个输出片段允许的最大字符数（默认 10000）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input_file:
        input_files = [Path(args.input_file)]
    else:
        input_dir = Path(args.input_dir)
        input_files = sorted(input_dir.glob("*.jsonl"))
        if not input_files:
            raise ValueError(f"No .jsonl files found in directory: {input_dir}")

    build_ailastatutes_corpus_with_langchain(
        input_files=input_files,
        output_path=args.output_file,
        min_desc_chars=args.min_desc_chars,
        max_part_chars=args.max_part_chars,
    )
    print(f"Done. Corpus saved to {args.output_file}")
