"""Utility helpers for structuring AILA statute documents."""

import os
import re
import sys
from typing import List

# 将 data_augmentation 的 code 目录加入系统路径，复用公共 LLM 与生成器逻辑
CURRENT_DIR = os.path.dirname(__file__)
PARENT_CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "code"))
if PARENT_CODE_DIR not in sys.path:
    sys.path.append(PARENT_CODE_DIR)

LAW_NAME = "[Indian Penal Code]"
ANCHOR_PATTERN = re.compile(r"^(Section|Article)\s+\d+\.", re.MULTILINE)
TAIL_SEGMENT_PATTERN = re.compile(r"^(Explanation|Exception|Illustration)[.:]?", re.IGNORECASE)


def _find_anchor_indices(text: str) -> List[int]:
    return [match.start() for match in ANCHOR_PATTERN.finditer(text)]


def structural_chunker(text: str, law_name: str = LAW_NAME) -> List[str]:
    """Split IPC statutes into self-contained chunks with contextual headers.

    Algorithm:
    1) Anchor recognition: detect ``Section N.`` or ``Article N.`` headings.
    2) Look-ahead aggregation: keep attaching trailing ``Explanation``,
       ``Exception`` or ``Illustration`` paragraphs to the anchor until the
       next anchor appears.
    3) Context completion: prefix every chunk with the law name to remove
       pronoun ambiguity.
    """

    if not text or not text.strip():
        return []

    anchor_indices = _find_anchor_indices(text)
    if not anchor_indices:
        return [f"{law_name} {text.strip()}"]

    chunks: List[str] = []
    for idx, anchor_start in enumerate(anchor_indices):
        anchor_end = anchor_indices[idx + 1] if idx + 1 < len(anchor_indices) else len(text)
        raw_chunk = text[anchor_start:anchor_end].strip()

        # Ensure that Explanation/Exception/Illustration immediately following
        # the chunk are retained even if they are separated by blank lines.
        trailing = []
        lookahead_text = text[anchor_end:].lstrip()
        while True:
            tail_match = TAIL_SEGMENT_PATTERN.match(lookahead_text)
            if not tail_match:
                break
            next_anchor = ANCHOR_PATTERN.search(lookahead_text, tail_match.end())
            tail_end = next_anchor.start() if next_anchor else len(lookahead_text)
            trailing.append(lookahead_text[:tail_end].strip())
            lookahead_text = lookahead_text[tail_end:].lstrip()

        if trailing:
            raw_chunk = f"{raw_chunk}\n\n" + "\n\n".join(trailing)

        chunks.append(f"{law_name} {raw_chunk}")

    return chunks


__all__ = ["structural_chunker", "LAW_NAME"]
