import os
from hashlib import md5
from typing import Iterable, Tuple


def compute_md5(text: str) -> str:
    """Return a stable md5 hash for deduplication / caching."""

    return md5(text.encode()).hexdigest()


def clean_content(content: str) -> str:
    if content is None:
        raise ValueError("content is None.")

    content = content.strip()

    if content.startswith('\"') and content.endswith('\"'):
        content = content[1:-1]

    if content.startswith("```\n") and content.endswith("\n```"):
        content = content[4:-4]

    if content.startswith("```") and content.endswith("```"):
        content = content[3:-3]

    return content.strip()


def ensure_dir(path: str) -> str:
    """Create directory if missing and return the normalized path."""

    os.makedirs(path, exist_ok=True)
    return path


def chunked(iterable: Iterable, size: int) -> Iterable[Tuple]:
    """Yield chunks from an iterable in a memory-friendly way."""

    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)
