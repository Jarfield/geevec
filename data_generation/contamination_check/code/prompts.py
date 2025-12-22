"""Prompt helpers for contamination check query rewriting."""

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that rewrites search queries. "
    "Preserve the original intent and key constraints while changing the surface form. "
    "Do not add, remove, or contradict information. Respond with only the rewritten query."
)


def build_rewrite_prompt(query: str) -> str:
    """Create a rewrite prompt for a single query."""
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Original query:\n{query}\n\n"
        "Rewritten query:"
    )


__all__ = ["SYSTEM_INSTRUCTION", "build_rewrite_prompt"]
