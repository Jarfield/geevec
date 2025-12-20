"""Prompt templates for AILA statute reasoning."""

import os
import sys
from typing import Optional

# 将 data_augmentation 的 code 目录加入系统路径，复用公共 LLM 与生成器逻辑
CURRENT_DIR = os.path.dirname(__file__)
PARENT_CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "code"))
if PARENT_CODE_DIR not in sys.path:
    sys.path.append(PARENT_CODE_DIR)


def get_layman_summary_prompt(statute: str) -> str:
    """Prompt the LLM to rewrite a statute into a layman-friendly summary."""

    return f"""\
You are a legal explainer. Rewrite the following statute into a layman-friendly summary in English.
- Keep the key conditions and limitations.
- Avoid legalese and numbered references.

[Statute]
{statute}

Your output must be a concise paragraph that an average reader can understand."""


def get_syllogism_scenario_prompt(statute: str, layman_summary: Optional[str] = None) -> str:
    """Reverse legal syllogism prompt to instantiate scenarios."""

    summary_hint = (
        f"\nLayman bridge (use as semantic guide, not verbatim source):\n{layman_summary}\n"
        if layman_summary
        else ""
    )

    return f"""\
You are generating fact patterns that demonstrate how a statute applies. Work in three steps:
1) Component extraction: list the essential elements required by the statute (e.g., dishonest intent, movement of property).
2) Scenario instantiation: craft a concrete situation with people, actions, and context that satisfies each element (e.g., Suresh quietly picks up a ring).
3) De-jargonization: describe only observable behaviors and outcomes. Do not cite legal terms or conclusions.

Statute:
{statute}
{summary_hint}

Return a single coherent scenario (S0) in English. Avoid bullet points and legal terminology; focus on narrative facts only."""


__all__ = ["get_layman_summary_prompt", "get_syllogism_scenario_prompt"]
