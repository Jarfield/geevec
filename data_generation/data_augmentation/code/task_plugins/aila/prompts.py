"""AILAStatutes-specific prompt builders following a legal syllogism pattern."""

from typing import List, Optional

from data_generation.shared.constants import Task, TaskType


def _render_examples(examples: Optional[List[dict]], gen_output: str) -> str:
    if not examples:
        return ""

    rendered = []
    for idx, example in enumerate(examples, 1):
        rendered.append(
            f"""- Example {idx}:
    Statute:
    {example.get("input") or example.get("context") or example.get("content")}

    Expected scenario ({gen_output}): {example.get("output") or example.get("target")}

"""
        )
    return "Here are a few examples for reference:\n" + "".join(rendered)


def build_generation_prompt(
    task: Task,
    text: str,
    examples: Optional[List[dict]] = None,
    narrative_focus: Optional[str] = None,
) -> str:
    if task.task_type is not TaskType.ailastatutes:
        raise ValueError("AILA plugin received a non-AILA task.")

    focus_hints = {
        "victim_focus": "Describe the harm and how the affected person experiences the events.",
        "investigation_focus": "Highlight observable actions, evidence, and procedures that reveal the violation.",
        "judgment_focus": "Spell out the critical facts that would matter in a verdict without naming charges.",
        "social_impact_focus": "Show the wider ripple effects on community, commerce, or safety.",
        "neutral_brief": "Keep the narrative concise and purely factual.",
    }
    focus_hint = ""
    if narrative_focus and narrative_focus in focus_hints:
        focus_hint = f"\nPerspective hint: {focus_hints[narrative_focus]}"

    gen_output = "a single concrete situation (one paragraph)"
    examples_block = _render_examples(examples, gen_output)

    return f"""You are constructing realistic fact patterns that demonstrate when a statute applies.
Work through reverse legal syllogism:
1) Element extraction: list the observable conditions the statute requires (actors, actions, thresholds).
2) Scenario instantiation: create a concrete situation that satisfies each element with people, actions, and context.
3) De-jargonization: narrate only facts and behaviors—avoid legal terms, citations, or conclusions.

Statute (reference only, do NOT quote directly):
[Begin of Statute]
{text}
[End of Statute]{focus_hint}

Output requirements:
- Return exactly {gen_output}.
- Avoid numbered lists, bullet points, or headings.
- Do not repeat the statute text or include legal labels.

{examples_block}Your output:"""


__all__ = ["build_generation_prompt"]
