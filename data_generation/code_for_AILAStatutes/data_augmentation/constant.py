from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


class TaskType(Enum):
    # MIRACL-style web / Wikipedia passage retrieval
    miracl = "Given a question, retrieve Wikipedia passages that answer the question."
    # AILAStatutes: statute retrieval for a described legal situation
    ailastatutes = "Identifying the most relevant statutes for a given situation."


class Language(Enum):
    en = "English"  # 英语
    zh = "Simplified Chinese"  # 简体中文
    ar = "Arabic"  # 阿拉伯语
    bn = "Bengali"  # 孟加拉语
    es = "Spanish"  # 西班牙语
    fa = "Persian"  # 波斯语
    fi = "Finnish"  # 芬兰语
    fr = "French"  # 法语
    hi = "Hindi"  # 印地语
    id = "Indonesian"  # 印度尼西亚语
    ja = "Japanese"  # 日语
    ko = "Korean"  # 韩语
    ru = "Russian"  # 俄语
    sw = "Swahili"  # 斯瓦希里语
    te = "Telugu"  # 泰卢固语
    th = "Thai"  # 泰语
    de = "German"  # 德语
    yo = "Yoruba"  # 约鲁巴语


@dataclass
class Task:
    task_type: TaskType
    language: Language
    task_instruction: Optional[str] = None


def get_task(
    task_type: str,
    language: str,
) -> Task:
    """
    Helper to build a Task object from string names,
    e.g. get_task("miracl", "en") or get_task("ailastatutes", "en").
    """
    task_enum = TaskType[task_type]
    lang_enum = Language[language]

    return Task(
        task_type=task_enum,
        language=lang_enum,
        task_instruction=task_enum.value,
    )


def get_generation_prompt(
    task: Task,
    text: str,
    examples: Optional[List[dict]] = None,
    narrative_focus: Optional[str] = None,
) -> str:
    """
    Build a generation prompt for creating queries / situations from a document.

    narrative_focus:
        - Only meaningful for TaskType.ailastatutes. Expected values:
          "victim_focus", "investigation_focus", "judgment_focus",
          "social_impact_focus", "neutral_brief".
        - If None, we default to "neutral_brief".
    """
    task_to_gen_instruction: Dict[TaskType, str] = {
        TaskType.miracl: (
            "Given a Wikipedia passage in {language}, generate a question in {language} "
            "that the passage can answer."
        ),
        TaskType.ailastatutes: (
            "Given a statute in {language}, generate a realistic legal situation in {language} "
            "in which the statute is clearly and directly applicable."
        ),
    }

    task_to_gen_output: Dict[TaskType, str] = {
        TaskType.miracl: "the generated question in {language}",
        TaskType.ailastatutes: "the generated situation in {language}",
    }

    gen_instruction = task_to_gen_instruction[task.task_type]
    gen_output = task_to_gen_output[task.task_type]

    gen_instruction = gen_instruction.replace("{language}", task.language.value)
    gen_output = gen_output.replace("{language}", task.language.value)

    # 统一的“给定内容”前缀
    prefix = "The given content (document):"

    # ===== 针对 AILAStatutes 的额外约束 & 叙事视角 =====
    extra_guidelines = ""
    narrative_guidelines = ""

    if task.task_type is TaskType.ailastatutes:
        extra_guidelines = """
Additional requirements for the situation:
- Do NOT mention the exact name, section number, or title of the statute.
- Do NOT use phrases like "under this statute", "under the above provision",
  "under section X", or similar explicit references.
- Describe the facts, roles, actions, and consequences in natural language,
  as if summarizing a real legal case or incident.
- It is fine to mention that criminal charges or punishment may follow, but
  you do NOT need to repeat the exact wording or full sentencing range from the statute.
"""

        focus = narrative_focus or "neutral_brief"

        narrative_guidelines = f"""
Diversity and narrative-style requirements:
- Narrative focus: {focus}

  The narrative focus must be interpreted as follows:
  * If the focus is "victim_focus":
    - Describe the situation mainly from the perspective of the victim or affected person(s),
      including their circumstances, experiences, and the impact of the wrongdoing on them.
    - You may mention the investigation or charges briefly, but do not make them the main theme.

  * If the focus is "investigation_focus":
    - Describe how the conduct was discovered and investigated by the police, regulators,
      or other authorities.
    - Emphasise the sequence of investigative steps, evidence collection, and how the
      authorities formed their conclusions.

  * If the focus is "judgment_focus":
    - Write as if summarising the facts and findings in a court judgment or written decision.
    - Include how the court characterised the conduct, the key legal issues, and the outcome
      of the case (e.g., conviction or acquittal), without copying the statute text.

  * If the focus is "social_impact_focus":
    - Emphasise the broader consequences of the conduct on the community, public trust,
      or specific sectors (such as the financial system or online platforms).
    - Media coverage, public debate, and institutional responses may be described,
      while details of the exact sentence can remain vague.

  * If the focus is "neutral_brief":
    - Provide a concise and neutral case summary, focusing on who did what to whom,
      in what context, and with what consequences.
    - Keep the description relatively short and to the point.

General style constraints for all focuses:
- Your output must be a single, coherent case description, written as plain prose.
- It must be fully understandable without reading the statute.
- It must NOT contain pronouns such as "it", "this", "that", "the above", or
  "the given statute" that refer back to the prompt.
- Do NOT include headings, bullet points, or any metadata (such as "Case:", "Facts:", etc.).
- Aim for 4–8 sentences of natural, well-structured English.
"""

    gen_prompt = f"""{gen_instruction}
{extra_guidelines}{narrative_guidelines}

{prefix}
[Begin of Content]
{text}
[End of Content]

- Your output must always be a single string, only containing {gen_output}.
- Your output should be self-contained and understandable without looking at the given document.
  In particular, it should not contain pronouns such as "it", "this", "that",
  "the given", "the provided", etc.
"""

    if examples is not None and len(examples) > 0:
        examples_str_list = [
            f"""\
- Example {i + 1}:
    {prefix}
    [Begin of Content]
    {example['input']}
    [End of Content]

    Expected Output ({gen_output}): {example['output']}

"""
            for i, example in enumerate(examples)
        ]

        gen_prompt += f"""\
Here are a few examples for your reference:
{''.join(examples_str_list)}"""

    gen_prompt += "\nRemember: do not explain your output or output anything else. Your output:"

    return gen_prompt

def get_statute_synthesis_prompt(
    task: Task,
    seed_text: str,
    extra_style_examples: Optional[List[str]] = None,
) -> str:
    """
    New helper specifically for AILAStatutes-style statute synthesis.

    Goal:
    - Use one or more real statutes as style / structure references.
    - Ask the LLM to generate ONE NEW synthetic statute-like document
      that is plausible, self-contained, and not a paraphrase of the input.

    Args:
        task: should have task.task_type == TaskType.ailastatutes.
        seed_text: main reference statute(s), concatenated as a single string.
        extra_style_examples: optional list of additional statutes (strings)
                              used only as extra style references.
    """
    if task.task_type is not TaskType.ailastatutes:
        raise ValueError(
            "get_statute_synthesis_prompt is only intended for TaskType.ailastatutes."
        )

    lang = task.language.value

    prompt = f"""\
You are an expert legal drafter. Your task is to write ONE NEW synthetic statute provision in {lang}.

The new statute must satisfy ALL of the following requirements:
1. **Style & structure**:
   - It should follow the same legal style, tone, and level of formality as the reference statutes.
   - It may use numbered sections / clauses if appropriate.
2. **New content**:
   - It must address a different legal rule or situation than the reference statutes.
   - It must NOT be a simple rewrite or paraphrase of any reference sentence.
3. **Self-contained**:
   - It must be understandable on its own, without referring to the reference statutes.
   - It must NOT contain expressions such as "the above-mentioned statute", "the previous section",
     "this Act as stated above", etc.
4. **No meta-commentary**:
   - Do NOT mention that this is an example, a sample, a template, or a synthetic provision.
   - Output ONLY the text of the new statute.

Below are some reference statutes. They are **style guides only** and must not be copied.

[Begin of Reference Statutes]
{seed_text}
[End of Reference Statutes]
"""

    if extra_style_examples:
        joined_examples = "\n\n---\n\n".join(extra_style_examples)
        prompt += f"""

Additional style references:
[Begin of Additional Reference Statutes]
{joined_examples}
[End of Additional Reference Statutes]
"""

    prompt += """

Now, write ONE new statute provision that follows the style of the references,
but has different content. Output ONLY the text of the new statute, with no explanation,
no comments, and no additional markers.
Your output:
"""

    return prompt


def get_quality_control_prompt(
    task: Task,
    query: str,
    pos: str,
) -> str:
    """
    Build a classification prompt to judge (query, document) compatibility.

    For now:
    - miracl: question vs Wikipedia passage
    - ailastatutes: legal situation vs statute
    """
    task_to_qc_mission: Dict[TaskType, Tuple[str, str, str, List[str]]] = {
        TaskType.miracl: (
            "judge whether the Wikipedia passage can answer the question",
            "the question",
            "the Wikipedia passage",
            [
                "Yes, the Wikipedia passage can answer the question.",
                "No, the Wikipedia passage cannot answer the question.",
            ],
        ),
        TaskType.ailastatutes: (
            "judge whether the statute is clearly applicable to the given legal situation",
            "the situation describing a legal case or fact pattern",
            "the statute",
            [
                "Yes, the statute is clearly applicable and provides a direct legal basis for resolving the situation.",
                "No, the statute is not clearly applicable to the situation or is only weakly or indirectly related.",
            ],
        ),
    }

    qc_mission, query_type, doc_type, qc_options = task_to_qc_mission[task.task_type]

    pos_option = qc_options[0]
    neg_option = qc_options[1]

    qc_prompt = f"""\
Given an information retrieval task (Task), a query (Query), and a document (Document),
your mission is to {qc_mission}.

Task:
{task.task_instruction}
Query ({query_type}):
{query}
Document ({doc_type}):
{pos}
Your output must be one of the following options:
- 0: {neg_option}
- 1: {pos_option}

Do not explain your answer in the output. Your output must be a single number (0 or 1).

Your output:"""

    return qc_prompt