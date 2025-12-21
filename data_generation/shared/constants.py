from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


class TaskType(Enum):
    # MIRACL：维基百科段落检索
    miracl = "Given a question, retrieve Wikipedia passages that answer the question."
    # CovidRetrieval：新冠相关新闻检索
    covidretrieval = "Given a question on COVID-19, retrieve news articles that answer the question."
    # AILA：法规匹配检索
    ailastatutes = "Identifying the most relevant statutes for a given situation."
    # SCIDOCS：论文摘要检索
    scidocs = "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper."
    # ArguAna：论证段落检索
    arguana = "Given a claim, find documents that refute the claim."


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
    """根据字符串快速构造 Task 对象，例如 get_task("miracl", "en")。"""
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
    narrative_focus: Optional[str] = None,   # optional focus control
) -> str:
    task_to_gen_instruction: Dict[TaskType, str] = {
        TaskType.miracl: (
            "Given a Wikipedia passage in {language}, generate a question in {language} that the passage can answer."
        ),
        TaskType.covidretrieval: (
            "Given a news article related to COVID-19 in {language}, generate a question in {language} that the article can answer."
        ),
        TaskType.scidocs: (
            "Given a scientific paper abstract in {language}, generate a scientific paper title in {language} that would likely cite this paper as a reference."
        ),
        TaskType.arguana: (
            "Given a document in {language}, generate a claim in {language} that the document would refute."
        ),
    }

    task_to_gen_output: Dict[TaskType, str] = {
        TaskType.miracl: "the generated question in {language}",
        TaskType.covidretrieval: "the generated question in {language}",
        TaskType.scidocs: "the generated scientific paper title in {language}",
        TaskType.arguana: "the generated claim in {language}",
    }

    gen_instruction = task_to_gen_instruction.get(
        task.task_type,
        "Given a document in {language}, generate a retrieval query in {language} that directly matches it.",
    ).replace("{language}", task.language.value)
    gen_output = task_to_gen_output.get(
        task.task_type, "the generated query in {language}"
    ).replace("{language}", task.language.value)

    prefix = "The given content:"

    # ---------- narrative focus hints ----------
    focus_hint = ""
    if task.task_type is TaskType.covidretrieval and narrative_focus is not None:
        covid_focus_hints: Dict[str, str] = {
            "covid_fact_detail": (
                "\n\nFor this question, please focus on **concrete factual details** "
                "in the article, such as specific times, locations, case numbers, "
                "virus variants, or research findings. Ask a precise question that "
                "can be directly answered from these factual details in the article."
            ),
            "covid_policy_measure": (
                "\n\nFor this question, please focus on **COVID-19 related policies or measures**, "
                "such as control policies, travel restrictions, quarantine requirements, "
                "testing regulations, or campus/community management measures. Ask a question "
                "that directly targets these policies or measures themselves."
            ),
            "covid_vaccine_treatment": (
                "\n\nFor this question, please focus on **vaccines, drugs, or clinical treatment**, "
                "such as vaccine types, target populations, protective effects, side effects, "
                "drug trial results, or treatment protocols. Ask a question centered on "
                "vaccines, medications, or clinical studies mentioned in the article."
            ),
            "covid_risk_protection": (
                "\n\nFor this question, please focus on **risk and protection**, such as "
                "transmission risk, infection probability in specific scenarios, high-risk groups, "
                "personal protective advice, or expert reminders. Ask a question related to "
                "risk assessment or protective recommendations."
            ),
            "covid_social_impact": (
                "\n\nFor this question, please focus on **the social and economic impacts of COVID-19**, "
                "such as its effects on the economy, education, employment, mental health, or social order. "
                "Ask a question that highlights these broader social impacts or long-term consequences."
            ),
        }
        if narrative_focus in covid_focus_hints:
            focus_hint = covid_focus_hints[narrative_focus]

    elif task.task_type is TaskType.scidocs and narrative_focus is not None:
        # 预留接口：后续可以在这里根据 narrative_focus 控制 title 的侧重点
        # 例如：methodology / application / dataset / theoretical_contribution 等
        pass

    # ---------- main prompt ----------
    gen_prompt = f"""\
{gen_instruction}{focus_hint}

{prefix}
[Begin of Content]
{text}
[End of Content]

- Your output must consist of a single {gen_output}, without any extra explanation.
- Your output should be independent of the given passage, which means that it should not contain pronouns such as "it", "this", "that", "the given", "the provided", etc.
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
{''.join(examples_str_list)}
"""

    gen_prompt += "Remember: do not explain your output or output anything else. Your output:"

    return gen_prompt


def get_pair_scoring_prompt(
    task: Task,
    query: str,
    doc: str,
) -> str:
    """构造 Query-Document 打分的提示词，返回 <score>1-5</score>。"""
    relevance_scale = (
        "- 5 (Highly Relevant): The document is directly and fully responsive to the query, providing comprehensive, accurate, and specific information that completely addresses all aspects of the query.\n"
        "- 4 (Relevant): The document is largely relevant and provides most of the information needed, but may have minor omissions, slight inaccuracies, or not be perfectly aligned with the query’s intent.\n"
        "- 3 (Moderately Relevant): The document has some relevance and offers partial information, but it may be incomplete, vague, or include some irrelevant content.\n"
        "- 2 (Slightly Relevant): The document has minimal relevance, with only a small portion tangentially related to the query.\n"
        "- 1 (Irrelevant): The document is completely unrelated to the query and provides no useful information."
    )

    mission = (
        "Now given a query and a document in this retrieval task, your mission is to:\n"
        "1. Analyze the query and document to understand the information need.\n"
        "2. Judge the relevance between them using the following 1-5 scoring guide.\n"
        "3. Output the final score strictly between <score> and </score> tags with no extra text inside the tags."
    )

    prompt = f"""\
Task: {task.task_instruction}

{mission}
Scoring Guide:
{relevance_scale}

Query:
[Begin of Query]
{query}
[End of Query]

Document:
[Begin of Document]
{doc}
[End of Document]

Final score (only the number inside the tags):
<score>
</score>
"""
    return prompt

def get_doc_synthesis_prompt(
    task: Task,
    seed_text: str,
    narrative_attributes: Optional[str] = None,
    extra_style_examples: Optional[List[str]] = None,
) -> str:
    """
    Generate a prompt for document synthesis / rewriting.

    Core idea (as instructions to the model):
    - Internally: first extract the topic and a simplified conceptual summary
      of the seed document.
    - Then: based on that summary, write a new document whose meaning is
      very similar to the original, but whose wording, style, and surface
      form are clearly different.
    - The differences in style / tone / length can be controlled by
      `narrative_attributes`.

    Final output format (for the model):
    - Title: <one-sentence title>
    - Desc: <multi-paragraph body>
    """

    # 1) 针对不同任务给一个“领域说明”，方便复用到 miracl / covid 等
    task_to_domain_desc: Dict[TaskType, str] = {
        TaskType.miracl: (
            "Wikipedia-style explanatory articles and general knowledge passages"
        ),
        TaskType.covidretrieval: (
            "public health and COVID-19 related news articles"
        ),
    }
    domain_desc = task_to_domain_desc.get(
        task.task_type,
        "general informative articles"
    )

    language = task.language.value 

    # 2) narrative attributes
    if narrative_attributes is None:
        narrative_attributes = (
            "a neutral, informative style with clear structure and moderate length"
        )

    # 3) 主体 prompt：描述内部步骤
    prompt = f"""\
You are an expert writer who works with {domain_desc} in {language}.

Your goal is to synthesize a **new document** based on a given seed document.
The new document must:
- Preserve the core meaning and main ideas of the original.
- Be clearly different in wording, phrasing, and surface form.
- Avoid copying sentences or long fragments from the original.
- Follow the narrative attributes described below.

Internally, you should follow these steps (but DO NOT show these steps in your output):

1. Carefully read the seed document.
2. Identify its main topic and central theme.
3. Extract a simplified conceptual summary of the content:
   - What happened, who is involved (in abstract terms), what is reported,
     what actions or outcomes are described.
   - Generalize away from specific numbers, names, and locations.
4. Based on this conceptual summary, write a new document that:
   - Has very similar meaning to the original,
   - But uses different wording, different sentence structures,
     and possibly a slightly different organization.
   - Applies the given narrative attributes.

You must also obey the following transformation rules:
- Do NOT copy any sentences from the seed document.
- Do NOT reuse specific named entities (people, organizations, cities, etc.) from the seed document; replace them with new but plausible ones.
- Change all numerical values (dates, counts, percentages, etc.) to new, reasonable values.
- You may add new, plausible details as long as they remain consistent with the core meaning.
- The final document should be coherent, self-contained, and suitable for publication as {domain_desc}.

Narrative attributes to apply in the new document:
- {narrative_attributes}

The seed document (for internal analysis only, do NOT copy text from it directly):
[Seed Document Start]
{seed_text}
[Seed Document End]
"""

    # 4) 额外风格参考文本
    if extra_style_examples:
        joined = "\n\n---\n\n".join(extra_style_examples)
        prompt += f"""\

Here are additional reference texts for style inspiration ONLY (do NOT copy them):

[Additional Style References Start]
{joined}
[Additional Style References End]
"""

    # 5) 最终输出格式：只要 Title / Desc，内部步骤不要输出
    prompt += f"""\

Now, based on your internal analysis and the transformation rules above,
produce the final synthesized document in {language}.

Your output must strictly follow this format:

Title: <a single-sentence title summarizing the new synthesized document>
Desc: <multi-paragraph body text in {language}, paragraphs separated by blank lines>

Important:
- Do NOT include any explanations.
- Do NOT describe your internal steps.
- Do NOT output the topic or summary explicitly.
- Only output the two lines starting with "Title:" and "Desc:".

Your output:
"""

    return prompt

def get_quality_control_prompt(
    task: Task,
    query: str,
    pos: str,
) -> str:
    """
    构造一个分类 prompt，用于判断 (query, document) 是否匹配。
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
        TaskType.covidretrieval: (
            "judge whether the COVID-19 news article can answer the question",
            "the question about COVID-19",
            "the news article related to COVID-19",
            [
                "Yes, the news article can answer the question or provide direct information to address it.",
                "No, the news article cannot answer the question or is only weakly or indirectly related.",
            ],
        ),
        TaskType.ailastatutes: (
            "judge whether the statute is directly applicable to the situation",
            "the situation",
            "the statute",
            [
                "Yes, the statute applies directly to the described situation.",
                "No, the statute does not apply or is only tangentially related.",
            ],
        ),
        TaskType.scidocs: (
            "judge whether the abstract discusses the topic asked by the research question",
            "the research question",
            "the scientific abstract",
            [
                "Yes, the abstract clearly addresses or answers the research question.",
                "No, the abstract does not answer the question or is only weakly related.",
            ],
        ),
        TaskType.arguana: (
            "judge whether the passage supports or contests the given claim",
            "the claim",
            "the argumentative passage",
            [
                "Yes, the passage directly supports or contests the claim.",
                "No, the passage is unrelated to the claim.",
            ],
        ),
    }

    qc_mission, query_type, doc_type, qc_options = task_to_qc_mission[task.task_type]

    pos_option = qc_options[0]
    neg_option = qc_options[1]

    qc_prompt = f"""\
Given an information retrieval task (Task), a query (Query), and a document (Document), your mission is to {qc_mission}.

Task: {task.task_instruction}
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
