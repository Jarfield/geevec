from enum import Enum
from dataclasses import dataclass
from typing import Optional, List


class TaskType(Enum):
    # 这里只保留 CovidRetrieval
    covidretrieval = "Given a question on COVID-19, retrieve news articles that answer the question."


class Language(Enum):
    # 只保留中文
    zh = "Simplified Chinese"  # 简体中文


@dataclass
class Task:
    task_type: TaskType
    language: Language
    task_instruction: str = None


def get_task(
    task_type: str,
    language: str,
) -> Task:
    task_instruction = TaskType[task_type].value

    task = Task(
        task_type=TaskType[task_type],
        language=Language[language],
        task_instruction=task_instruction,
    )
    return task


def get_generation_prompt(
    task: Task,
    text: str,
    examples: Optional[List[dict]] = None,
) -> str:
    """
    对 CovidRetrieval 语料打分：
    让 LLM 只输出 1–5 的整数（字符串），综合衡量：
    1）与“新冠 / COVID-19 疫情与防控新闻”的相关性；
    2）信息质量（是否提供具体、可靠、面向公众的信息）。
    """
    if task.task_type != TaskType.covidretrieval:
        raise ValueError(f"Unsupported task_type: {task.task_type}")

    # ===== 主指令：1–5 相关性 + 质量打分（尽可能严格） =====
    gen_instruction = (
        "You are given a document written in {language}. "
        "Your task is to read the document carefully and assign a single integer quality "
        "score from 1 to 5 indicating **how suitable this document is to be used as a "
        "COVID-19 (coronavirus) news / information article**.\n\n"
        "You must consider BOTH:\n"
        "(A) topical relevance to COVID-19 (epidemic, coronavirus, SARS-CoV-2, variants, vaccines, "
        "public-health measures, etc.), AND\n"
        "(B) information quality: whether the content actually provides concrete, factual, and useful "
        "information for the public (not just vague slogans or generic emotional text).\n\n"
        "Scoring guidelines (1 = worst, 5 = best):\n"
        "1 – Completely unsuitable. One of the following:\n"
        "    • Totally unrelated to COVID-19 (e.g., mobile game ads like \"一刀999\", 小说、八卦、情感故事、泛娱乐内容、购物促销等);\n"
        "    • Only contains extremely vague or decorative mentions of the epidemic (for example just "
        "      saying \"疫情期间大家都很辛苦\" with no real information);\n"
        "    • Obvious spam, nonsense text, or content that is clearly not news / information.\n"
        "2 – Weakly suitable. One of the following:\n"
        "    • The document occasionally mentions words like \"疫情\" or \"新冠\" but the **main topic is not** "
        "      COVID-19 news or information (for example, mainly promoting products、讲情感故事，只顺带提到疫情背景);\n"
        "    • Or it talks about very general health /养生/免疫力建议 without clearly focusing on COVID-19;\n"
        "    • Or the document is very short / superficial and does not provide concrete, verifiable facts.\n"
        "3 – Borderline / moderately suitable:\n"
        "    • COVID-19 is an important theme, but the article mixes a lot of其他生活话题、泛讨论，"
        "      so it is not a clean COVID-19 news article;\n"
        "    • Or it contains some useful COVID-related facts (e.g. basic政策、简单防护建议), "
        "      but the information is limited, incomplete, or not clearly structured as news/information;\n"
        "    • Such documents may be kept only if we use a low threshold, but you should NOT give 4 or 5.\n"
        "4 – Good COVID-19 information article:\n"
        "    • The majority of the content is clearly about COVID-19 (疫情形势、确诊/死亡数据、变异株、疫苗接种、"
        "      防控政策、医学研究结果等);\n"
        "    • It provides reasonably concrete information (such as dates, locations,明确政策措施、官方通告要点等), "
        "      with limited unrelated content;\n"
        "    • It is suitable as a training document, but may still contain some general narrative.\n"
        "5 – Very high-quality, strongly suitable COVID-19 news / information article:\n"
        "    • The document is almost entirely focused on COVID-19 **and** presents clear, concrete, "
        "      and informative content: e.g., case/death statistics, government or WHO announcements, "
        "      detailed vaccine/药物试验结果, public-health measures, or rigorous expert explanations;\n"
        "    • It reads like a serious news report, official notice, or专业科普文章 about COVID-19, with "
        "      minimal unrelated content.\n\n"
        "IMPORTANT STRICT RULES:\n"
        "    • If the document is clearly a game advertisement (for example, texts about \"一刀999\", "
        "      \"传奇\", \"手游充值\"), shopping promotion,泛娱乐内容, or anything obviously unrelated "
        "      to COVID-19, you MUST output 1.\n"
        "    • If COVID-19 is only used as a背景/噱头 and the main purpose is selling products, telling "
        "      romance stories, or general life感想, you MUST NOT output more than 2.\n"
        "    • You should only output 4 or 5 when BOTH topical relevance and information quality are high.\n\n"
        "Think carefully and then output ONLY one integer from 1, 2, 3, 4, or 5."
    ).replace("{language}", task.language.value)

    gen_output = "a single integer (1, 2, 3, 4, or 5) as the quality score"

    prefix = "The given content:"

    constraints = "\n".join(
        [
            f"- Your output must always be a string, only containing {gen_output}.",
            "- Do not output any other words, symbols, or explanations.",
            "- Do not output spaces or line breaks before or after the number.",
            "- Valid outputs are exactly one of: 1, 2, 3, 4, 5.",
        ]
    )

    gen_prompt = f"""\
{gen_instruction}

{prefix}
[Begin of Content]
{text}
[End of Content]

{constraints}

"""

    # few-shot 示例（可选）
    if examples is not None:
        examples_str_list = [
            f"""\
- Example {i + 1}:
    {prefix}
    [Begin of Content]
    {example['input']}
    [End of Content]
    
    Expected Output (a single integer 1–5): {example['output']}

"""
            for i, example in enumerate(examples)
        ]

        gen_prompt += f"""\
Here are a few examples for your reference:
{''.join(examples_str_list)}
"""

    gen_prompt += "Remember do not explain your output or output anything else. Your output:"

    return gen_prompt
