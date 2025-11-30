from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


class TaskType(Enum):
    # MIRACL: Web / Wikipedia passage retrieval
    miracl = "Given a question, retrieve Wikipedia passages that answer the question."
    # CovidRetrieval: Covid-19 news article retrieval
    covidretrieval = "Given a question on COVID-19, retrieve news articles that answer the question."


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
    task_instruction: str = None


def get_task(
    task_type: str,
    language: str,
) -> Task:
    """
    Helper to build a Task object from string names,
    e.g. get_task("miracl", "en") or get_task("covidretrieval", "zh").
    """
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
    narrative_focus: Optional[str] = None,   # ⭐ 新增
) -> str:
    task_to_gen_instruction: Dict[TaskType, str] = {
        TaskType.miracl: (
            "Given a Wikipedia passage in {language}, generate a question in {language} "
            "that the passage can answer."
        ),
        TaskType.covidretrieval: (
            "Given a news article related to COVID-19 in {language}, "
            "generate a question in {language} that the article can answer."
        ),
    }

    task_to_gen_output: Dict[TaskType, str] = {
        TaskType.miracl: "the generated question in {language}",
        TaskType.covidretrieval: "the generated question in {language}",
    }

    gen_instruction = task_to_gen_instruction[task.task_type]
    gen_output = task_to_gen_output[task.task_type]

    gen_instruction = gen_instruction.replace("{language}", task.language.value)
    gen_output = gen_output.replace("{language}", task.language.value)

    prefix = "The given content:"

    # ===== ⭐ CovidRetrieval 专用的 focus 提示 =====
    focus_hint = ""
    if task.task_type is TaskType.covidretrieval and narrative_focus is not None:
        if narrative_focus == "covid_fact_detail":
            focus_hint = (
                "\n\nFor this question, please focus on **concrete factual details** "
                "in the article (such as时间、地点、病例数、病毒变异信息、研究结论等)，"
                "提出一个可以通过本文直接回答的具体问题。"
            )
        elif narrative_focus == "covid_policy_measure":
            focus_hint = (
                "\n\nFor this question, please focus on **COVID-19 related policies or measures** "
                "(例如防控政策、出行管控、隔离要求、核酸/抗原检测规定、校园或社区管控措施等)，"
                "提出一个围绕政策或措施本身的提问。"
            )
        elif narrative_focus == "covid_vaccine_treatment":
            focus_hint = (
                "\n\nFor this question, please focus on **vaccines, drugs or clinical treatment** "
                "(例如疫苗类型、接种对象、保护效果、不良反应、药物试验结果、治疗方案等)，"
                "提出一个围绕疫苗/药物/临床研究的提问。"
            )
        elif narrative_focus == "covid_risk_protection":
            focus_hint = (
                "\n\nFor this question, please focus on **risk and protection** "
                "(例如传播风险、特定场景的感染概率、高危人群、个人防护建议、专家提醒等)，"
                "提出一个与风险评估或防护建议相关的问题。"
            )
        elif narrative_focus == "covid_social_impact":
            focus_hint = (
                "\n\nFor this question, please focus on **the social and economic impacts of COVID-19** "
                "(例如对经济、教育、就业、心理健康、社会秩序等方面的影响)，"
                "提出一个突出社会影响或长期后果的问题。"
            )
        # 其他值就当作 None 处理（不加约束）

    gen_prompt = f"""\
{gen_instruction}{focus_hint}

{prefix}
[Begin of Content]
{text}
[End of Content]

- Your output must always be a string, only containing {gen_output}.
- Your output should be independent of the given passage, which means that it should not contain the pronouns such as "it", "this", "that", "the given", "the provided", etc.
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

    gen_prompt += "Remember do not explain your output or output anything else. Your output:"

    return gen_prompt

def get_covid_doc_synthesis_prompt(
    task: Task,
    seed_text: str,
    extra_style_examples: Optional[List[str]] = None,
) -> str:
    """
    CovidRetrieval 专用：给定一篇高质量的中文疫情相关文章，生成一篇
    内容不同、但主题与文体相似的「新文章」。

    - 主要用于“doc 仿写 / 扩充 corpus”，不是生成 query。
    - 默认假设 language 是 zh，文章内容为中文。
    """
    lang = task.language.value  # 对 CovidRetrieval 来说通常是 "Simplified Chinese"

    prompt = f"""\
你是一名擅长撰写公共卫生和疫情防控相关文章的专业中文写作者。
现在给你一篇与新冠疫情相关的中文文章，作为「写作风格和结构」的参考样例。

你的任务是：
- 阅读参考文章的内容和结构，但**不要**直接照搬句子或段落；
- 写出一篇全新的中文文章，主题仍然与新冠疫情 / 防控 / 疫苗 / 复工复产等相关，
  但具体情节、数字、地名、机构名称等都要与参考文章不同；
- 你可以借鉴参考文章的写作节奏和信息组织方式（例如“背景 → 最新通报 → 防控措施 → 温馨提示”），
  但必须确保新文章在事实细节上是新的、合理的、自洽的；
- 新文章的内容应当具备一定的信息量，可以独立阅读，适合出现在新闻报道、健康科普或者政务发布中；
- 不要提到“参考文章”“原文”“以上内容”“如下所示”等元话语，也不要显式提到你在改写文章。

参考文章（仅作为风格与结构示例，请不要抄写原句）：
[参考文章开始]
{seed_text}
[参考文章结束]
"""

    if extra_style_examples:
        joined = "\n\n---\n\n".join(extra_style_examples)
        prompt += f"""\

以下是一些额外的参考文章片段（同样只用于把握风格，不允许直接照抄）：
[更多参考片段开始]
{joined}
[更多参考片段结束]
"""

    # 统一输出格式：Title / Desc，方便下游 split_title_desc 处理
    prompt += """\

请根据以上要求，直接输出一篇**全新的中文文章**，并严格使用以下格式：

Title: <一句话中文标题，简要点出文章核心内容>
Desc: <多段中文正文，每一段之间可以用空行分隔>

注意：
- 只输出上述两行结构，不要添加多余说明；
- 不要出现任何英文解释或元说明文字。

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

    - miracl: question vs Wikipedia passage
    - covidretrieval: Covid 问题 vs 疫情新闻文章
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