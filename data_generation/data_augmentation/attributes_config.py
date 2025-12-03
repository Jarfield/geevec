"""Attribute samplers and prompt builders for different tasks.

Each task exposes two functions:
- sample_<task>_attributes(): returns a dataclass bundle of soft constraints
- <task>_attributes_to_hint_text(bundle): converts the bundle into prompt text

A light-weight dispatcher is provided for consumers that want to stay task-agnostic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

from constant import TaskType, Task
from constant import get_covid_doc_synthesis_prompt

# ===== Covid Retrieval attributes =====
DOC_TYPE = [
    "新闻简讯",
    "深度报道",
    "政策解读文章",
    "专家访谈稿",
    "问答式科普文章",
    "辟谣和澄清公告",
]

SOURCE_TYPE = [
    "国家级权威媒体",
    "地方电视台或报纸",
    "政府官方网站或政务公众号",
    "疾控中心或卫生健康部门",
    "三甲医院或医务人员",
    "互联网门户网站或专业科普平台",
    "社交平台上的个人账号或自媒体",
]

TIME_PHASE = [
    "疫情暴发初期",
    "本地病例快速上升阶段",
    "疫情基本得到控制阶段",
    "大规模疫苗接种推进阶段",
    "出现新的变异毒株并引发局部反弹阶段",
    "逐步恢复常态化防控和生产生活秩序阶段",
]

MAIN_TOPIC = [
    "本地新增病例通报和流调信息",
    "风险地区调整和行程管理政策",
    "核酸或抗原检测安排与注意事项",
    "隔离、封控和居家健康监测措施",
    "疫苗研发进展和临床试验信息",
    "疫苗接种政策、流程及注意事项",
    "个人防护措施和健康生活方式建议",
    "重点人群（如老年人、慢性病患者）健康管理",
    "复工复产和校园复课的防疫安排",
    "涉疫谣言澄清与错误信息辟谣",
    "疫情期间的心理健康与人文关怀",
]

GEOGRAPHY_SCOPE = [
    "以某一个社区或区县为重点",
    "以某个城市为重点",
    "以某个省份或大区为重点",
    "覆盖全国范围的总体情况通报",
    "适度对比不同国家或地区的防控经验",
]

POPULATION_FOCUS = [
    "老年人群",
    "儿童和在校学生",
    "医务人员和其他一线工作者",
    "慢性病患者等高风险人群",
    "普通城市居民",
    "农村或偏远地区居民",
    "来访人员和境外输入相关人员",
]

DATA_STYLE = [
    "包含较多具体数字和时间节点（如病例数、接种量）",
    "以文字叙事为主，仅适度提及关键数据",
    "重点列出条目式政策要点和执行要求",
    "通过个体或家庭故事引出政策和数据背景",
]

TONE = [
    "整体语气正式、客观、中性",
    "整体语气偏安抚、鼓励和积极正面",
    "整体语气偏提醒和风险警示",
    "整体语气偏回顾和经验总结",
]


@dataclass
class CovidArticleAttributeBundle:
    doc_type: str
    source_type: str
    time_phase: str
    main_topic: str
    geography_scope: str
    population_focus: str
    data_style: str
    tone: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===== AILA statute attributes (simplified) =====
@dataclass
class StatuteAttributeBundle:
    area_of_law: str
    conduct_type: str
    severity_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


STATUTE_AREA = [
    "public administration and corruption",
    "financial misconduct",
    "cybercrime and data protection",
    "violent offences",
    "environmental protection",
]
STATUTE_CONDUCT = [
    "abuse of a position of trust",
    "misuse of confidential information",
    "fraudulent misrepresentation",
    "unauthorised interference with digital systems",
    "neglect of a statutory duty",
]
STATUTE_SEVERITY = ["low", "medium", "high"]


# ===== SCIDOCS attributes =====
@dataclass
class SciDocsAttributeBundle:
    field: str
    contribution_type: str
    evidence_depth: str
    tone: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SCIDOC_FIELDS = [
    "computer science",
    "biology",
    "medicine",
    "physics",
    "social sciences",
]
SCIDOC_CONTRIBUTION = [
    "survey of recent work",
    "methodology introduction",
    "benchmark result report",
    "case study",
    "theoretical analysis",
]
SCIDOC_EVIDENCE = [
    "high-level summary with few details",
    "balanced mix of background and results",
    "result-heavy description with metrics",
]
SCIDOC_TONE = ["neutral academic", "enthusiastic", "cautiously optimistic"]


# ===== ArguAna attributes =====
@dataclass
class ArguAnaAttributeBundle:
    stance: str
    domain: str
    evidence_style: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ARGUANA_STANCE = ["support", "oppose", "neutral overview"]
ARGUANA_DOMAIN = [
    "public policy",
    "technology ethics",
    "environmental regulation",
    "healthcare",
    "education",
]
ARGUANA_EVIDENCE = [
    "uses statistics and studies",
    "relies on expert testimony",
    "anecdotal narrative",
    "comparative reasoning",
]


# ===== Generic helpers =====
def _get_rng(rng: Optional[random.Random] = None) -> random.Random:
    return rng or random


def sample_covid_attributes(rng: Optional[random.Random] = None) -> CovidArticleAttributeBundle:
    r = _get_rng(rng)
    return CovidArticleAttributeBundle(
        doc_type=r.choice(DOC_TYPE),
        source_type=r.choice(SOURCE_TYPE),
        time_phase=r.choice(TIME_PHASE),
        main_topic=r.choice(MAIN_TOPIC),
        geography_scope=r.choice(GEOGRAPHY_SCOPE),
        population_focus=r.choice(POPULATION_FOCUS),
        data_style=r.choice(DATA_STYLE),
        tone=r.choice(TONE),
    )


def covid_attributes_to_hint_text(bundle: CovidArticleAttributeBundle) -> str:
    return f"""\
在撰写新的疫情相关文章时，请同时满足以下高层次属性要求：
- 文章类型：整体应当呈现为「{bundle.doc_type}」风格。
- 信息来源/叙事主体：文章语气和视角应主要符合「{bundle.source_type}」的口吻和身份。
- 疫情阶段：描述的背景大致处于「{bundle.time_phase}」，在行文中体现该阶段的特征。
- 主要主题：文章主要围绕「{bundle.main_topic}」展开。
- 空间范围：报道或说明的重点区域应是「{bundle.geography_scope}」。
- 重点人群：文章应特别关注「{bundle.population_focus}」。
- 数据与叙事风格：写作时应当「{bundle.data_style}」。
- 整体语气基调：全文的语气应当「{bundle.tone}」。
这些属性是软约束，用来塑造行文方向。"""


def sample_statute_attributes(rng: Optional[random.Random] = None) -> StatuteAttributeBundle:
    r = _get_rng(rng)
    return StatuteAttributeBundle(
        area_of_law=r.choice(STATUTE_AREA),
        conduct_type=r.choice(STATUTE_CONDUCT),
        severity_level=r.choice(STATUTE_SEVERITY),
    )


def statute_attributes_to_hint_text(bundle: StatuteAttributeBundle) -> str:
    return f"""\
When drafting the statute, keep it within {bundle.area_of_law}, targeting conduct such as {bundle.conduct_type}.
Treat the offence as {bundle.severity_level}-severity and make the penalties proportionate. Write it as a self-contained clause.\n"""


def sample_scidocs_attributes(rng: Optional[random.Random] = None) -> SciDocsAttributeBundle:
    r = _get_rng(rng)
    return SciDocsAttributeBundle(
        field=r.choice(SCIDOC_FIELDS),
        contribution_type=r.choice(SCIDOC_CONTRIBUTION),
        evidence_depth=r.choice(SCIDOC_EVIDENCE),
        tone=r.choice(SCIDOC_TONE),
    )


def scidocs_attributes_to_hint_text(bundle: SciDocsAttributeBundle) -> str:
    return f"""\
Shape the synthetic abstract as a {bundle.contribution_type} piece in the field of {bundle.field}.
Keep the level of detail as "{bundle.evidence_depth}" and maintain a {bundle.tone} tone throughout.\n"""


def sample_arguana_attributes(rng: Optional[random.Random] = None) -> ArguAnaAttributeBundle:
    r = _get_rng(rng)
    return ArguAnaAttributeBundle(
        stance=r.choice(ARGUANA_STANCE),
        domain=r.choice(ARGUANA_DOMAIN),
        evidence_style=r.choice(ARGUANA_EVIDENCE),
    )


def arguana_attributes_to_hint_text(bundle: ArguAnaAttributeBundle) -> str:
    return f"""\
Write a passage about {bundle.domain} that clearly takes a {bundle.stance} stance.
Support the argument using {bundle.evidence_style}.\n"""


# ===== Dispatcher =====
ATTRIBUTE_SAMPLERS = {
    TaskType.covidretrieval: sample_covid_attributes,
    TaskType.ailastatutes: sample_statute_attributes,
    TaskType.scidocs: sample_scidocs_attributes,
    TaskType.arguana: sample_arguana_attributes,
}

ATTRIBUTE_RENDERERS = {
    TaskType.covidretrieval: covid_attributes_to_hint_text,
    TaskType.ailastatutes: statute_attributes_to_hint_text,
    TaskType.scidocs: scidocs_attributes_to_hint_text,
    TaskType.arguana: arguana_attributes_to_hint_text,
}


def sample_attributes_for_task(task_type: TaskType, rng: Optional[random.Random] = None):
    sampler = ATTRIBUTE_SAMPLERS.get(task_type)
    if sampler is None:
        return None
    return sampler(rng=rng)


def attributes_to_hint_text(task_type: TaskType, bundle: Any) -> str:
    if bundle is None:
        return ""
    renderer = ATTRIBUTE_RENDERERS.get(task_type)
    return renderer(bundle) if renderer is not None else ""


def get_base_synthesis_prompt(task: Task, seed_text: str) -> str:
    """Pick the right base synthesis prompt for a task."""
    if task.task_type is TaskType.covidretrieval:
        return get_covid_doc_synthesis_prompt(task=task, seed_text=seed_text, extra_style_examples=None)

    if task.task_type is TaskType.ailastatutes:
        return f"""\
You are drafting a new statute in {task.language.value}. Read the reference statute below as a template for style,
but produce a fresh provision that targets a *different* fact pattern within the same legal area.
Avoid copying sentences verbatim and do not mention section numbers.

[Reference Statute]
{seed_text}
[End Reference]

Write the new statute as a standalone clause without bullet points.
"""

    if task.task_type is TaskType.scidocs:
        return f"""\
You are rewriting a scientific abstract in {task.language.value}. Use the reference abstract for tone and structure,
but produce a new abstract that covers a different study with self-consistent details.

[Reference Abstract]
{seed_text}
[End Reference]
"""

    if task.task_type is TaskType.arguana:
        return f"""\
You are writing an argumentative passage in {task.language.value}. Use the reference paragraph for style,
but create a new passage on a related topic with different claims and evidence.

[Reference Passage]
{seed_text}
[End Reference]
"""

    raise ValueError(f"No synthesis prompt defined for task {task.task_type}")


__all__ = [
    "CovidArticleAttributeBundle",
    "sample_covid_attributes",
    "covid_attributes_to_hint_text",
    "StatuteAttributeBundle",
    "sample_statute_attributes",
    "statute_attributes_to_hint_text",
    "SciDocsAttributeBundle",
    "sample_scidocs_attributes",
    "scidocs_attributes_to_hint_text",
    "ArguAnaAttributeBundle",
    "sample_arguana_attributes",
    "arguana_attributes_to_hint_text",
    "sample_attributes_for_task",
    "attributes_to_hint_text",
    "get_base_synthesis_prompt",
]
