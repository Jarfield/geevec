"""数据增强所需的属性采样与提示构建器。

核心设计：
- 提供一套通用的叙事属性字段，便于跨任务复用。
- 每个任务可以在通用属性的基础上附加少量领域特定的取值，
  以保证灵活性而不过度耦合。
- 产出的提示文本直接用于 `get_doc_synthesis_prompt` 的
  `narrative_attributes` 参数，从而让文档改写保持一致的接口。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

from constant import TaskType, Task, get_doc_synthesis_prompt

# ======================== 通用属性定义 ========================
@dataclass
class NarrativeAttributeBundle:
    """描述文档改写的高层次叙事属性。"""

    genre: str
    tone: str
    detail_level: str
    structure: str
    audience: str
    length: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BASE_GENRES = [
    "新闻报道风格",
    "百科式解释风格",
    "评论员口吻",
    "专家解读",
    "问答式说明",
]

BASE_TONES = [
    "中性客观",
    "积极鼓励",
    "严肃谨慎",
    "理性分析",
]

BASE_DETAIL = [
    "以事实细节为主，提供充足的数据和时间线",
    "叙事为主，数据点适度点缀",
    "侧重政策条目或要点列表",
]

BASE_STRUCTURE = [
    "先给结论再展开论据",
    "按时间顺序讲述",
    "先背景再观点，最后总结",
]

BASE_AUDIENCE = [
    "面向普通大众",
    "面向专业读者",
    "面向政策制定者",
    "面向学生或初学者",
]

BASE_LENGTH = [
    "适中篇幅（2-4段）",
    "较长篇幅（5-7段）",
    "简洁短篇（1-2段）",
]


# ======================== 任务特定加成 ========================
TASK_SPECIFIC_OPTIONS: Dict[TaskType, Dict[str, List[str]]] = {
    TaskType.covidretrieval: {
        "genre": ["疫情通报", "防疫政策解读", "健康科普指南"],
        "tone": ["安抚式", "提醒式"],
        "audience": ["面向社区居民", "面向医务人员"],
    },
    TaskType.ailastatutes: {
        "genre": ["法律条文式叙述"],
        "tone": ["正式权威"],
        "structure": ["分点列出义务与责任", "单段落的完整法律表述"],
    },
    TaskType.scidocs: {
        "genre": ["学术摘要风格", "研究亮点综述"],
        "tone": ["学术中性"],
        "audience": ["面向研究人员", "面向跨学科读者"],
    },
    TaskType.arguana: {
        "genre": ["论证性评论", "议题辩论摘要"],
        "tone": ["立场鲜明"],
        "structure": ["提出观点-列证据-得结论"],
    },
}


def _get_rng(rng: Optional[random.Random] = None) -> random.Random:
    return rng or random


def _sample_from_base(rng: random.Random, task_type: TaskType) -> NarrativeAttributeBundle:
    """从通用集合中采样，并融合任务特定的可选项。"""

    def pick(key: str, base: List[str]) -> str:
        candidates = list(base)
        task_opts = TASK_SPECIFIC_OPTIONS.get(task_type, {}).get(key, [])
        candidates.extend(task_opts)
        return rng.choice(candidates)

    return NarrativeAttributeBundle(
        genre=pick("genre", BASE_GENRES),
        tone=pick("tone", BASE_TONES),
        detail_level=pick("detail_level", BASE_DETAIL),
        structure=pick("structure", BASE_STRUCTURE),
        audience=pick("audience", BASE_AUDIENCE),
        length=pick("length", BASE_LENGTH),
    )


# ======================== 外部接口 ========================
def sample_attributes_for_task(task_type: TaskType, rng: Optional[random.Random] = None):
    """按任务采样叙事属性。未声明的任务也会返回通用配置。"""
    rng = _get_rng(rng)
    return _sample_from_base(rng, task_type)


def attributes_to_hint_text(bundle: Optional[NarrativeAttributeBundle]) -> str:
    """将属性捏合为可读的提示文本，便于直接嵌入 prompt。"""
    if bundle is None:
        return ""

    return (
        "请在改写时满足以下高层次要求："
        f"\n- 文档体裁：{bundle.genre}"
        f"\n- 行文语气：{bundle.tone}"
        f"\n- 细节深度：{bundle.detail_level}"
        f"\n- 结构方式：{bundle.structure}"
        f"\n- 目标读者：{bundle.audience}"
        f"\n- 篇幅：{bundle.length}"
    )


def get_base_synthesis_prompt(
    task: Task,
    seed_text: str,
    narrative_hint: Optional[str] = None,
    extra_style_examples: Optional[List[str]] = None,
) -> str:
    """封装对 `get_doc_synthesis_prompt` 的调用，便于向后兼容。"""
    return get_doc_synthesis_prompt(
        task=task,
        seed_text=seed_text,
        narrative_attributes=narrative_hint or None,
        extra_style_examples=extra_style_examples,
    )


__all__ = [
    "NarrativeAttributeBundle",
    "sample_attributes_for_task",
    "attributes_to_hint_text",
    "get_base_synthesis_prompt",
]
