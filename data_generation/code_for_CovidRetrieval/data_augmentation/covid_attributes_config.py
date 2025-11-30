from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

# ===== 1. 定义属性空间：围绕「疫情资讯/报道」设计 =====

# 文稿类型：新闻、政策解读、科普、辟谣等
DOC_TYPE = [
    "新闻简讯",
    "深度报道",
    "政策解读文章",
    "专家访谈稿",
    "问答式科普文章",
    "辟谣和澄清公告",
]

# 信息来源/叙事主体
SOURCE_TYPE = [
    "国家级权威媒体",
    "地方电视台或报纸",
    "政府官方网站或政务公众号",
    "疾控中心或卫生健康部门",
    "三甲医院或医务人员",
    "互联网门户网站或专业科普平台",
    "社交平台上的个人账号或自媒体",
]

# 所处疫情阶段（模糊描述，避免和真实时间强绑定）
TIME_PHASE = [
    "疫情暴发初期",
    "本地病例快速上升阶段",
    "疫情基本得到控制阶段",
    "大规模疫苗接种推进阶段",
    "出现新的变异毒株并引发局部反弹阶段",
    "逐步恢复常态化防控和生产生活秩序阶段",
]

# 文章主要主题/切入点
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

# 空间范围：聚焦哪一级别
GEOGRAPHY_SCOPE = [
    "以某一个社区或区县为重点",
    "以某个城市为重点",
    "以某个省份或大区为重点",
    "覆盖全国范围的总体情况通报",
    "适度对比不同国家或地区的防控经验",
]

# 重点关注的人群/对象
POPULATION_FOCUS = [
    "老年人群",
    "儿童和在校学生",
    "医务人员和其他一线工作者",
    "慢性病患者等高风险人群",
    "普通城市居民",
    "农村或偏远地区居民",
    "来访人员和境外输入相关人员",
]

# 数据/叙事风格
DATA_STYLE = [
    "包含较多具体数字和时间节点（如病例数、接种量）",
    "以文字叙事为主，仅适度提及关键数据",
    "重点列出条目式政策要点和执行要求",
    "通过个体或家庭故事引出政策和数据背景",
]

# 整体语气/基调
TONE = [
    "整体语气正式、客观、中性",
    "整体语气偏安抚、鼓励和积极正面",
    "整体语气偏提醒和风险警示",
    "整体语气偏回顾和经验总结",
]


# ===== 2. 属性数据结构 =====

@dataclass
class CovidArticleAttributeBundle:
    """
    一组高层次的 Covid 文稿属性，用来在 prompt 中约束 LLM 生成
    「什么类型、什么视角、什么主题、什么语气」的疫情相关文章。
    """
    doc_type: str           # 文稿类型
    source_type: str        # 来源/叙事主体
    time_phase: str         # 疫情阶段
    main_topic: str         # 主要主题
    geography_scope: str    # 空间范围
    population_focus: str   # 重点人群
    data_style: str         # 数据/叙事风格
    tone: str               # 整体语气

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===== 3. 采样与转文本工具 =====

def _get_rng(rng: Optional[random.Random] = None) -> random.Random:
    """内部小工具：如果没传 rng，就用全局 random。"""
    return rng or random


def sample_covid_attributes(
    rng: Optional[random.Random] = None,
) -> CovidArticleAttributeBundle:
    """
    随机采样一组 Covid 文章属性，用于指导合成文稿生成。

    如果想复现实验，可以在外部创建 random.Random(seed)，再传进来。
    """
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


def attributes_to_hint_text(bundle: CovidArticleAttributeBundle) -> str:
    """
    将一组属性转换成可以直接拼到 LLM prompt 里的中文说明。
    """
    return f"""\
在撰写新的疫情相关中文文章时，请同时满足以下高层次属性要求（它们是软约束，但应对文章的类型、视角和内容起到实质性影响）：

- 文章类型：整体应当呈现为「{bundle.doc_type}」风格。
- 信息来源/叙事主体：文章语气和视角应主要符合「{bundle.source_type}」的口吻和身份。
- 疫情阶段：描述的背景大致处于「{bundle.time_phase}」，在行文中可以适度体现该阶段的特征。
- 主要主题：文章主要围绕「{bundle.main_topic}」展开，其他内容只作辅助。
- 空间范围：报道或说明的重点区域应当是「{bundle.geography_scope}」。
- 重点人群：文章应特别关注「{bundle.population_focus}」，在内容中体现其处境、需求或相关安排。
- 数据与叙事风格：写作时应当「{bundle.data_style}」，注意控制数据的密度与呈现方式。
- 整体语气基调：全文的语气应当「{bundle.tone}」，避免与该基调明显冲突的表达方式。

这些属性是对文章类型、关注对象和叙事方式的整体指导，而不是逐句的硬性限制。
你可以在不违背上述属性的前提下，写出连贯、自然、信息充分的中文文章。
"""


__all__ = [
    "CovidArticleAttributeBundle",
    "sample_covid_attributes",
    "attributes_to_hint_text",
]
