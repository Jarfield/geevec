from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


# ===== 1. 定义属性空间 =====

AREA_OF_LAW = [
    "criminal law",
    "financial and banking regulation",
    "public administration and corruption",
    "labour and employment law",
    "family and personal status law",
    "cybercrime and online communication",
    "environmental protection",
]

PRIMARY_ACTOR = [
    "a public servant or official",
    "a banker or financial intermediary",
    "an employer or supervisor",
    "a medical or healthcare professional",
    "an online platform operator or service provider",
    "a person having custody or care of a vulnerable person",
    "a landlord or property manager",
    "any member of the public",
]

VICTIM_TYPE = [
    "a member of the general public",
    "a customer or client",
    "a minor",
    "a vulnerable adult",
    "an employee or subordinate",
    "the government or public revenue",
    "the environment or public resources",
]

CONTEXT = [
    "in a physical, offline setting",
    "through electronic or online communication",
    "within an employment or workplace relationship",
    "in the provision of essential public or private services",
    "during the performance of public duties or official functions",
    "in the course of financial or commercial transactions",
]

CONDUCT_TYPE = [
    "abuse or misuse of a position of trust",
    "failure to perform a legally required duty",
    "unauthorised disclosure or misuse of confidential information",
    "manipulation, falsification, or destruction of records or evidence",
    "the use of threats, coercion, or harassment",
    "unlawful discrimination or targeted harassment",
    "unauthorised interference with or access to digital systems or data",
]

MENTAL_STATE = [
    "intentionally",
    "knowingly",
    "recklessly",
    "with gross negligence",
]

HARM_TYPE = [
    "causes or is likely to cause substantial financial loss",
    "causes or is likely to cause serious risk to life or bodily integrity",
    "causes serious psychological or emotional harm",
    "seriously undermines public trust in institutions",
    "obstructs or seriously interferes with the administration of justice",
    "causes significant or long-lasting environmental damage",
]

ENFORCEMENT_FOCUS = [
    "is usually difficult to detect without documentary or digital evidence",
    "typically involves a repeated or continuous pattern of behaviour",
    "is often committed through misuse of organisational or institutional power",
    "often leaves a digital or electronic trail",
    "is frequently committed against persons in a position of vulnerability",
]

SEVERITY_LEVEL = [
    "low",
    "medium",
    "high",
]

# === 新增：法条结构相关属性 ===

OFFENCE_FORM = [
    "a single, discrete act",
    "a continuing course of conduct",
    "an attempt, preparation, or incomplete offence",
    "abetment, instigation, or conspiracy to commit an offence",
]

PENALTY_TYPE = [
    "primarily punishable with imprisonment only",
    "primarily punishable with fine only",
    "punishable with both imprisonment and fine",
    "punishable with a graduated scale of penalties depending on the gravity of harm",
]

EXCEPTIONS_CLAUSE = [
    "includes explicit defences, exceptions, or lawful justifications",
    "does not contain any explicit defences or exceptions and operates as a strict rule",
]


# ===== 2. 属性数据结构 =====

@dataclass
class StatuteAttributeBundle:
    """一组高层次的 statute 属性，用来指导 LLM 生成新条文。"""
    area_of_law: str
    primary_actor: str
    victim_type: str
    context: str
    conduct_type: str
    mental_state: str
    harm_type: str
    enforcement_focus: str
    severity_level: str

    offence_form: str          # ★ 新增
    penalty_type: str          # ★ 新增
    exceptions_clause: str     # ★ 新增

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===== 3. 采样与转文本工具 =====

def _get_rng(rng: Optional[random.Random] = None) -> random.Random:
    """内部小工具：如果没传 rng，就用全局 random。"""
    return rng or random


def sample_statute_attributes(
    rng: Optional[random.Random] = None,
) -> StatuteAttributeBundle:
    """
    随机采样一组 statute 属性，用于指导合成法条生成。

    如果想复现实验，可以在外部创建 random.Random(seed)，再传进来。
    """
    r = _get_rng(rng)
    return StatuteAttributeBundle(
        area_of_law=r.choice(AREA_OF_LAW),
        primary_actor=r.choice(PRIMARY_ACTOR),
        victim_type=r.choice(VICTIM_TYPE),
        context=r.choice(CONTEXT),
        conduct_type=r.choice(CONDUCT_TYPE),
        mental_state=r.choice(MENTAL_STATE),
        harm_type=r.choice(HARM_TYPE),
        enforcement_focus=r.choice(ENFORCEMENT_FOCUS),
        severity_level=r.choice(SEVERITY_LEVEL),
        offence_form=r.choice(OFFENCE_FORM),
        penalty_type=r.choice(PENALTY_TYPE),
        exceptions_clause=r.choice(EXCEPTIONS_CLAUSE),
    )


def attributes_to_hint_text(bundle: StatuteAttributeBundle) -> str:
    """
    将一组属性转换成可以直接拼到 LLM prompt 里的英文说明。
    """
    return f"""\
When drafting the new statute, you must respect the following high-level attributes:

- Area of law: The provision should primarily fall within {bundle.area_of_law}.
- Primary actor: It should regulate conduct mainly by {bundle.primary_actor}.
- Victim or protected interest: It should protect {bundle.victim_type}.
- Typical context: The conduct usually occurs {bundle.context}.
- Nature of conduct: The core wrongdoing involves {bundle.conduct_type}.
- Mental state: The offence should require that the actor acts {bundle.mental_state}.
- Harm or risk: The prohibited conduct {bundle.harm_type}.
- Enforcement focus: The offence {bundle.enforcement_focus}.
- Overall severity: The offence should be treated as a {bundle.severity_level}-severity criminal offence,
  with penalties that are proportionate to this level (for example, a mixture of imprisonment and/or fine).
- Offence form: The offence should be structured as {bundle.offence_form}.
- Penalty structure: The offence is {bundle.penalty_type}.
- Exceptions or defences: The statute {bundle.exceptions_clause}.

These attributes are soft constraints that should meaningfully shape the type of conduct,
actors, harms, and legal structure regulated by the new statute, while still allowing you
to write a natural and coherent legal provision.
"""


__all__ = [
    "StatuteAttributeBundle",
    "sample_statute_attributes",
    "attributes_to_hint_text",
]
