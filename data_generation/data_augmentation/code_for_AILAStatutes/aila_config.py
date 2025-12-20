"""Configuration helpers for the AILA statutes pipeline."""

import os
import sys
from dataclasses import dataclass

# 将 data_augmentation 的 code 目录加入系统路径，复用公共 LLM 与生成器逻辑
CURRENT_DIR = os.path.dirname(__file__)
PARENT_CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "code"))
if PARENT_CODE_DIR not in sys.path:
    sys.path.append(PARENT_CODE_DIR)


@dataclass(frozen=True)
class AILAFieldMapping:
    """Dataset-specific field names."""

    id_key: str = "_id"
    text_key: str = "text"
    title_key: str = "title"


class AILAPaths:
    """Centralised default paths for AILA statute generation artifacts."""

    GENERATED_ROOT = os.path.join("data", "generated_data", "ailastatutes")

    @classmethod
    def ensure_dir(cls, *parts: str) -> str:
        path = os.path.join(cls.GENERATED_ROOT, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def summaries_dir(cls) -> str:
        return cls.ensure_dir("layman_summaries")

    @classmethod
    def scenarios_dir(cls) -> str:
        return cls.ensure_dir("base_scenarios")

    @classmethod
    def evolutions_dir(cls) -> str:
        return cls.ensure_dir("evolutions")

    @classmethod
    def near_miss_dir(cls) -> str:
        return cls.ensure_dir("near_miss_negatives")


__all__ = ["AILAFieldMapping", "AILAPaths"]
