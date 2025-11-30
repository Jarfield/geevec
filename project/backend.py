# backend.py
# ============================================
# GPU-only 数组后端：
#   - 统一用 cupy 作为 xp
#   - 提供 to_cpu 用于做指标 / 保存模型
# ============================================

from __future__ import annotations

import numpy as np
import cupy as cp


# 训练 /前向 / 反向 全部用 xp = cp
xp = cp


def to_cpu(x):
    """
    将 cupy.ndarray 转成 numpy.ndarray：
    - 用于 accuracy / save_model 等需要在 CPU 上操作的场景
    """
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x
