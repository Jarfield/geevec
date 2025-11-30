# optim.py
# ============================================
# 优化器：SGD（随机梯度下降，支持 GPU 张量）
# ============================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np  # 类型标注用
from models import Model


class Optimizer(ABC):
    """
    优化器抽象基类。
    """

    @abstractmethod
    def step(self) -> None:
        """
        使用当前梯度更新参数。
        """
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self) -> None:
        """
        将所有梯度清零。
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    随机梯度下降（Stochastic Gradient Descent）

    公式：
        param ← param - lr * grad

    说明：
        - param / grad 实际为 backend.xp.ndarray（cupy，GPU 上）
    """

    def __init__(self, model: Model, lr: float = 1e-2) -> None:
        self.model = model
        self.lr = lr

    def step(self) -> None:
        params: List[np.ndarray] = self.model.params()
        grads: List[np.ndarray] = self.model.grads()

        assert len(params) == len(grads), "params and grads length mismatch"

        for p, g in zip(params, grads):
            # 有些层可能没有参数，其 grads() 返回空列表，这里不影响
            if g is None:
                continue
            # 这里的 p, g 实际是 cupy.ndarray，运算会在 GPU 上进行
            p -= self.lr * g

    def zero_grad(self) -> None:
        """
        将所有梯度清零，避免下一次 step 时累加旧梯度。
        """
        grads: List[np.ndarray] = self.model.grads()
        for g in grads:
            if g is not None:
                g.fill(0.0)  # cupy.ndarray.fill 在 GPU 上就地置零
