# loss.py
# ============================================
# 损失函数：Softmax + CrossEntropy（GPU 版）
# 使用 backend.xp（cupy）计算
# ============================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np  # 仅用于类型标注 / float 转换
from backend import xp  # 实际计算用的数组库：cupy


class Loss(ABC):
    """
    损失函数抽象基类。
    约定：
      - forward(pred, target) 返回标量损失 (float)
      - backward() 返回对 pred 的梯度（形状与 pred 相同，xp.ndarray）
    """

    @abstractmethod
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        pred / target 实际上是 xp.ndarray（GPU 上），这里类型标注沿用 numpy 风格。
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self) -> np.ndarray:
        raise NotImplementedError


class SoftmaxCrossEntropyLoss(Loss):
    """
    Softmax + 交叉熵损失，多分类常用组合。

    使用方式：
        loss_fn = SoftmaxCrossEntropyLoss()
        loss = loss_fn.forward(logits, targets)
        grad_logits = loss_fn.backward()

    约定：
      - logits: (N, C)，xp.ndarray
      - targets:
          * 若是整数类别: (N,)，每个元素在 [0, C-1]
          * 若是 one-hot: (N, C)
    """

    def __init__(self) -> None:
        # 缓存中间结果用于 backward（实际会存 xp.ndarray）
        self._logits: Optional[np.ndarray] = None
        self._probs: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None
        self._loss: Optional[float] = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        logits: (N, C)，xp.ndarray
        targets: (N,) 或 (N, C)，xp.ndarray
        返回：标量损失（float，在 CPU 上）
        """
        # 保存原始 logits 和 targets
        self._logits = logits
        self._targets = targets

        # 数值稳定的 softmax：减去每行最大值
        shifted = logits - xp.max(logits, axis=1, keepdims=True)
        exp_scores = xp.exp(shifted)
        probs = exp_scores / xp.sum(exp_scores, axis=1, keepdims=True)  # (N, C)
        self._probs = probs

        N, C = logits.shape

        # 处理 targets 既可以是整数类别也可以是 one-hot
        if targets.ndim == 1:
            # 整数标签
            if targets.shape[0] != N:
                raise ValueError("targets length must match batch size")

            # 取出每个样本对应类别的概率
            correct_logprobs = -xp.log(probs[xp.arange(N), targets] + 1e-12)
            loss = float(xp.mean(correct_logprobs))
        elif targets.ndim == 2:
            # one-hot
            if targets.shape != probs.shape:
                raise ValueError("one-hot targets must have same shape as logits/probs")

            # 交叉熵：-sum(y * log(p))，再对 batch 取平均
            correct_logprobs = -xp.sum(targets * xp.log(probs + 1e-12), axis=1)
            loss = float(xp.mean(correct_logprobs))
        else:
            raise ValueError("targets must be 1D (class indices) or 2D (one-hot)")

        self._loss = loss
        return loss

    def backward(self) -> np.ndarray:
        """
        返回对 logits 的梯度，形状 (N, C)，xp.ndarray（GPU 上）。

        Softmax + CrossEntropy 的简洁公式：
            grad = (probs - y_one_hot) / N
        """
        if self._logits is None or self._probs is None or self._targets is None:
            raise RuntimeError("Must call forward() before backward().")

        logits = self._logits
        probs = self._probs
        targets = self._targets

        N, C = logits.shape

        # 构建 one-hot 形式的标签
        if targets.ndim == 1:
            y_one_hot = xp.zeros_like(probs)
            y_one_hot[xp.arange(N), targets] = 1.0
        else:
            y_one_hot = targets

        grad_logits = (probs - y_one_hot) / N  # (N, C)
        return grad_logits
