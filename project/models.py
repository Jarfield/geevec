# models.py
# ============================================
# 模型抽象基类 + 顺序模型 + MLP + ResNetSmall（GPU 版）
# ============================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np  # 用于类型标注 & 在 CPU 上做 argmax
from backend import xp, to_cpu  # xp: cupy; to_cpu: cupy -> numpy

from layers import (
    Layer,
    Linear,
    ReLU,
    Flatten,
    Conv2D,
    MaxPool2D,
    GlobalAvgPool2D,
    BatchNorm2D,
    Dropout,
)


class Model(ABC):
    """
    所有模型的抽象基类。
    约定：
      - forward(x): 前向计算得到输出（logits），x / logits 实际是 xp.ndarray（GPU 上）
      - backward(grad_out): 从损失对输出的梯度开始，反向传播
      - params(): 返回所有可训练参数（xp array 列表）
      - grads(): 返回与 params 对应的梯度（与 params 一一对应）
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def params(self) -> List[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def grads(self) -> List[np.ndarray]:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        分类任务的便捷接口：
        - 输入: x (N, D 或 N,C,H,W)，xp.ndarray（GPU）
        - 输出: 预测类别下标 (N,)，numpy.ndarray（在 CPU 上）
        """
        logits = self.forward(x)        # xp.ndarray on GPU
        logits_cpu = to_cpu(logits)     # 拉回 CPU
        return np.argmax(logits_cpu, axis=1)


class Sequential(Model):
    """
    一个简单的顺序模型：按顺序堆叠多个 Layer。
    """

    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers: List[Layer] = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> List[np.ndarray]:
        all_params: List[np.ndarray] = []
        for layer in self.layers:
            all_params.extend(layer.params())
        return all_params

    def grads(self) -> List[np.ndarray]:
        all_grads: List[np.ndarray] = []
        for layer in self.layers:
            all_grads.extend(layer.grads())
        return all_grads


# -------------------- MLP 分类器 --------------------


class MLPClassifier(Sequential):
    """
    多层感知机分类器（MLP），用于 Fashion-MNIST 等向量输入的多分类任务。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
    ) -> None:
        layers: List[Layer] = []

        in_dim = input_dim
        for h in hidden_dims:
            layers.append(Linear(in_dim, h))
            layers.append(ReLU())
            in_dim = h

        layers.append(Linear(in_dim, num_classes))
        super().__init__(layers)


# -------------------- 残差块 & ResNetSmall --------------------


class ResidualBlock(Layer):
    """
    简化版 ResNet BasicBlock：

      main branch:
        conv3x3 -> (bn) -> relu -> conv3x3 -> (bn) -> (dropout)

      shortcut:
        - 若 in_channels == out_channels 且 stride == 1: 直接 identity
        - 否则: conv1x1(+bn) 做 projection

      输出:
        y = relu( main(x) + shortcut(x) )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_projection: bool = False,
        use_bn: bool = True,
        dropout_p: float = 0.0,
    ) -> None:
        self.use_bn = use_bn
        self.use_dropout = dropout_p > 0.0

        # 主分支
        self.conv1 = Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not use_bn,
        )
        self.bn1 = BatchNorm2D(out_channels) if use_bn else None
        self.relu1 = ReLU()

        self.conv2 = Conv2D(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not use_bn,
        )
        self.bn2 = BatchNorm2D(out_channels) if use_bn else None

        self.dropout = Dropout(dropout_p) if self.use_dropout else None

        # shortcut 分支
        self.use_projection = (
            use_projection or (in_channels != out_channels or stride != 1)
        )
        if self.use_projection:
            self.proj = Conv2D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            self.proj_bn = BatchNorm2D(out_channels) if use_bn else None
        else:
            self.proj = None
            self.proj_bn = None

        self.relu_out = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        # main branch
        out = self.conv1.forward(x)
        if self.bn1 is not None:
            out = self.bn1.forward(out)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        if self.bn2 is not None:
            out = self.bn2.forward(out)
        if self.dropout is not None:
            out = self.dropout.forward(out)

        # shortcut
        shortcut = x
        if self.use_projection and self.proj is not None:
            shortcut = self.proj.forward(x)
            if self.proj_bn is not None:
                shortcut = self.proj_bn.forward(shortcut)

        out = out + shortcut
        out = self.relu_out.forward(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        对 y = relu( main(x) + shortcut(x) ) 求梯度。
        """
        # 先过最后一个 ReLU
        grad = self.relu_out.backward(grad_out)

        # 对加法: 对 main 和 shortcut 的梯度都是 grad
        grad_main = grad
        grad_short = grad

        # main branch 反向
        if self.dropout is not None:
            grad_main = self.dropout.backward(grad_main)
        if self.bn2 is not None:
            grad_main = self.bn2.backward(grad_main)
        grad_main = self.conv2.backward(grad_main)

        grad_main = self.relu1.backward(grad_main)
        if self.bn1 is not None:
            grad_main = self.bn1.backward(grad_main)
        grad_main = self.conv1.backward(grad_main)

        # shortcut branch 反向
        if self.use_projection and self.proj is not None:
            if self.proj_bn is not None:
                grad_short = self.proj_bn.backward(grad_short)
            grad_short = self.proj.backward(grad_short)
        # 否则: grad_short 直接作为对输入的梯度

        grad_x = grad_main + grad_short
        return grad_x

    def params(self) -> List[np.ndarray]:
        modules = [
            self.conv1,
            self.bn1,
            self.conv2,
            self.bn2,
            self.proj,
            self.proj_bn,
        ]
        params: List[np.ndarray] = []
        for m in modules:
            if m is None:
                continue
            params.extend(m.params())
        return params

    def grads(self) -> List[np.ndarray]:
        modules = [
            self.conv1,
            self.bn1,
            self.conv2,
            self.bn2,
            self.proj,
            self.proj_bn,
        ]
        grads: List[np.ndarray] = []
        for m in modules:
            if m is None:
                continue
            grads.extend(m.grads())
        return grads


class ResNetSmall(Sequential):
    """
    针对 Fashion-MNIST(1x28x28) 的简易 ResNet 结构：

        stem:
          Conv2D(1→32, k3,s1,p1) + BN + ReLU
          MaxPool2D(k2,s2)              # 28x28 -> 14x14

        stage1:
          ResidualBlock(32, 32, stride=1) x 2    # 14x14

        stage2:
          ResidualBlock(32, 64, stride=2, proj)  # 14x14 -> 7x7
          ResidualBlock(64, 64, stride=1)

        head:
          GlobalAvgPool2D()   # (N,64,7,7)->(N,64)
          Flatten()
          Linear(64 -> num_classes)
    """

    def __init__(self, num_classes: int = 10) -> None:
        layers: List[Layer] = []

        # stem
        layers.append(
            Conv2D(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        layers.append(BatchNorm2D(32))
        layers.append(ReLU())
        layers.append(MaxPool2D(kernel_size=2, stride=2))  # 28x28 -> 14x14

        # stage1: 32 channels, spatial 14x14
        layers.append(
            ResidualBlock(
                in_channels=32,
                out_channels=32,
                stride=1,
                use_projection=False,
                use_bn=True,
                dropout_p=0.0,
            )
        )
        layers.append(
            ResidualBlock(
                in_channels=32,
                out_channels=32,
                stride=1,
                use_projection=False,
                use_bn=True,
                dropout_p=0.0,
            )
        )

        # stage2: 32 -> 64 channels, spatial 14x14 -> 7x7
        layers.append(
            ResidualBlock(
                in_channels=32,
                out_channels=64,
                stride=2,
                use_projection=True,
                use_bn=True,
                dropout_p=0.0,
            )
        )
        layers.append(
            ResidualBlock(
                in_channels=64,
                out_channels=64,
                stride=1,
                use_projection=False,
                use_bn=True,
                dropout_p=0.0,
            )
        )

        # head
        layers.append(GlobalAvgPool2D())  # (N,64,7,7) -> (N,64)
        layers.append(Flatten())          # (N,64) -> (N,64)（只是为了通用）
        layers.append(Linear(64, num_classes))

        super().__init__(layers)