# layers.py
# ============================================
# 各种网络层的抽象与基础实现骨架（GPU 版）
# 使用 backend.xp（cupy）进行所有数值计算
# 支持：
#   - Linear / ReLU / Flatten
#   - Conv2D / MaxPool2D / GlobalAvgPool2D
#   - BatchNorm2D
#   - Dropout
# ============================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np  # 类型标注 / 标量运算
from backend import xp  # 实际计算：cupy


def _pair(v) -> Tuple[int, int]:
    """把 int 或 (int,int) 规范成 (h, w) 形式。"""
    if isinstance(v, int):
        return (v, v)
    return v


# -------------------- 抽象基类 --------------------


class Layer(ABC):
    """
    所有层的抽象基类。
    约定：
      - forward(x): 计算前向输出，并缓存反向传播需要的中间变量
      - backward(grad_out): 计算对输入的梯度，并累计本层参数的梯度
      - params(): 返回本层所有“可训练参数”的列表（xp 数组）
      - grads(): 返回与 params 一一对应的梯度列表
    """

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params(self) -> List[np.ndarray]:
        return []

    def grads(self) -> List[np.ndarray]:
        return []


# -------------------- 基础 MLP 层 --------------------


class Linear(Layer):
    """
    全连接层： y = x W^T + b

    约定：
      - 输入 x: 形状 (N, D_in)
      - 权重 W: 形状 (D_out, D_in)
      - 偏置 b: 形状 (D_out,)
      - 输出 y: 形状 (N, D_out)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

        scale = 1.0 / (in_features ** 0.5)
        self.W = (xp.random.randn(out_features, in_features) * scale).astype(xp.float32)
        self.b = xp.zeros(out_features, dtype=xp.float32)

        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)

        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = x @ self.W.T + self.b
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("Must call forward before backward.")
        x = self._x
        grad_y = grad_out

        self.dW += grad_y.T @ x
        self.db += grad_y.sum(axis=0)
        grad_x = grad_y @ self.W
        return grad_x

    def params(self) -> List[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> List[np.ndarray]:
        return [self.dW, self.db]


class ReLU(Layer):
    """ReLU 激活：y = max(0, x)"""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0)
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("Must call forward before backward.")
        return grad_out * self._mask


class Flatten(Layer):
    """
    展平层：
      - 输入: (N, C, H, W)
      - 输出: (N, C*H*W)
    """

    def __init__(self) -> None:
        self._input_shape: Tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("Must call forward before backward.")
        return grad_out.reshape(self._input_shape)


# -------------------- Conv2D / Pooling --------------------


class Conv2D(Layer):
    """
    2D 卷积层（简化版，支持 stride / padding / bias）

    约定：
      - 输入 x: (N, C_in, H, W)
      - 权重 W: (C_out, C_in, K_h, K_w)
      - 偏置 b: (C_out,)
      - 输出 y: (N, C_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.bias_enabled = bias

        k_h, k_w = self.kernel_size
        scale = 1.0 / (in_channels * k_h * k_w) ** 0.5

        self.W = (xp.random.randn(out_channels, in_channels, k_h, k_w) * scale).astype(
            xp.float32
        )
        self.dW = xp.zeros_like(self.W)

        if bias:
            self.b = xp.zeros(out_channels, dtype=xp.float32)
            self.db = xp.zeros_like(self.b)
        else:
            self.b = None
            self.db = None

        self._x: Optional[np.ndarray] = None

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        pad_h, pad_w = self.padding
        if pad_h == 0 and pad_w == 0:
            return x
        return xp.pad(
            x,
            pad_width=((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0.0,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (N, C_in, H, W)
        """
        self._x = x

        N, C_in, H, W = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride

        x_padded = self._pad_input(x)
        _, _, H_p, W_p = x_padded.shape

        H_out = (H_p - k_h) // s_h + 1
        W_out = (W_p - k_w) // s_w + 1

        out = xp.zeros((N, self.out_channels, H_out, W_out), dtype=x.dtype)

        # 遍历输出空间位置，每个位置用 tensordot 做一次卷积
        for oh in range(H_out):
            h_start = oh * s_h
            h_end = h_start + k_h
            for ow in range(W_out):
                w_start = ow * s_w
                w_end = w_start + k_w

                patch = x_padded[:, :, h_start:h_end, w_start:w_end]  # (N, C_in, k_h, k_w)
                # (N,C_in,k_h,k_w) · (C_out,C_in,k_h,k_w) -> (N,C_out)
                out[:, :, oh, ow] = xp.tensordot(
                    patch, self.W, axes=([1, 2, 3], [1, 2, 3])
                )

        if self.b is not None:
            out += self.b.reshape(1, -1, 1, 1)

        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out: (N, C_out, H_out, W_out)
        返回：grad_x: (N, C_in, H, W)
        """
        if self._x is None:
            raise RuntimeError("Must call forward before backward.")
        x = self._x

        N, C_in, H, W = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        pad_h, pad_w = self.padding

        x_padded = self._pad_input(x)
        _, _, H_p, W_p = x_padded.shape

        _, C_out, H_out, W_out = grad_out.shape

        grad_x_padded = xp.zeros_like(x_padded)

        # bias 梯度
        if self.db is not None:
            self.db += grad_out.sum(axis=(0, 2, 3))

        # W 和 x 的梯度
        for oh in range(H_out):
            h_start = oh * s_h
            h_end = h_start + k_h
            for ow in range(W_out):
                w_start = ow * s_w
                w_end = w_start + k_w

                patch = x_padded[:, :, h_start:h_end, w_start:w_end]  # (N, C_in, k_h, k_w)
                grad_slice = grad_out[:, :, oh, ow]                    # (N, C_out)

                # dW 累加： (C_out, C_in, k_h, k_w)
                self.dW += xp.tensordot(grad_slice, patch, axes=([0], [0]))

                # 对 x_padded 的梯度： (N, C_in, k_h, k_w)
                grad_patch = xp.tensordot(grad_slice, self.W, axes=([1], [0]))
                grad_x_padded[:, :, h_start:h_end, w_start:w_end] += grad_patch

        # 去掉 padding，得到对原始 x 的梯度
        if pad_h > 0 or pad_w > 0:
            grad_x = grad_x_padded[
                :, :, pad_h : pad_h + H, pad_w : pad_w + W
            ]
        else:
            grad_x = grad_x_padded

        return grad_x

    def params(self) -> List[np.ndarray]:
        params: List[np.ndarray] = [self.W]
        if self.b is not None:
            params.append(self.b)
        return params

    def grads(self) -> List[np.ndarray]:
        grads: List[np.ndarray] = [self.dW]
        if self.db is not None:
            grads.append(self.db)
        return grads


class MaxPool2D(Layer):
    """
    2D 最大池化层（无参数）

    约定：
      - 输入 x: (N, C, H, W)
      - 输出 y: (N, C, H_out, W_out)
    """

    def __init__(
        self,
        kernel_size: int | Tuple[int, int] = 2,
        stride: int | Tuple[int, int] = 2,
    ) -> None:
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

        self._input_shape: Optional[Tuple[int, int, int, int]] = None
        # 记录每个输出位置对应的窗口内 argmax 索引（0 ~ k_h*k_w-1）
        self._indices: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (N, C, H, W)
        """
        self._input_shape = x.shape
        N, C, H, W = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride

        H_out = (H - k_h) // s_h + 1
        W_out = (W - k_w) // s_w + 1

        out = xp.zeros((N, C, H_out, W_out), dtype=x.dtype)
        indices = xp.zeros((N, C, H_out, W_out), dtype=xp.int32)

        for oh in range(H_out):
            h_start = oh * s_h
            h_end = h_start + k_h
            for ow in range(W_out):
                w_start = ow * s_w
                w_end = w_start + k_w

                window = x[:, :, h_start:h_end, w_start:w_end]  # (N, C, k_h, k_w)
                flat = window.reshape(N, C, -1)                 # (N, C, k_h*k_w)

                idx = xp.argmax(flat, axis=2)                   # (N, C)
                indices[:, :, oh, ow] = idx

                # 取出最大值
                max_vals = xp.take_along_axis(flat, idx[..., None], axis=2)[..., 0]
                out[:, :, oh, ow] = max_vals

        self._indices = indices
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out: (N, C, H_out, W_out)
        返回：grad_x: (N, C, H, W)
        """
        if self._input_shape is None or self._indices is None:
            raise RuntimeError("Must call forward before backward.")

        N, C, H, W = self._input_shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride

        _, _, H_out, W_out = grad_out.shape
        grad_x = xp.zeros((N, C, H, W), dtype=grad_out.dtype)

        for oh in range(H_out):
            h_base = oh * s_h
            for ow in range(W_out):
                w_base = ow * s_w

                grad_slice = grad_out[:, :, oh, ow]  # (N, C)
                idx = self._indices[:, :, oh, ow]    # (N, C)

                kh = idx // k_w
                kw = idx % k_w

                h_idx = h_base + kh  # (N, C)
                w_idx = w_base + kw  # (N, C)

                # 展平所有索引
                n_flat = xp.repeat(xp.arange(N), C)
                c_flat = xp.tile(xp.arange(C), N)
                h_flat = h_idx.reshape(-1)
                w_flat = w_idx.reshape(-1)
                g_flat = grad_slice.reshape(-1)

                grad_x[n_flat, c_flat, h_flat, w_flat] += g_flat

        return grad_x


class GlobalAvgPool2D(Layer):
    """
    全局平均池化：
      - 输入: (N, C, H, W)
      - 输出: (N, C)
    """

    def __init__(self) -> None:
        self._input_shape: Optional[Tuple[int, int, int, int]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input_shape = x.shape
        # 对空间维度 (H, W) 求平均
        return x.mean(axis=(2, 3))

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("Must call forward before backward.")
        N, C, H, W = self._input_shape

        # 每个位置平分梯度：1 / (H*W)
        grad = grad_out.reshape(N, C, 1, 1) / (H * W)
        return xp.ones((N, C, H, W), dtype=grad_out.dtype) * grad


# -------------------- BatchNorm2D --------------------


class BatchNorm2D(Layer):
    """
    2D 批归一化：针对 conv 输出 (N, C, H, W)。

    只实现训练用的反向传播；
    推理时若 track_running_stats=True，可以使用 running_mean/var。
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.9,
        eps: float = 1e-5,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.gamma = xp.ones(num_features, dtype=xp.float32)
            self.beta = xp.zeros(num_features, dtype=xp.float32)
            self.dgamma = xp.zeros_like(self.gamma)
            self.dbeta = xp.zeros_like(self.beta)
        else:
            self.gamma = None
            self.beta = None
            self.dgamma = None
            self.dbeta = None

        if track_running_stats:
            self.running_mean = xp.zeros(num_features, dtype=xp.float32)
            self.running_var = xp.ones(num_features, dtype=xp.float32)
        else:
            self.running_mean = None
            self.running_var = None

        self.training = True  # 外部可手动切换

        # 缓存
        self._x: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._var: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (N, C, H, W)
        """
        self._x = x
        N, C, H, W = x.shape

        # 按通道统计 mean/var：对 N,H,W 求平均
        if self.training or not self.track_running_stats or self.running_mean is None:
            mean = x.mean(axis=(0, 2, 3))  # (C,)
            var = x.var(axis=(0, 2, 3))    # (C,)

            if self.track_running_stats and self.running_mean is not None:
                self.running_mean = (
                    self.momentum * self.running_mean + (1.0 - self.momentum) * mean
                )
                self.running_var = (
                    self.momentum * self.running_var + (1.0 - self.momentum) * var
                )
        else:
            # 推理时使用 running stats
            mean = self.running_mean
            var = self.running_var

        self._mean = mean
        self._var = var

        mean_b = mean.reshape(1, C, 1, 1)
        std_b = xp.sqrt(var + self.eps).reshape(1, C, 1, 1)

        x_hat = (x - mean_b) / std_b  # 标准化

        if self.affine and self.gamma is not None and self.beta is not None:
            y = self.gamma.reshape(1, C, 1, 1) * x_hat + self.beta.reshape(1, C, 1, 1)
        else:
            y = x_hat

        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out: (N, C, H, W)
        返回：grad_x: (N, C, H, W)
        """
        if self._x is None or self._mean is None or self._var is None:
            raise RuntimeError("Must call forward before backward.")

        if not self.training:
            # 通常不会在 eval 模式下反向，这里简单报错提醒
            raise RuntimeError("BatchNorm2D backward only valid in training mode.")

        x = self._x
        mean = self._mean
        var = self._var

        N, C, H, W = x.shape
        N_prime = N * H * W

        mean_b = mean.reshape(1, C, 1, 1)
        var_eps = var + self.eps  # (C,)
        std = xp.sqrt(var_eps).reshape(1, C, 1, 1)

        x_mu = x - mean_b
        x_hat = x_mu / std

        # gamma / beta 梯度
        if self.affine and self.gamma is not None and self.beta is not None:
            self.dbeta += grad_out.sum(axis=(0, 2, 3))
            self.dgamma += (grad_out * x_hat).sum(axis=(0, 2, 3))
            gamma = self.gamma
        else:
            gamma = xp.ones(C, dtype=x.dtype)

        # 对 x 的梯度
        dx_hat = grad_out * gamma.reshape(1, C, 1, 1)

        inv_std = 1.0 / xp.sqrt(var_eps)           # (C,)
        inv_std3 = inv_std ** 3                    # (C,)

        dvar = xp.sum(
            dx_hat * x_mu * (-0.5) * inv_std3.reshape(1, C, 1, 1),
            axis=(0, 2, 3),
        )  # (C,)

        dmean = xp.sum(
            dx_hat * (-inv_std.reshape(1, C, 1, 1)),
            axis=(0, 2, 3),
        ) + dvar * xp.sum(-2.0 * x_mu, axis=(0, 2, 3)) / N_prime  # (C,)

        dx = (
            dx_hat * inv_std.reshape(1, C, 1, 1)
            + dvar.reshape(1, C, 1, 1) * 2.0 * x_mu / N_prime
            + dmean.reshape(1, C, 1, 1) / N_prime
        )

        return dx

    def params(self) -> List[np.ndarray]:
        if self.affine and self.gamma is not None and self.beta is not None:
            return [self.gamma, self.beta]
        return []

    def grads(self) -> List[np.ndarray]:
        if self.affine and self.dgamma is not None and self.dbeta is not None:
            return [self.dgamma, self.dbeta]
        return []


# -------------------- Dropout --------------------


class Dropout(Layer):
    """
    Dropout 层（inverted dropout）：

      - 训练时：随机置零部分神经元，并放大剩余神经元 1/(1-p)
      - 推理时：直接输出，不做任何处理
    """

    def __init__(self, p: float = 0.5) -> None:
        assert 0.0 <= p < 1.0, "Dropout p must be in [0,1)."
        self.p = p
        self.training = True
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self._mask = None
            return x

        keep_prob = 1.0 - self.p
        mask = (xp.random.rand(*x.shape) < keep_prob).astype(x.dtype)
        self._mask = mask
        return x * mask / keep_prob

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0 or self._mask is None:
            return grad_out

        keep_prob = 1.0 - self.p
        return grad_out * self._mask / keep_prob
