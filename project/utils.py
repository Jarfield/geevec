# utils.py
# ============================================
# 通用工具函数：
#   - set_seed: 统一设置随机种子（numpy + cupy）
#   - accuracy_from_logits: 计算分类准确率（先转回 CPU）
#   - evaluate: 在某个 DataLoader 上跑一遍评估
#   - save_model / load_model: 保存/加载模型参数（npz，CPU 上）
#   - confusion_matrix: 混淆矩阵（先转回 CPU）
# ============================================

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from backend import xp, to_cpu
from models import Model
from loss import Loss


def set_seed(seed: int) -> None:
    """
    设置 numpy 和 cupy 的随机种子。
    """
    np.random.seed(seed)
    xp.random.seed(seed)


def accuracy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    根据 logits 计算 Top-1 准确率。

    logits: (N, C)，实际为 cupy.ndarray（GPU）
    targets: (N,)   实际也为 cupy.ndarray
    """
    logits_cpu = to_cpu(logits)
    targets_cpu = to_cpu(targets)

    preds = np.argmax(logits_cpu, axis=1)
    correct = np.sum(preds == targets_cpu)
    return float(correct) / float(targets_cpu.shape[0])


def evaluate(
    model: Model,
    loss_fn: Loss,
    data_loader,
    split_name: str = "Val",
) -> Tuple[float, float]:
    """
    在给定的数据集（data_loader）上做一次完整评估。

    返回:
        avg_loss: 平均损失
        avg_acc:  平均准确率
    """
    running_loss = 0.0
    running_correct = 0.0
    total_samples = 0

    for xb, yb in data_loader:
        logits = model.forward(xb)             # GPU
        loss = loss_fn.forward(logits, yb)     # GPU 内部计算

        batch_size = xb.shape[0]
        running_loss += loss * batch_size
        running_correct += accuracy_from_logits(logits, yb) * batch_size
        total_samples += batch_size

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    # print(f"{split_name}: Loss: {avg_loss:.4f}  Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc


def save_model(model: Model, path: str, makedirs: bool = True) -> None:
    """
    把模型的所有参数保存到一个 .npz 文件中。

    注意：
      - 只是按 model.params() 的顺序保存为 p0, p1, ...
      - 加载时需要用同一个模型结构调用 load_model
    """
    params = model.params()
    if makedirs:
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

    # 保存时先把 GPU 上的参数拉回 CPU
    save_dict = {f"p{i}": to_cpu(p) for i, p in enumerate(params)}
    np.savez_compressed(path, **save_dict)
    # print(f"Model parameters saved to: {path}")


def load_model(model: Model, path: str) -> None:
    """
    从 .npz 文件中加载参数到模型。

    要求：
      - 模型结构与保存时一致
      - 参数数量和形状能一一对应
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = np.load(path)
    params = model.params()

    for i, p in enumerate(params):
        key = f"p{i}"
        if key not in data:
            raise KeyError(f"Parameter {key} not found in checkpoint {path}")
        loaded = data[key]
        if loaded.shape != p.shape:
            raise ValueError(
                f"Shape mismatch for param {key}: checkpoint {loaded.shape}, model {p.shape}"
            )
        # p 是 cupy.ndarray，loaded 是 numpy.ndarray，p[...] = loaded 会自动做 H2D 拷贝
        p[...] = loaded

    print(f"Model parameters loaded from: {path}")


def confusion_matrix(
    logits: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    计算混淆矩阵（confusion matrix）。

    logits: (N, C)，实际为 cupy.ndarray
    targets: (N,)
    返回:
        cm: (num_classes, num_classes)
            行 = 真实标签, 列 = 预测标签
    """
    logits_cpu = to_cpu(logits)
    targets_cpu = to_cpu(targets)

    preds = np.argmax(logits_cpu, axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(targets_cpu, preds):
        cm[int(t), int(p)] += 1

    return cm
