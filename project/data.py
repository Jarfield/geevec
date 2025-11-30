# data.py
# ============================================
# Fashion-MNIST 数据加载 & 简单 DataLoader 实现（GPU 版）
# - 磁盘读取 & 全量存储仍然用 numpy（CPU 内存）
# - 每次从 DataLoader 取 batch 时转成 backend.xp（GPU）数组
# ============================================

from __future__ import annotations

import os
import gzip
from typing import Tuple, Optional, Iterator

import numpy as np
from backend import xp  # xp 是 cupy（GPU）


# ---------- 低层：idx 文件读取 ----------

def _read_idx_images(path: str) -> np.ndarray:
    """
    读取 idx 格式的图像文件（如 train-images-idx3-ubyte.gz）
    返回形状 (N, 1, H, W) 的 uint8 数组 (numpy)。
    """
    with gzip.open(path, "rb") as f:
        # idx 格式头部：magic(4) + num(4) + rows(4) + cols(4)，全部大端
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file {path}")

        num_images = int.from_bytes(f.read(4), "big")
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")

        buf = f.read(num_images * num_rows * num_cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, 1, num_rows, num_cols)  # N, C=1, H, W
    return data


def _read_idx_labels(path: str) -> np.ndarray:
    """
    读取 idx 格式的标签文件（如 train-labels-idx1-ubyte.gz）
    返回形状 (N,) 的 uint8 数组 (numpy)。
    """
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file {path}")

        num_items = int.from_bytes(f.read(4), "big")
        buf = f.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


# ---------- 中层：封装成 Fashion-MNIST 加载函数 ----------

def load_fashion_mnist(
    root: str,
    kind: str = "train",
    normalize: bool = True,
    flatten: bool = False,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 root 目录加载 Fashion-MNIST 数据（numpy 数组）。

    参数
    ----
    root: 数据所在目录，里面包含官方的 .gz 文件
          - train-images-idx3-ubyte.gz
          - train-labels-idx1-ubyte.gz
          - t10k-images-idx3-ubyte.gz
          - t10k-labels-idx1-ubyte.gz
    kind: "train" 或 "test"（"test" 等价于官方的 t10k）
    normalize: 是否除以 255.0 映射到 [0, 1]
    flatten: 是否展平为 (N, 784)，否则为 (N, 1, 28, 28)
    dtype: 浮点类型，默认 np.float32

    返回
    ----
    images: np.ndarray, 形状:
        - flatten=False: (N, 1, 28, 28)
        - flatten=True : (N, 784)
    labels: np.ndarray, 形状 (N,)，dtype=int64
    """
    if kind == "train":
        images_path = os.path.join(root, "train-images-idx3-ubyte.gz")
        labels_path = os.path.join(root, "train-labels-idx1-ubyte.gz")
    elif kind in ("test", "t10k"):
        # 官方原始文件名是 t10k-...，不要写成 test-...
        images_path = os.path.join(root, "test-images-idx3-ubyte.gz")
        labels_path = os.path.join(root, "test-labels-idx1-ubyte.gz")
    else:
        raise ValueError(f"Unknown kind: {kind}. Expected 'train' or 'test'.")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Image file not found: {images_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    images = _read_idx_images(images_path)
    labels = _read_idx_labels(labels_path).astype(np.int64)

    if normalize:
        images = images.astype(dtype) / dtype(255.0)
    else:
        images = images.astype(dtype)

    if flatten:
        # (N, 1, 28, 28) -> (N, 784)
        images = images.reshape(images.shape[0], -1)

    return images, labels


# ---------- 工具：train/val 划分 ----------

def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将训练集划分为 train / val 两部分（numpy 操作）。

    返回
    ----
    X_train, y_train, X_val, y_val
    """
    assert X.shape[0] == y.shape[0], "X 和 y 的样本数不一致"

    n_samples = X.shape[0]
    n_val = int(n_samples * val_ratio)

    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)

    if shuffle:
        rng.shuffle(indices)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    return X_train, y_train, X_val, y_val


# ---------- 高层：DataLoader 实现 ----------

class DataLoader:
    """
    一个简单的 DataLoader：
    - 输入/存储使用 numpy 数组（CPU 内存）
    - 每次迭代返回的 batch 会转换成 GPU 上的 xp.ndarray

    用法示例：
        loader = DataLoader(X_train, y_train, batch_size=64, shuffle=True)
        for batch_x, batch_y in loader:
            # 此时 batch_x, batch_y 都是 cupy 数组（xp.ndarray）
            ...
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        assert X.shape[0] == y.shape[0], "X 和 y 的样本数不一致"

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.n_samples = X.shape[0]
        self._rng = np.random.default_rng(seed)

        # 迭代状态
        self._indices: Optional[np.ndarray] = None
        self._cursor: int = 0

    def __len__(self) -> int:
        """
        返回一个 epoch 中的 batch 数量。
        """
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        启动一个新的 epoch。
        注意：这里的类型注解仍然写 np.ndarray，
        实际返回的是 xp.asarray(...) 后的数组（cupy）。
        """
        self._indices = np.arange(self.n_samples)

        if self.shuffle:
            self._rng.shuffle(self._indices)

        self._cursor = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回下一个 batch。
        实际返回类型是 (xp.ndarray, xp.ndarray)，在 GPU 上。
        """
        if self._indices is None:
            # 如果用户直接 next(loader)，而不是先 iter(loader)
            self.__iter__()

        assert self._indices is not None  # 类型检查用

        if self._cursor >= self.n_samples:
            # 一个 epoch 结束
            raise StopIteration

        start = self._cursor
        end = start + self.batch_size
        self._cursor = end

        batch_indices = self._indices[start:end]

        # 处理最后一个 batch 不足 batch_size 的情况
        if self.drop_last and batch_indices.shape[0] < self.batch_size:
            raise StopIteration

        # 在 CPU 上索引切片，然后转成 GPU 上的 xp 数组
        batch_X = xp.asarray(self.X[batch_indices])
        batch_y = xp.asarray(self.y[batch_indices])
        return batch_X, batch_y


# ---------- 便捷函数：一键创建 train/val/test DataLoader ----------

def create_fashion_mnist_loaders(
    root: str,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    seed: Optional[int] = None,
    normalize: bool = True,
    flatten: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    一次性创建 train / val / test 三个 DataLoader。
    方便在 train.py 里直接调用。

    注意：底层调用 load_fashion_mnist 和 train_val_split。
    """
    X_train_full, y_train_full = load_fashion_mnist(
        root=root,
        kind="train",
        normalize=normalize,
        flatten=flatten,
    )
    X_test, y_test = load_fashion_mnist(
        root=root,
        kind="test",
        normalize=normalize,
        flatten=flatten,
    )

    X_train, y_train, X_val, y_val = train_val_split(
        X_train_full,
        y_train_full,
        val_ratio=val_ratio,
        shuffle=True,
        random_state=seed,
    )

    train_loader = DataLoader(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        seed=seed,
    )
    val_loader = DataLoader(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )
    test_loader = DataLoader(
        X_test, y_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        seed=seed,
    )

    return train_loader, val_loader, test_loader
