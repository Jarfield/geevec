# config.py
from __future__ import annotations
import os

# 小工具函数，用环境变量覆盖默认值
def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))

def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))

def _get_str(name: str, default: str) -> str:
    return os.getenv(name, default)


# 数据路径
DATA_ROOT = _get_str("DATA_ROOT", "/data/share/project/psjin/code/project/dataset")

# 训练相关
BATCH_SIZE = _get_int("BATCH_SIZE", 1024)
VAL_RATIO = float(os.getenv("VAL_RATIO", 0.1))   # 一般不扫，固定就行
NUM_EPOCHS = _get_int("NUM_EPOCHS", 1000)

# 模型结构（MLP 相关，ResNet 用不到也没关系）
INPUT_DIM = 28 * 28          # 784
HIDDEN_DIMS = [256, 128]
NUM_CLASSES = 10

# 优化相关
LEARNING_RATE = _get_float("LEARNING_RATE", 3e-3)

# 随机种子
SEED = _get_int("SEED", 42)

# ===== 模型 & checkpoint 配置 =====
MODEL_NAME = _get_str("MODEL_NAME", "resnet_small")
CHECKPOINT_DIR = _get_str("CHECKPOINT_DIR", "checkpoints")

# 把 batch_size / lr / epoch 都写进文件名，方便区分不同实验
CHECKPOINT_NAME = f"{MODEL_NAME}_bs{BATCH_SIZE}_lr{LEARNING_RATE:g}_ep{NUM_EPOCHS}.npz"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_NAME}"
