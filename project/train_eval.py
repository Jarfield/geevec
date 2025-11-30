# train.py
# ============================================
# 训练入口脚本（GPU 版）
#   - 支持 MLP / ResNetSmall，两种模型二选一
#   - 数据通过 DataLoader 自动搬到 GPU
# ============================================

from __future__ import annotations

import time
from typing import Tuple

from data import create_fashion_mnist_loaders
from models import MLPClassifier, ResNetSmall, Model
from loss import SoftmaxCrossEntropyLoss
from optim import SGD
from config import (
    DATA_ROOT,
    BATCH_SIZE,
    VAL_RATIO,
    NUM_EPOCHS,
    INPUT_DIM,
    HIDDEN_DIMS,
    NUM_CLASSES,
    LEARNING_RATE,
    SEED,
    CHECKPOINT_PATH,
)
from utils import (
    set_seed,
    accuracy_from_logits,
    evaluate,   # 评估函数已经在 utils 里实现
    save_model, # 用于保存最优模型
)

# ===== 这里切换用哪个模型 =====
USE_RESNET = True   # True: ResNetSmall；False: MLPClassifier
# ============================


def train_one_epoch(
    model: Model,                      # 用通用 Model，而不是只写 MLPClassifier
    loss_fn: SoftmaxCrossEntropyLoss,
    optimizer: SGD,
    train_loader,
    epoch: int,
) -> Tuple[float, float]:
    running_loss = 0.0
    running_correct = 0.0
    total_samples = 0

    start_time = time.time()

    for batch_idx, (xb, yb) in enumerate(train_loader):
        # 1. 清零梯度
        optimizer.zero_grad()

        # 2. 前向（在 GPU 上）
        logits = model.forward(xb)
        loss = loss_fn.forward(logits, yb)

        # 3. 反向
        grad_logits = loss_fn.backward()
        model.backward(grad_logits)

        # 4. 更新参数（在 GPU 上）
        optimizer.step()

        # 统计（准确率在 CPU 上计算）
        batch_size = xb.shape[0]
        batch_acc = accuracy_from_logits(logits, yb)

        running_loss += loss * batch_size
        running_correct += batch_acc * batch_size
        total_samples += batch_size

        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / total_samples
            avg_acc = running_correct / total_samples
            # print(
            #    f"Train Epoch {epoch} [{batch_idx+1}/{len(train_loader)}]  "
            #    f"Loss: {avg_loss:.4f}  Acc: {avg_acc:.4f}"
            # )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    end_time = time.time()
    # print(
    #     f"Train Epoch {epoch} DONE  "
    #     f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  "
    #     f"Time: {end_time - start_time:.1f}s"
    # )
    return epoch_loss, epoch_acc


def main() -> None:
    # 1. 随机种子（同时设置 numpy / cupy）
    set_seed(SEED)

    # 2. 数据
    #    - 若用 MLP：需要 flatten=True，得到 (N, 784)
    #    - 若用 ResNetSmall：保持图片形状，flatten=False，(N, 1, 28, 28)
    flatten_flag = not USE_RESNET

    train_loader, val_loader, test_loader = create_fashion_mnist_loaders(
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        normalize=True,
        flatten=flatten_flag,
    )

    # 3. 模型 & 损失 & 优化器（参数在 GPU 上）
    if USE_RESNET:
        model: Model = ResNetSmall(num_classes=NUM_CLASSES)
        print(">> Using model: ResNetSmall")
    else:
        model = MLPClassifier(
            input_dim=INPUT_DIM,
            hidden_dims=HIDDEN_DIMS,
            num_classes=NUM_CLASSES,
        )
        print(">> Using model: MLPClassifier")

    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = SGD(model, lr=LEARNING_RATE)

    # 4. 训练循环
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, loss_fn, optimizer, train_loader, epoch
        )
        val_loss, val_acc = evaluate(model, loss_fn, val_loader, split_name="Val")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # print(f"==> New best val acc: {best_val_acc:.4f}")
            # 保存当前最优模型（会自动把参数拉回 CPU 再写 npz）
            save_model(model, CHECKPOINT_PATH)

    print("Training finished.")
    print("Evaluating on TEST set...")

    test_loss, test_acc = evaluate(model, loss_fn, test_loader, split_name="Test")
    print(f"Final Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
