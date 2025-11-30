#!/usr/bin/env bash
set -e  # 一旦有命令出错就退出，避免夜里一直跑错误

# 可选：如果你有多个 Python env，在这里激活
# conda activate /data/share/project/public_envs/embedder_train_eval

cd /data/share/project/psjin/code/project

# 固定的一些设置
export DATA_ROOT="/data/share/project/psjin/code/project/dataset"
export MODEL_NAME="resnet_small"
export CHECKPOINT_DIR="checkpoints"

# 日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 超参网格（你可以按需改）
BATCH_SIZES=(512)
LRS=(0.01 0.03 0.1)
EPOCHS=(50 100 200)   # 建议扫参先用短一点的 epoch，找到好配置再单独拉长到 1000

for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LRS[@]}"; do
    for ep in "${EPOCHS[@]}"; do
      echo "============================================="
      echo "Run: bs=${bs}, lr=${lr}, epochs=${ep}"
      echo "============================================="

      export BATCH_SIZE="${bs}"
      export LEARNING_RATE="${lr}"
      export NUM_EPOCHS="${ep}"

      # 日志文件名
      LOG_FILE="${LOG_DIR}/train_bs${bs}_lr${lr}_ep${ep}.log"

      # 顺序执行每个组合，并把 stdout/stderr 记录到日志
      python train_eval.py 2>&1 | tee "${LOG_FILE}"
    done
  done
done
