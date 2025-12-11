# SCIDOCS 专用增强流程

本目录替换了旧版 `code_for_SCIDOCS`，提供两个阶段脚本：

- **Stage A (`build_scirepeval_scidocs_like.py`)**：从 `allenai/scirepeval` 抽取 SCIDOCS-like citation triples，自动过滤 SciDOCS 评测分布并按 FOS/年份筛选。
- **Stage B (`add_scincl_neighbors.py`)**：在 Stage A 产出的 JSONL 上追加 SciNCL 风格的邻域采样，自动补充额外正例与 hard negative。

所有产物与 `data_augmentation` 的 JSONL 格式兼容，可直接喂给 `run_pair_scoring.py` 继续做 LLM 打分，或直接用于 embedding 训练。

## 目录结构
- `build_scirepeval_scidocs_like.py`：Stage A 主脚本。
- `add_scincl_neighbors.py`：Stage B 主脚本，带有 `encode_batch` 占位符，请替换为你的 embedding 模型。
- `run_stage_a.sh`：封装 Stage A，方便用环境变量控制输出路径、采样数量等。
- `run_stage_b.sh`：封装 Stage B，指定输入/输出、邻域规模与索引参数。

## 快速开始
### 1. 构造 SCIDOCS-like 训练集（Stage A）
```bash
cd data_generation/code_for_SCIDOCS
# 环境变量可选：SAVE_ROOT / OUTPUT_PATH / NUM_SAMPLES / MIN_YEAR / FOS_FILTER / MIN_TEXT_LEN / OVERWRITE
./run_stage_a.sh
# 默认输出：${SAVE_ROOT:-./../../original_data}/scidocs/scirep_citation_train/en_scirep.jsonl
```
核心参数：
- `--fos`：默认 `Computer Science`，大小写不敏感，可置空跳过。
- `--min_year`：默认 2015；将过滤掉更早的论文。
- 自动跳过所有 `allenai/scirepeval` 中 SciDOCS 相关的 `corpus_id`，避免泄漏到 MTEB-SCIDOCS。

### 2. 追加 SciNCL 邻域采样（Stage B，可选）
> 先在 `add_scincl_neighbors.py` 中实现 `encode_batch`，例如接入 SentenceTransformers 或自有模型，返回 `np.ndarray`。
```bash
cd data_generation/code_for_SCIDOCS
# 环境变量可选：INPUT_PATH / OUTPUT_PATH / NUM_POS_NEIGHBORS / NUM_HARD_NEGATIVES / SEARCH_DEPTH / INDEX_FACTORY / BATCH_SIZE / OVERWRITE
./run_stage_b.sh
# 默认输入：沿用 Stage A（`SAVE_ROOT`）默认输出；默认输出：同目录下 `_scincl.jsonl` 后缀
```
- 产出仍是 `{query, pos: [str], neg: [str]}` 结构，可直接进入 `data_augmentation/script/run_pair_scoring.sh` 做 LLM 打分。
- `num_pos_neighbors` / `num_hard_negatives` 控制一次追加多少邻居；`index_factory` 允许切换其他 FAISS 索引。

### 3. 与 data_augmentation 联动
Stage A/B 输出格式与 `data_augmentation/code/run_pair_scoring.py` 兼容，可直接执行：
```bash
cd data_generation/data_augmentation
TASK_TYPE="scidocs" LANGUAGE="en" \
INPUT_PATH="/path/to/en_scirep.jsonl" OUTPUT_PATH="/path/to/en_scirep_scored.jsonl" \
MODEL_NAME="Qwen2-5-72B-Instruct" PORT=8000 NUM_PROCESSES=8 \
./script/run_pair_scoring.sh
```
随后可复用你的 embedding 训练脚本或继续下一步数据处理。

## 注意事项
- 需要 `datasets`, `faiss`, `numpy`, `tqdm` 等依赖；Stage B 还需你实现 `encode_batch`。
- Stage A 默认保存到 `DATA_AUG_ORIGINAL_ROOT` 环境变量指向的目录（未设置则使用当前目录下的 `original_data`）。
- 若想跨样本共享文档，可在 Stage A 输出中改用文本 hash 作为 `doc_id`（`add_scincl_neighbors.py` 已在缺失 doc_pool 时自动使用此策略）。
