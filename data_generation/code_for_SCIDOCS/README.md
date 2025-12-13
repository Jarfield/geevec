# SCIDOCS 专用增强流程
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

## SCIDOCS 域自适应增强思路（升级版）
为了缓解「训练语料与 SCIDOCS 评测分布差异过大、上线后分数掉到 24」的问题，Stage A 新增了 **vLLM 驱动的主题锚点** 与 **只过滤 query 的策略**，旨在让训练覆盖测试集主题、又不误伤正/负样本的多样性。

### 核心想法
- **用 vLLM 总结测试主题**：通过 `--topic_summary_model` + `--topic_summary_endpoint` 直接调用 vLLM/OpenAI 接口，对 SCIDOCS 评测集（titles + abstracts，默认最多 200 条可调）进行关键词总结，生成的主题词集合就是锚点词表。这样锚点更贴合真实评测分布，而不是简单高频词。若未指定模型，则回落到原来的高频词方案。
- **锚点只约束 query，不过滤 doc**：我们只要求 query title 与锚点有足够重叠，用于挑选「确实是目标主题的查询」。Pos/Neg 文档保持原始分布，不再因为锚点缺失被丢弃，从而缓解之前“doc 被过滤导致分布差异化”的问题。
- **继续年份/FOS 过滤**：如果 doc 有元数据，则仍按 `--min_year` / `--fos` 过滤，避免过早或偏题论文。

### 使用方式（含全量测试集跑法）
- 仍在 `run_stage_a.sh` 的基础上运行，新增可调参数：
  - `--topic_summary_model`：启用 vLLM/开放接口的模型名（例如 `qwen2.5-32b-instruct`）。为空则使用高频词锚点。
  - `--topic_summary_endpoint`：vLLM/OpenAI 兼容的 base_url（可设 `VLLM_ENDPOINT=http://localhost:8000/v1`）。
  - `--topic_summary_max_docs`：摘要入 prompt 的评测文档数量上限，默认 200；当置为 `0` 或 `-1` 时会自动 **使用全部 SCIDOCS 测试集**（脚本已内置分块调用，不会因 token 过长报错）。
  - `--topic_keywords_per_chunk`：每个 prompt chunk 期望返回的关键词数量，默认 48。
  - `--anchor_vocab_size` / `--anchor_min_token_len` / `--min_anchor_overlap`：保留，作为高频词后备或调节 query 过滤强度；`--min_anchor_overlap>0` 时只作用于 query。
- 推荐流程：
  1. **先启动 vLLM 服务**（Terminal A）：例如
     ```bash
     CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
         --model /data/share/project/shared_models/Qwen/Qwen2.5-72B-Instruct \
         --served-model-name Qwen2-5-72B-Instruct \
         --tensor-parallel-size 4 \
         --port 8000 \
         --trust-remote-code
     ```
     等待 `Application startup complete.` 后再继续。
  2. **再运行 Stage A（Terminal B）**：示例 1——跑全量测试集做主题锚点
     ```bash
     cd data_generation/code_for_SCIDOCS
     export VLLM_ENDPOINT="http://localhost:8000/v1"
     ./run_stage_a.sh --topic_summary_model "Qwen2-5-72B-Instruct" \
                      --topic_summary_max_docs -1 \
                      --min_anchor_overlap 3
     ```
     示例 2——继续采样（非全量）时可改回默认 200：
     ```bash
     ./run_stage_a.sh --topic_summary_model "Qwen2-5-72B-Instruct" \
                      --topic_summary_max_docs 200 \
                      --min_anchor_overlap 3
     ```

### 为什么能缓解分布差异
- **锚点紧贴评测主题**：直接用 vLLM 对 SCIDOCS 测试集做主题总结，锚点词表由真实评测主题驱动，减少领域错配。
- **不再丢弃文档**：pos/neg 不受锚点过滤，只要 query 通过锚点检查就保留整条样本，避免训练集分布被过度收窄。
- **可复用高频词回落**：若 vLLM 不可用，自动回落到高频词构造锚点，仍能提供基础的领域约束。
