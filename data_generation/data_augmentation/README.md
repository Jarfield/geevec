# 通用数据增强工具箱

本目录提供跨任务复用的数据增强脚本，已在 MIRACL、CovidRetrieval、AILA、SCIDOCS 与 ArguAna 等场景验证。目录结构经过整理，分为：

- `code/`：核心 Python 代码与示例。
- `script/`：可直接运行的命令行脚本示例。
- `README.md`：使用说明与流程概览。

## 核心文件（位于 `code/`）
- `task_configs.py`：集中管理各任务的语料、qrels 及少 shot 样例路径。可通过环境变量 `DATA_AUG_ROOT`、`DATA_AUG_GENERATED_ROOT` 或启动参数覆盖。
- `constant.py`：任务枚举与通用提示模板（查询生成、文档改写等）。
- `attributes_config.py`：统一的叙事属性采样器，支持在通用属性上叠加任务特定风格，用于控制文档改写的语气与结构。
- `corpus_generator.py`：按任务加载与过滤语料（支持 qrels 过滤）。
- `triplet_generator.py`：将语料转换为 (query, pos) 训练三元组。
- `doc_synthesis_generator.py`：结合属性提示改写或扩展语料。
- `run_generation.py`：查询–文档三元组生成入口。
- `run_corpus_generation.py`：语料改写生成入口。

`code/gen_examples/` 中包含少量示例，便于 Few-shot 生成。

## 快速上手
1. **配置数据路径**：修改或覆盖 `task_configs.py` 中的默认路径。
2. **先生成合成语料（generated_corpus）**：
   ```bash
   cd data_generation/data_augmentation
   TASK_TYPE=scidocs LANGUAGE=en \
   NUM_VARIANTS_PER_SEED=2 NUM_THREADS=8 \
   ./script/run_corpus.sh
   ```
   - 默认输出位于 `/data/share/project/psjin/data/generated_data/<task>/generation_results/generated_corpus/<lang>_synth_corpus.jsonl`，可通过 `SAVE_ROOT` 覆盖。
   - 若需要自定义底库或 qrels，可设置 `CORPUS_PATH=/path/to/corpus` 与 `QRELS_PATH=/path/to/qrels`。

3. **基于改写语料生成三元组**：
   ```bash
   python code/run_generation.py \
     --task_type covidretrieval \
     --language zh \
     --save_dir /path/to/save \
     --corpus_path /custom/corpus.arrow \   # 可选覆盖
     --qrels_path /custom/qrels.arrow        # 可选覆盖
   ```
   - 未提供 `--examples_dir` 时，脚本会根据 `task_configs.py` 自动寻找少量示例。
   - `--num_rounds` 可用于多轮生成，CovidRetrieval 等任务会自动轮换关注点。

## 处理流程概览
- `run_corpus_generation.py` 使用 `CorpusGenerator` 读取底库和可选 qrels，`DocSynthesisGenerator` 则为每条种子文档采样叙事属性（`attributes_config.py`）后，通过 LLM 改写并产出 `generated_corpus`。输出路径默认为 `/data/share/project/psjin/data/generated_data/<task>/generation_results/generated_corpus`。
- `run_generation.py` 通过 `CorpusGenerator` 加载语料、分发到 `TripletGenerator`，最终将输出写入 `<save_dir>/<language>/<task>/...`。
- 所有脚本均通过命令行参数控制，可快速切换任务、路径和模型端口。

## 扩展到新任务
- 在 `constant.py` 中增加新的 `TaskType` 并补充提示模板。
- 在 `task_configs.py` 注册默认语料与示例路径。
- 在 `attributes_config.py` 为新任务添加特定属性选项，即可复用改写与生成流程。

## 运行脚本与任务切换

仓库下的所有示例脚本都改为自动解析项目根目录，不再依赖硬编码路径，便于直接切换任务（例如在 AILAStatutes 与 SCIDOCS 间切换）。脚本位于 `script/`，核心思路如下：

1. **统一入口与根目录**：脚本会计算 `REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)`，然后 `cd` 到该目录或 `code/` 子目录，确保相对路径生效。
2. **环境变量驱动**：常用参数（任务、语言、模型、端口、缓存目录等）都可以通过环境变量覆盖，默认值覆盖了 CovidRetrieval 的生产配置。
3. **Prompt 任务切换**：`run_generation.sh` 直接调用 `code/run_generation.py`，其中 `--task_type` 取值来自 `constant.py` 的 `TaskType`。脚本默认 `TASK_TYPE=covidretrieval`，可在命令前设置 `TASK_TYPE=scidocs` 或 `TASK_TYPE=ailastatutes`，即可复用相同的生成 prompt 去生产不同任务的数据。

### 典型用法

- **批量生成三元组**（切换到 SCIDOCS）：
  ```bash
  cd data_generation/data_augmentation
  TASK_TYPE="covidretrieval"LANGUAGES="zh" \
  NUM_EXAMPLES=10 \
  NUM_SAMPLES=10000 \
  NUM_VARIANTS_PER_DOC=1 \
  NUM_ROUNDS=5 \
  NUM_PROCESSES=8 \
  MODE="prod" \
  MODEL_NAME="Qwen2-5-72B-Instruct" \
  MODEL_TYPE="-open-source" \
  PORT=8000 \
  OVERWRITE=1 \
  ./script/run_generation.sh
  ```
  逻辑：脚本会进入 `code/` 目录，调用 `run_generation.py`，按语言循环生成并将结果写入 `${SAVE_ROOT}/${TASK_TYPE}/generation_results/prod_augmentation`。提示词会自动匹配 `TaskType.scidocs` 的生成模板。

- **生成 AILAStatutes few-shot 示例**：
  ```bash
  cd data_generation/data_augmentation
  EXAMPLES_SAVE_DIR=/tmp/generated_examples \
  AILA_TRAIN_ROOT=/path/to/ailastatutes_dataset \
  ./script/run_gen_examples.sh
  ```
  逻辑：脚本进入 `code/`，运行 `python -m gen_examples.examples --save_dir ...`。`examples.py` 支持通过命令行或环境变量覆盖训练集根目录、采样数量与语言列表，输出 JSON 示例供 `run_generation.py` few-shot 使用。

- **硬负例挖掘**：
  ```bash
  cd data_generation/data_augmentation
  TASK_TYPE=covidretrieval ROUNDS=3 ./script/run_hn_mine.sh
  ```
  逻辑：脚本固定到仓库根目录，按轮次读取 `prod_augmentation` 产出的三元组，调用 `mine_v2_modified.py` 构建索引并追加硬负例。各类路径（生成数据、索引目录、底库路径、模型名等）均支持环境变量覆盖。

- **部署开源 LLM 服务**：
  ```bash
  cd data_generation/data_augmentation
  MODEL_PATH=/data/share/project/shared_models/Qwen2-5-72B-Instruct ./script/run_open_source_llm.sh
  ```
  逻辑：脚本将根目录固定后调用 `vllm_deploy/run_open_source_llm.py`，主要运行参数（模型路径、服务名、并行度、显存占用等）可在命令前通过环境变量调整。

### 代码运行链路概览

1. `run_generation.sh`/`run_corpus_generation.py` 读取 `task_configs.py`，确定当前 `task_type` 的语料、示例与过滤规则，并加载 `TaskType` 对应的提示模板（`constant.py`）。
2. `CorpusGenerator` 根据配置加载或过滤语料，`TripletGenerator` 基于 `get_generation_prompt` 生成查询，提示会随 `task_type` 切换，从而同一套 pipeline 可直接输出 AILAStatutes 与 SCIDOCS 的查询。
3. 改写语料阶段，`DocSynthesisGenerator` 为每条种子文档调用 `sample_attributes_for_task` 采样叙事风格，拼装 `get_base_synthesis_prompt`，并通过 `LLM.chat` 生成标题与正文，落盘时记录 `seed_id` 及属性以方便回溯。
4. 生成的三元组按任务与语言分目录存储，后续 `run_hn_mine.sh` 可复用这些产物进行硬负例挖掘；`gen_examples/examples.py` 则提供少量示例，进一步加强 few-shot 效果。

## 代码文件与函数链路详解（改写语料）
- `code/run_corpus_generation.py`：脚本入口，解析命令行后会：
  1. 根据 `task_configs.py` 的默认路径或 CLI 覆盖项，调用 `CorpusGenerator.run` 加载种子文档并可选按 qrels 过滤；
  2. 初始化 `DocSynthesisGenerator`，并传入模型类型、端口等 LLM 配置；
  3. 通过 `run(...)` 批量改写，传入线程数、每条种子生成的变体数量，最后将结果写入 `generated_corpus` 下的 JSONL。
- `code/corpus_generator.py`：负责按任务读取底库 Arrow/JSONL，并使用 `task_configs.py` 中的字段映射（`text_key`、`id_key` 等）完成清洗、长度过滤和 qrels 正例过滤，返回统一格式的种子文档列表。
- `code/doc_synthesis_generator.py`：承载主要的 LLM 调用逻辑。
  - `_build_prompt` 会调用 `attributes_config.get_base_synthesis_prompt`，并附加 `attributes_to_hint_text` 返回的叙事提示，确保改写后的文档在语气、结构上与种子有明显差异。
  - `generate_single_doc` 为每条种子采样属性（`sample_attributes_for_task`），并使用 `LLM.chat` 生成输出，随后解析 `title` 和 `desc` 字段并保留原始 `seed_id`。
  - `run` 负责并行调度，将上述函数组合成线程池任务，最终返回包含属性与追溯信息的改写文档列表。
- `code/attributes_config.py`：定义跨任务的叙事属性空间（风格、口吻、结构等），并提供 `sample_attributes_for_task`、`attributes_to_hint_text`、`get_base_synthesis_prompt` 等工具帮助构建 Prompt。
