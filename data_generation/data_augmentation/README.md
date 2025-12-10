# 数据增强与查询生成指南

本目录提供从“语料改写”到“查询生成”的完整流水线，主要包含两个脚本：

- `code/run_corpus_generation.py`：将原始语料改写成合成语料（generated_corpus）。
- `code/run_generation.py`：基于合成语料逐条生成查询-正例三元组。

下文按函数、脚本参数与使用示例展开说明，便于快速排查和二次开发。

## 主要函数职责
- **`load_generated_corpus`（code/run_generation.py）**：读取 `run_corpus_generation.py` 产出的合成语料（`.jsonl` 或 Arrow`），只要含有 `text/desc` 字段即保留，可选 `num_samples` 截断。
- **`load_examples_pool`（code/run_generation.py）**：加载 few-shot 示例，兼容 JSON / JSONL，并自动将不同字段名映射成 `{input, output}` 结构，缺失字段会被跳过。
- **`gen_triplets`（code/run_generation.py` + `triplet_generator.py`）**：包装 `TripletGenerator.run`，按进程数并行生成查询与正例。
- **`save_triplets` / `save_triplets_for_round`（code/run_generation.py）**：单轮 / 多轮模式下落盘三元组，自动创建目录并按语言、任务组织文件名。
- **`TripletGenerator.generate_triplets`（code/triplet_generator.py）**：对单条文档构造 Prompt 并调用 `LLM.chat`；可选 few-shot 示例与叙事焦点（`narrative_focus`）。
- **`TripletGenerator.run_single`**：提供缓存与重试入口；若指定 `cache_dir`，同一文档会复用历史生成结果。
- **`TripletGenerator.run`**：线程池并行调度 `run_single`，并透传任务、语言、示例、变体数等参数。
- **`CorpusGenerator.run`（code/corpus_generator.py）**：读取并清洗原始语料，可按 qrels 过滤正例；供 `run_corpus_generation.py` 调用。
- **`DocSynthesisGenerator.run`（code/doc_synthesis_generator.py）**：基于采样的叙事属性（`attributes_config.py`）改写种子文档，产出 `title/desc` 字段的合成语料。

## 脚本参数与含义
### `code/run_corpus_generation.py`
- `--task_type`：任务名，枚举见 `constant.TaskType`。
- `--language`：语言（ISO 639-1 简写）。
- `--save_path`：合成语料输出路径（默认 `DATA_AUG_GENERATED_ROOT/<task>/generation_results/generated_corpus/<lang>_synth_corpus.jsonl`）。
- `--cache_dir`：生成缓存目录。
- `--corpus_path`：自定义原始语料路径；默认读取 `task_configs.py` 中的配置。
- `--qrels_path`：可选 qrels 路径，用于过滤正例。
- `--num_variants_per_seed`：每条种子文档生成多少变体。
- `--num_threads`：并行线程数。
- `--num_seed_samples`：使用多少条种子文档（-1 表示全量）。
- `--model` / `--model_type` / `--port`：LLM 名称、类型与 vLLM 端口。
- `--overwrite`：若目标文件存在是否覆盖。

### `code/run_generation.py`
- `--task_type`：任务名，枚举见 `constant.TaskType`。
- `--save_dir`：三元组输出根目录（结构为 `<save_dir>/<language>/<task>/<...>.jsonl`）。
- `--examples_dir`：few-shot 示例根目录，若为空则使用 `task_configs` 默认路径。
- `--num_examples`：每次生成使用的示例数量。
- `--cache_dir`：生成缓存目录（也作为多轮缓存前缀）。
- `--corpus_path`：合成语料路径，未提供时默认读取 `generated_corpus/<lang>_synth_corpus.jsonl`。
- `--qrels_path`：已不再用于生成过滤，仅为兼容保留。
- `--language`：语言（ISO 639-1）。
- `--num_samples`：截取多少条合成语料参与生成；-1 表示全部。
- `--model` / `--model_type` / `--port`：LLM 名称、类型与 vLLM 端口。
- `--num_processes`：生成并行度（线程池大小）。
- `--overwrite`：是否覆盖已存在的输出文件。
- `--num_variants_per_doc`：单轮模式下每条文档生成多少条查询。
- `--num_rounds`：多轮生成次数；>1 时每轮输出单独文件，并可按任务自动轮换 `narrative_focus`。

### `script/run_corpus.sh`
通过环境变量封装 `run_corpus_generation.py`，常用变量：
- `TASK_TYPE`、`LANGUAGE`、`NUM_VARIANTS_PER_SEED`、`NUM_THREADS`、`NUM_SEED_SAMPLES`。
- `CACHE_DIR`、`CORPUS_PATH`、`QRELS_PATH`、`SAVE_ROOT`。
- `MODEL_NAME`、`MODEL_TYPE`、`PORT`、`OVERWRITE`（1 表示覆盖）。

### `script/run_generation.sh`
通过环境变量封装 `run_generation.py`，常用变量：
- `TASK_TYPE`、`LANGUAGES`（可逗号分隔多语言）。
- `NUM_EXAMPLES`、`NUM_SAMPLES`、`NUM_VARIANTS_PER_DOC`、`NUM_ROUNDS`、`NUM_PROCESSES`。
- `CACHE_DIR`、`EXAMPLES_DIR`、`CORPUS_PATH`。
- `MODEL_NAME`、`MODEL_TYPE`、`PORT`、`MODE`（影响 `save_dir` 子目录）、`OVERWRITE`。

## 快速上手
1. **生成合成语料**
   ```bash
   cd data_generation/data_augmentation
   TASK_TYPE="covidretrieval" LANGUAGE="zh" \
   NUM_VARIANTS_PER_SEED=2 NUM_THREADS=8 NUM_SEED_SAMPLES=1 \
   ./script/run_corpus.sh
   ```
   输出：`DATA_AUG_GENERATED_ROOT/<task>/generation_results/generated_corpus/<lang>_synth_corpus.jsonl`。

2. **基于合成语料生成查询**（默认按语料顺序逐条生成，不做 qrels 过滤）
   ```bash
   cd data_generation/data_augmentation/code
   python run_generation.py \
     --task_type covidretrieval \
     --language zh \
     --save_dir /data/share/project/psjin/data/generated_data/covidretrieval/generation_results/prod_augmentation \
     --num_samples 100 \
     --num_examples 10 \
     --model Qwen2-5-72B-Instruct --model_type open-source --port 8000
   ```
   - 若示例文件是 JSONL 或字段名为 `context/query`，无需修改格式，脚本会自动映射。
   - 如需复用现有生成缓存，可设置 `--cache_dir`；多轮生成时会自动分轮使用不同缓存目录。

## 详细说明
### 输入输出关系
- **run_corpus_generation.py**：
  - 输入：原始语料（`corpus_path` 或默认路径）、可选 qrels、模型信息。
  - 输出：含 `title/desc`/`text` 的合成语料 JSONL（`generated_corpus`）。
- **run_generation.py**：
  - 输入：合成语料、可选 few-shot 示例、模型信息、生成参数。
  - 输出：查询-正例三元组 JSONL，按任务与语言分目录存储；多轮模式会生成 `_round{idx}` 后缀文件。

### 常见排查
- **示例字段不匹配导致报错**：`load_examples_pool` 会尝试从 `input/context/content/text/doc/document` 里取输入，`output/target/query/question/label` 里取输出；缺失会打印警告并跳过，不再因 KeyError 终止生成。
- **没有生成任何三元组**：确认 `generated_corpus/<lang>_synth_corpus.jsonl` 存在且包含 `text/desc` 字段；`--num_samples` 不要设置为 0；确保 LLM 服务可用并端口正确。
- **希望并发加速**：`--num_processes` 控制线程池大小，通常设置为 CPU 核心数的 70%-80% 即可。

### 多轮生成的叙事焦点
- `covidretrieval`：轮换 `covid_fact_detail`、`covid_policy_measure`、`covid_vaccine_treatment`、`covid_risk_protection`、`covid_social_impact`。
- `ailastatutes`：轮换 `victim_focus`、`investigation_focus`、`judgment_focus`、`social_impact_focus`、`neutral_brief`。
- 其他任务：默认不使用 `narrative_focus`，每轮仍会单独输出文件。

有任何新的数据格式或任务需求，可以在 `constant.py`/`task_configs.py` 中扩展任务枚举、提示模板与默认路径，即可复用同一套脚本完成生成。
