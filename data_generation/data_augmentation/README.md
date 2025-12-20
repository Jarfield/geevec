# 数据增强与查询生成指南

本目录提供从“语料改写”到“查询生成”的完整流水线，主要包含两个脚本：

- `code/run_corpus_generation.py`：将原始语料改写成合成语料（generated_corpus）。
- `code/run_generation.py`：基于合成语料逐条生成查询-正例三元组。

下文按函数、脚本参数与使用示例展开说明，便于快速排查和二次开发。

## 主要函数职责
- **`get_pair_scoring_prompt`（code/constant.py）**：为“查询-文档”相关性打分生成 LLM 提示词，输出固定的 `<score>1-5</score>` 标签。
- **`QueryDocPairScorer.score_pair`（code/query_doc_pair_scorer.py）**：调用 LLM 对单个 query-doc 生成分数，可选基于 MD5 的缓存。
- **`QueryDocPairScorer.score_item` / `run`**：批量遍历 mined 数据（`query` + `pos/neg/topk`），根据阈值切分正例 / 难负例并保存 `score_details`。
- **`load_generated_corpus`（code/run_generation.py）**：读取 `run_corpus_generation.py` 产出的合成语料（`.jsonl` 或 Arrow`），只要含有 `text/desc` 字段即保留，可选 `num_samples` 截断。
- **`load_examples_pool`（code/run_generation.py）**：加载 few-shot 示例，兼容 JSON / JSONL，并自动将不同字段名映射成 `{input, output}` 结构（支持 `input/context/content/text/doc/document/pos/positive/passage` 与 `output/target/query/question/label`），缺失字段会被跳过。
- **`gen_triplets`（`code/run_generation.py` + `triplet_generator.py`）**：包装 `TripletGenerator.run`，按进程数并行生成查询与正例。
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

### `code/run_pair_scoring.py`
基于 hn_mine / 检索候选的 query-doc 对进行 LLM 打分，便于过滤弱负样本：
- `--task_type` / `--language`：任务与语言枚举。
- `--input_path`：输入 JSONL，需包含 `query`，候选文档可以放在 `pos`/`neg`/`topk` 中。
- `--output_path`：打分后输出路径（会更新 `pos`/`neg`，并在 `score_details` 中保留每个文档的原始分数）。
- `--pos_threshold` / `--neg_threshold`：阈值，默认 ≥4 记为正例，≤2 记为难负。
- `--num_processes`：并行线程数；`--num_samples` 可截断用于快速试跑。
- `--cache_dir`：query+doc 级别的缓存目录，避免重复打分。
- `--overwrite`：目标文件存在时是否覆盖。

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

### `script/run_pair_scoring.sh`
封装 `run_pair_scoring.py`：
- `TASK_TYPE` / `LANGUAGE`：决定提示模板与输出目录后缀。
- `INPUT_PATH` / `OUTPUT_PATH`：输入/输出 JSONL 路径，默认沿用 `run_hn_mine.sh` 的输出目录。
- `POS_THRESHOLD` / `NEG_THRESHOLD`：打分阈值，覆盖 `run_pair_scoring.py` 的默认值。
- `MODEL_NAME` / `MODEL_TYPE` / `PORT` / `NUM_PROCESSES`：LLM 与并行配置。
- `CACHE_DIR`：启用则自动复用 query-doc 级别缓存。
- `OVERWRITE`：是否覆盖已有的输出文件。

### `code/export_original_pairs.py` & `script/run_export_original.sh`
将**原始元数据（非合成语料）**中 `score >= 1` 的 query-doc 正例抽取成 `{prompt, query, pos, neg}` JSONL，方便直接用测试集格式做训练对比。

关键参数（CLI 或同名环境变量均可）：
- `task_type` / `TASK_TYPE`：任务名，枚举见 `constant.TaskType`。
- `language` / `LANGUAGE`：语言后缀，用于输出文件命名。
- `corpus_path` / `CORPUS_PATH`：corpus 文件路径，默认来自 `task_configs`。
- `queries_path` / `QUERIES_PATH`：query 文件路径，默认来自 `task_configs`。
- `qrels_path` / `QRELS_PATH`：qrels 文件路径，默认来自 `task_configs`。
- `corpus_id_key`、`corpus_text_key`、`corpus_title_key`：corpus 字段名；若提供 `title` 会与正文拼接。
- `query_id_key`、`query_text_key`：query 文件的 id/text 字段名。
- `qrels_qid_key`、`qrels_pid_key`、`qrels_score_key`：qrels 中的 query/doc/score 字段名。
- `positive_score`：认定为正例的最小分数，默认 1。
- `min_len`：过滤过短文档的最小长度（默认跟随 `task_configs`）。
- `max_queries`：最多保留多少条 query（-1 表示全量）。
- `save_root`：输出根目录，默认 `DATA_AUG_ORIGINAL_ROOT`（与 `generated_data` 同层的 `original_data`）。
- `output_path`：完全自定义的输出路径，优先级最高。
- `prompt`：输出中 `prompt` 字段的值，默认空字符串；
- `overwrite` / `OVERWRITE`：已存在文件是否覆盖，1 表示覆盖。

所有任务的数据路径与字段名已集中在 `code/task_configs.py` 中（covidretrieval / arguana / scidocs / ailastatutes 已填好默认路径），通常只需指定 `TASK_TYPE` 与 `LANGUAGE` 即可运行脚本，无需再手动传入 corpus/queries/qrels 参数。

示例：
```bash
cd data_generation/data_augmentation
TASK_TYPE="covidretrieval" LANGUAGE="zh" POSITIVE_SCORE=1 \
QRELS_PATH="/path/to/qrels.arrow" QUERIES_PATH="/path/to/queries.arrow" \
./script/run_export_original.sh
# 输出：<DATA_AUG_ORIGINAL_ROOT>/covidretrieval/original_pairs/zh_original.jsonl
```

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
   - 若示例文件是 JSONL 或字段名为 `context/query/pos` 等，均无需修改格式，脚本会自动映射并拼接列表字段。
   - 如需复用现有生成缓存，可设置 `--cache_dir`；多轮生成时会自动分轮使用不同缓存目录。

3. **对 hn_mine 候选进行 query-doc 打分，筛难负样本**
   ```bash
   cd data_generation/data_augmentation
   TASK_TYPE="covidretrieval" LANGUAGE="zh" \
   INPUT_PATH="/path/to/hn_mine.jsonl" OUTPUT_PATH="/path/to/hn_mine_scored.jsonl" \
   MODEL_NAME="Qwen2-5-72B-Instruct" PORT=8000 NUM_PROCESSES=8 \
   ./script/run_pair_scoring.sh
   ```
   - 默认 ≥4 视为正例，≤2 视为难负；可通过 `POS_THRESHOLD`、`NEG_THRESHOLD` 覆盖。
   - 若提供 `CACHE_DIR`，会按 query+doc MD5 缓存打分，避免重复请求 LLM。

## 详细说明
### 输入输出关系
- **run_corpus_generation.py**：
  - 输入：原始语料（`corpus_path` 或默认路径）、可选 qrels、模型信息。
  - 输出：含 `title/desc`/`text` 的合成语料 JSONL（`generated_corpus`）。
- **run_generation.py**：
  - 输入：合成语料、可选 few-shot 示例、模型信息、生成参数。
  - 输出：查询-正例三元组 JSONL，按任务与语言分目录存储；多轮模式会生成 `_round{idx}` 后缀文件。
- **run_pair_scoring.py**：
  - 输入：含 `query` 及候选文档字段（`pos/neg/topk` 任一即可）的 JSONL，通常来自 `run_hn_mine.sh` 输出。
  - 输出：在 `score_details` 中补充每个文档的相关性分数，并依据阈值重写 `pos` / `neg`，便于后续筛选或训练。

### 常见排查
- **示例字段不匹配导致报错**：`load_examples_pool` 会尝试从 `input/context/content/text/doc/document/pos/positive/passage` 里取输入，`output/target/query/question/label` 里取输出；若字段为列表会自动拼接，缺失会打印警告并跳过，不再因 KeyError 终止生成。
- **没有生成任何三元组**：确认 `generated_corpus/<lang>_synth_corpus.jsonl` 存在且包含 `text/desc` 字段；`--num_samples` 不要设置为 0；确保 LLM 服务可用并端口正确。
- **希望并发加速**：`--num_processes` 控制线程池大小，通常设置为 CPU 核心数的 70%-80% 即可。

### 多轮生成的叙事焦点
- `covidretrieval`：轮换 `covid_fact_detail`、`covid_policy_measure`、`covid_vaccine_treatment`、`covid_risk_protection`、`covid_social_impact`。
- `ailastatutes`：轮换 `victim_focus`、`investigation_focus`、`judgment_focus`、`social_impact_focus`、`neutral_brief`。
- 其他任务：默认不使用 `narrative_focus`，每轮仍会单独输出文件。

有任何新的数据格式或任务需求，可以在 `constant.py`/`task_configs.py` 中扩展任务枚举、提示模板与默认路径，即可复用同一套脚本完成生成。
