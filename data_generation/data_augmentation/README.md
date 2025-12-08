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
2. **生成三元组**：
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

3. **改写语料（可选）**：
   ```bash
   python code/run_corpus_generation.py \
     --task_type arguana \
     --language en \
     --corpus_path /path/to/arguana.jsonl \
     --save_path /path/to/generated/arguana_synth.jsonl
   ```
   - 改写时会自动采样叙事属性，以提升风格多样性。

## 处理流程概览
- `run_generation.py` 通过 `CorpusGenerator` 加载语料、分发到 `TripletGenerator`，最终将输出写入 `<save_dir>/<language>/<task>/...`。
- `run_corpus_generation.py` 重用同一加载逻辑，随后由 `DocSynthesisGenerator` 在属性约束下改写语料并保存 JSONL。
- 所有脚本均通过命令行参数控制，可快速切换任务、路径和模型端口。

## 扩展到新任务
- 在 `constant.py` 中增加新的 `TaskType` 并补充提示模板。
- 在 `task_configs.py` 注册默认语料与示例路径。
- 在 `attributes_config.py` 为新任务添加特定属性选项，即可复用改写与生成流程。
