data_augmentation 说明
======================

本文档概述 `data_generation/data_augmentation` 下的 Python 代码与 Shell 脚本，帮助快速了解各函数的作用、实现逻辑与使用方式。示例路径均以仓库根目录为基准。

代码（code/）逐文件函数说明
--------------------------

### constant.py
- **get_task(task_type, language)**：根据字符串枚举名构造 `Task` 数据类实例，并自动填充任务说明。通过查表方式获得 `TaskType` 和 `Language` 枚举，无额外逻辑分支。
- **get_generation_prompt(task, text, examples=None, narrative_focus=None)**：拼接针对不同检索任务的查询生成提示词。先按任务选择生成指令与输出描述，插入语言占位，再可选附加叙事侧重点（如 COVID 的五类焦点），最后附带 few-shot 示例块，确保输出格式只包含生成文本。
- **get_pair_scoring_prompt(task, query, doc)**：生成 Query-Document 相关性打分提示，内含 1–5 分刻度说明，要求模型返回 `<score> </score>` 标签包裹的数字。
- **get_doc_synthesis_prompt(task, seed_text, narrative_attributes=None, extra_style_examples=None)**：为文档改写构造详细分步提示。描述内部分析/概括/改写流程、叙事属性要求、命名实体与数字替换规则，并固定输出格式为 `Title:` 与 `Desc:`。
- **get_quality_control_prompt(task, query, pos)**：生成二分类提示，判断文档能否回答查询。按任务切换判定说明与输出选项，最终要求输出 0/1。

### attributes_config.py
- **NarrativeAttributeBundle**：承载改写体裁、语气、细节、结构、受众、篇幅等属性的数据类，提供 `to_dict` 序列化。
- **sample_attributes_for_task(task_type, rng=None)**：在通用属性池基础上，融合任务特定候选项随机采样，返回 `NarrativeAttributeBundle`。
- **attributes_to_hint_text(bundle)**：将属性包转换为可直接插入 prompt 的中文提示，多行条目便于阅读。
- **get_base_synthesis_prompt(task, seed_text, narrative_hint=None, extra_style_examples=None)**：封装调用 `get_doc_synthesis_prompt`，便于对外保持兼容。

### llm.py
- **LLM.split_text(text, anchor_points=(0.4, 0.7))**：按随机锚点切分文本 token 序列，返回前后两段字符串。
- **LLM.chat(prompt, ...)**：统一的对话调用封装，支持 OpenAI、Azure、开源 vLLM 三类客户端。内部以线程执行请求并实现容错重试、超时保护，可选去除 `<think>` 片段。

### corpus_generator.py
- **CorpusGenerator.run(...)**：加载任务配置与语料，必要时按 qrels 过滤正样本，跳过空文本或过短样本，可选随机采样后返回 `{"_id","text"}` 列表。辅助函数 `_load_qrels_ids`、`_load_corpus_records`、`_filter_corpus` 分别完成 qrels 解析、语料读取与筛选。

### doc_synthesis_generator.py
- **DocSynthesisGenerator._build_prompt(task, seed_text, attr_bundle)**：把属性提示文本和基础改写 prompt 组合，去除结尾“Your output:”以减少模型赘词。
- **DocSynthesisGenerator.generate_single_doc(seed_doc, task, global_index, ...)**：为单条种子文档采样叙事属性，调用 `chat` 获取改写结果，清洗后拆分 `Title/Desc`，返回含 `seed_id` 与属性的合成文档；调试模式下保留原始输出和 prompt。
- **DocSynthesisGenerator.run(seed_docs, ...)**：批量生成改写文档。构造任务对象，为每个 seed 创建指定数量的变体，使用线程池并行执行 `_worker`，最终汇总结果列表。

### triplet_generator.py
- **TripletGenerator.generate_triplets(data, task, ...)**：针对单篇正例文本生成若干查询-正例对。可选 few-shot 示例与叙事焦点，调用 `get_generation_prompt` 并清洗模型输出，默认产出 `{prompt, query, pos, neg}` 结构。
- **TripletGenerator.run_single(data, task, ...)**：为单条数据提供生成+缓存逻辑，使用文本 MD5 命名缓存文件；调试模式保留全部生成结果，训练模式随机保留一条。
- **TripletGenerator.run(positives, task_type, ...)**：批量并行生成 triplets，先构造 `Task`，再用线程池映射到 `run_single` 并展平结果。

### query_doc_pair_scorer.py
- **QueryDocPairScorer.score_pair(query, doc, task, ...)**：调用 `get_pair_scoring_prompt` 让 LLM 打分，提取 `<score>` 数字；支持按 query/doc MD5 两级目录缓存。
- **QueryDocPairScorer.score_item(item, task, pos_threshold=4.0, neg_threshold=2.0, ...)**：对单条包含 `pos/neg/topk` 的样本逐文档打分，依据阈值重建正负例列表，并记录明细。
- **QueryDocPairScorer.run(data, task_type, ...)**：批量并行评分入口，内部构造任务并用线程池调用 `score_item`。

### run_generation.py
- **compute_md5(text)**：返回文本 MD5，用于去重。
- **get_args()**：定义查询生成脚本的所有命令行参数（任务、模型、样本量、进程数、语料路径、轮次数等）。
- **load_generated_corpus(corpus_path, num_samples)**：读取合成语料（JSONL/Arrow），提取 `text/desc` 字段并可截取前 N 条。
- **load_examples_pool(examples_path, sample_size)**：容错加载 few-shot 示例（JSON/JSONL），统一映射为 `{input, output}` 并随机抽样。
- **gen_triplets(...)**：构建并调用 `TripletGenerator.run`，主要做参数传递与缓存目录拼装。
- **get_save_path / save_triplets(...)**：生成单轮输出路径，合并旧文件并按 query/pos MD5 去重写入。
- **get_save_path_for_round / save_triplets_for_round(...)**：多轮模式下为每轮单独保存结果。
- **main(args)**：脚本入口。解析默认语料路径（依赖 `task_configs.default_generated_corpus_path`）、选择单轮或多轮生成流程，循环语言与轮次调用 `gen_triplets`，最终打印耗时。

### run_corpus_generation.py
- **get_args()**：定义语料改写脚本的参数（任务、语言、输出路径、模型、线程数、种子条数等）。
- **main(args)**：确定保存路径与覆盖策略，使用 `CorpusGenerator` 读取原始语料作为种子，再用 `DocSynthesisGenerator.run` 生成改写文档写入 JSONL。

### run_open_source_llm.py
- **get_args()**：解析开源 LLM 部署所需的模型路径、服务名、最大长度、并行度、显存占用、端口。
- **main()**：校验模型路径后组装 `vllm serve` 命令，直接通过 `os.system` 启动推理服务。

### export_original_pairs.py
- **_load_dataset(path)**：按扩展名选择 JSONL/Arrow 加载。
- **_to_lookup(ds, id_key, text_key, title_key, min_len)**：将数据集转换为 `id -> 文本` 映射，必要时拼接标题并过滤过短文本。
- **_load_positive_map(ds, qid_key, pid_key, score_key, min_positive)**：读取 qrels 构建 query 到正样本文档集合的映射。
- **parse_args() / resolve_paths(args, cfg)**：解析命令行并结合 `task_configs` 确定数据文件路径与字段名。
- **main()**：加载 corpus/queries/qrels，依据任务指令组装 `{prompt, query, pos, neg}` 结构列表，写入 `DATA_AUG_ORIGINAL_ROOT/<task>/original_pairs/<language>_original.jsonl`（或自定义路径）。

### mine.py（hard negative 挖掘核心）
- **get_corpus_dict(file_path) / get_corpus_from_arrow(arrow_path)**：从 JSONL 或 Arrow 语料读取文本并以内容 MD5 作为键构建字典。
- **DataArgs / ModelArgs 数据类**：分别描述挖掘所需的数据路径、采样范围、检索参数和向量模型配置（池化方式、max length、设备、批大小等）。
- **create_index(embeddings, use_gpu=False)**、**save_index(...)**：构建 FAISS 内积索引，可选 GPU；保存索引与文档 ID。
- 剩余函数（未全文列出）围绕向量化、批检索、负例筛选与结果落盘展开，逻辑集中在使用 FlagEmbedding 生成表示并按分数区间采样 hard negatives。

### run_open_source_llm.py 外的其余 run_* 入口
- **run_pair_scoring.py / run_hn_mine.py / run_corpus_generation.py / run_generation.py / run_export_original.py / run_gen_examples.py / run_open_source_llm.py / run_pair_scoring.py** 等脚本均通过 `get_args` + `main` 组合暴露 CLI，功能分别对应对偶打分、挖掘负例、语料改写、查询生成、导出原始对、生成示例、启动模型等。调用栈都基于上述核心类和工具函数。

脚本（script/）参数与示例
------------------------

以下示例均在仓库根目录执行。

- **run_generation.sh**：批量调用 `code/run_generation.py` 生成查询-正例三元组。关键环境变量：`TASK_TYPE`、`LANGUAGES`（空格分隔）、`NUM_EXAMPLES`、`NUM_SAMPLES`、`NUM_VARIANTS_PER_DOC`、`NUM_ROUNDS`、`NUM_PROCESSES`、`CACHE_DIR`、`EXAMPLES_DIR`、`MODEL_NAME`、`MODEL_TYPE`、`PORT`、`OVERWRITE`、`CORPUS_PATH`、`QRELS_PATH`。示例：  
  `TASK_TYPE=covidretrieval LANGUAGES="en zh" NUM_ROUNDS=3 ./data_generation/data_augmentation/script/run_generation.sh prod`

- **run_corpus.sh**：调用 `code/run_corpus_generation.py` 生成改写语料。环境变量：`TASK_TYPE`、`LANGUAGES`、`NUM_VARIANTS_PER_SEED`、`NUM_THREADS`、`NUM_SEED_SAMPLES`、`MODEL_NAME`、`MODEL_TYPE`、`PORT`、`OVERWRITE`、`SAVE_PATH`、`CORPUS_PATH`、`QRELS_PATH`。示例：  
  `TASK_TYPE=miracl NUM_VARIANTS_PER_SEED=2 ./data_generation/data_augmentation/script/run_corpus.sh`

- **run_pair_scoring.sh**：批量使用 `code/run_pair_scoring.py` 过滤/重打分查询-文档对。环境变量：`TASK_TYPE`、`LANGUAGE`、`INPUT_PATH`、`OUTPUT_PATH`、`POS_THRESHOLD`、`NEG_THRESHOLD`、`THREAD_COUNT`、`MODEL_NAME`、`MODEL_TYPE`、`PORT`、`CACHE_DIR`。示例：  
  `INPUT_PATH=./data/generated_data/covidretrieval/generation_results/prod_augmentation/en-triplets.jsonl OUTPUT_PATH=./scored.jsonl ./data_generation/data_augmentation/script/run_pair_scoring.sh`

- **run_hn_mine.sh**：驱动 `code/run_hn_mine.py` 进行硬负例挖掘。环境变量涵盖向量模型与检索配置：`TASK_TYPE`、`INPUT_FILE`、`OUTPUT_FILE`、`LANGUAGE`、`CANDIDATE_POOL`、`INDEX_SAVE_DIR`、`SEARCH_TOP_K`、`RANGE_FOR_SAMPLING`、`NEGATIVE_NUMBER`、`USE_GPU_FOR_SEARCHING`、`SEARCH_BATCH_SIZE`、`ADD_DOC_PREFIX_FOR_E5`、`EMBEDDER_NAME_OR_PATH` 等。示例：  
  `TASK_TYPE=scidocs INPUT_FILE=./triplets.jsonl OUTPUT_FILE=./triplets_hn.jsonl EMBEDDER_NAME_OR_PATH=BAAI/bge-base-en ./data_generation/data_augmentation/script/run_hn_mine.sh`

- **run_export_original.sh**：封装 `code/run_export_original.py`，从原始语料与 qrels 生成训练对。环境变量：`TASK_TYPE`、`LANGUAGE`、`SAVE_ROOT`、`OUTPUT_PATH`、`POSITIVE_SCORE`、`MAX_QUERIES` 等。示例：  
  `TASK_TYPE=arguana LANGUAGE=en OUTPUT_PATH=./original.jsonl ./data_generation/data_augmentation/script/run_export_original.sh`

- **run_gen_examples.sh**：调用 `code/run_gen_examples.py`（生成 few-shot 示例池）。主要变量：`TASK_TYPE`、`LANGUAGE`、`NUM_SAMPLES`、`SAVE_ROOT`、`MODEL_NAME`、`MODEL_TYPE`、`PORT`。示例：  
  `TASK_TYPE=ailastatutes NUM_SAMPLES=200 ./data_generation/data_augmentation/script/run_gen_examples.sh`

- **run_open_source_llm.sh**：简化启动开源 LLM 服务，包装 `code/run_open_source_llm.py`。变量：`MODEL_PATH`、`SERVE_NAME`、`MAX_LENGTH`、`PARALLEL_SIZE`、`GPU_MEMORY_UTILIZATION`、`PORT`。示例：  
  `MODEL_PATH=/models/Qwen2.5-72B-Instruct PORT=9000 ./data_generation/data_augmentation/script/run_open_source_llm.sh`

Prompt 与路径提示
-----------------
- 所有查询/打分/改写提示模板集中在 `code/constant.py`，调用端在 `TripletGenerator`（生成查询）、`QueryDocPairScorer`（相关性打分）、`DocSynthesisGenerator`（文档改写）中组装并传入 `LLM.chat`。
- 语料与示例默认路径由 `code/task_configs.py` 与相关脚本中的 `DEFAULT_GENERATED_ROOT`、`default_generated_corpus_path` 等函数推导；如需自定义，脚本参数/环境变量中的 `--corpus_path`、`--qrels_path`、`--save_dir`、`--save_path` 等可以覆盖默认值。
