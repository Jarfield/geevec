# 关键检索任务的数据增强流水线

本文档总结了本仓库中四个核心检索任务的端到端数据构建流水线：**AILAStatutes**、**ArguAna**、**CovidRetrieval** 和 **SCIDOCS**。每条流水线都遵循相同的两阶段结构——先基于正例种子合成文档，再生成用于训练的（query, positive）样本对——并在下文突出各任务的特定输入与叙事/主题控制方式。

## 共享组件

* **种子加载与过滤**：`CorpusGenerator` 读取任务对应的语料与可选的 qrels，过滤掉过短文本；当提供 qrels 时，仅保留正例 id。它支持 Arrow 与 JSONL 语料格式，并支持通过 `--num_seed_samples` 进行可选的子采样。
* **文档合成（改写）**：`DocSynthesisGenerator` 将每个种子改写为一个或多个不同文风的变体，并采样任务特化的叙事属性（例如：AILAStatutes 的法律文体、ArguAna 的辩论语气等）。生成器会将 LLM 输出清洗为 `title` 与 `text` 字段，并保留原 seed id 与采样到的属性信息。
* **查询生成**：`run_augmentation.py` 加载过滤/改写后的语料（优先使用 `data_preparation/preparation` 输出，否则回退至 `<DATA_AUG_GENERATED_ROOT>/<task>/generation_results/generated_corpus/<lang>_synth_corpus.jsonl`），逐条输入 `TripletGenerator`，并可选使用 few-shot 示例。脚本会按语言/任务写出去重后的 triplets，并支持对部分任务启用带叙事焦点的多轮生成。
* **LLM 打分与正/负筛选**：`data_augmentation/script/run_pair_scoring.sh`（封装 `code/run_pair_scoring.py`）可对 mined 结果或 Stage B 产出的 `{query, pos/neg/topk}` 做相关性打分，并按阈值（默认 `≥4` 记正例、`≤2` 记难负）重写 `pos/neg` 字段，用于过滤弱负样本或强化正例池。
* **任务提示词（prompts）**：`get_generation_prompt` 为每个任务定义生成指令（例如：法条任务“生成一个情境”、ArguAna“生成一个观点/主张”等），并在多轮模式下插入可选的 focus 提示以引导不同类型的查询生成。
* **数据集默认配置**：`task_configs.py` 统一管理各任务的 corpus/qrels 路径、id/text 字段名、最小长度过滤阈值、示例目录等默认项；这些默认值均可通过 runner 脚本的 CLI 参数进行覆盖。

## AILAStatutes 流水线

1. **种子选择与精细切分**：基于 UKLEX 法律条款进行实验，先对极长的条款做细粒度切分，再加载（或拼接）MTEB AILA Statutes 语料与 qrels（`aila_statutes-corpus.arrow`, `aila_statutes-test.arrow`），保留长度大于 200 字符的片段；当提供 qrels 时，仅保留正例 id。
2. **法条改写**：运行 `run_corpus_generation.py --task_type ailastatutes` 生成风格化的法律描述。`DocSynthesisGenerator` 在改写前会从该任务的属性选项中采样法律特有体裁（如“法律条文式叙述”）、正式语气与结构化条目风格等，再对每条（切分后的）法条进行改写。
3. **情境生成**：对合成语料执行 `run_augmentation.py --task ailastatutes`。在多轮模式下，脚本会在不同轮次循环切换叙事焦点（受害者、调查、判决、社会影响、中性简述），从而让每条法条在不同轮次生成风格各异的情境；单轮模式则不进行 focus 控制。输出 triplet 中，生成的情境作为 `query`，法条文本作为 `pos`。
4. **相关性打分过滤**：若需要进一步清洗正负样本，可用 `run_pair_scoring.sh` 对生成或挖掘得到的 `{query, pos/neg}` 进行 LLM 打分，筛掉弱负并保留高分正例。

## ArguAna 流水线

1. **种子选择**：读取 ArguAna 的 JSONL 语料与测试 qrels；保留正例标签且长度 ≥200 字符的段落。默认路径指向 `shared_models/datasets--mteb--arguana` 下的 MTEB 快照。
2. **论证改写**：使用 `run_corpus_generation.py --task_type arguana` 对论证性段落进行改写。属性采样器会加入辩论式体裁与“立场鲜明”等语气特征，以在保留论证结构的同时提升文本多样性。
3. **主张（claim）生成**：在合成语料上运行 `run_augmentation.py --task arguana`。任务提示词要求模型生成“该段落会反驳的主张（claim）”，从而得到 `{query: claim, pos: passage}` 的 triplets。ArguAna 默认走单轮生成路径（不启用叙事焦点循环）。
4. **相关性打分过滤**：可选调用 `run_pair_scoring.sh` 对 `{query, pos/neg}` 进行 LLM 打分，清洗弱负样本或强化正例池。

## CovidRetrieval 流水线

1. **种子选择**：从 C-MTEB 快照加载 COVID 新闻语料与 dev qrels，保留长度超过 200 字符的文本，并限制为正例 qrels 对应的 id，以保证种子与 COVID-19 问题主题相关。
2. **新闻改写**：执行 `run_corpus_generation.py --task_type covidretrieval` 合成同新闻风格的文章。属性采样器会在基础属性之上加入疫情相关体裁与语气（如“疫情通报”“防疫政策解读”等），以拓展覆盖面。
3. **带焦点的多轮问题生成（复用同一份合成 doc）**：调用 `run_augmentation.py --task covidretrieval --num_rounds N`。在多轮模式下，脚本会在五类 focus 提示间循环切换——事实细节、政策措施、疫苗/治疗、风险/防护、社会影响——让同一份合成文档在不同轮次反复利用、产出多类问题。结果按轮次保存（例如 `en-triplets_round1.jsonl`），并支持跨运行去重。
4. **相关性打分过滤**：可选用 `run_pair_scoring.sh` 对多轮生成的 query-doc 进行 LLM 打分，留下高质量正例并收集难负样本。

## SCIDOCS 流水线

1. **Stage A：构造 SCIDOCS-like 训练集**：在 `data_generation/code_for_SCIDOCS` 中运行 `run_stage_a.sh`（核心脚本 `build_scirepeval_scidocs_like.py`），可按领域（默认 `Computer Science`）、年份与主题锚点过滤，并避开 MTEB-SCIDOCS 泄漏；输出 `{query, pos, neg}` 结构的基础样本。
2. **Stage B：追加 SciNCL 邻域（可选）**：运行 `run_stage_b.sh`（`add_scincl_neighbors.py`）为 Stage A 样本追加相似正例与难负邻居，便于扩充训练难度；需在脚本内实现 `encode_batch` 以接入自己的 embedding 模型。
3. **LLM 相关性打分**：Stage A/B 输出均可直接送入 `data_augmentation/script/run_pair_scoring.sh`，由 LLM 给出 1-5 分并基于阈值重写 `pos/neg`，实现第二阶段的正负筛选与降噪。
4. **标题生成（如需三元组）**：若希望继续统一的 query 生成流程，可在完成打分清洗后调用 `run_augmentation.py --task scidocs`，生成“可能会引用该摘要的论文标题”，得到 `{query: title, pos: abstract}` 的 triplets；SCIDOCS 默认单轮，不启用叙事焦点。
