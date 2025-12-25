import os
import sys
import json
import random
import numpy as np
from tqdm import tqdm
from typing import Optional, List
from dataclasses import dataclass, field
from datasets import Dataset

import faiss
from transformers import HfArgumentParser
from FlagEmbedding import FlagAutoModel
from FlagEmbedding.abc.inference import AbsEmbedder

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_generation.data_preparation.code.task_configs import get_task_config  # <-- 引入 get_task_config
from data_generation.shared.constants import get_task
from data_generation.shared.utils import compute_md5

# ====================================================================
# 💡 修正 1：将 get_corpus_dict 和 get_corpus_from_arrow 提前
# ====================================================================

# load corpus dict
def get_corpus_dict(file_path: str):
    corpus_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            if "text" in line.keys():
                if "title" in line.keys():
                    # NOTE: use "{title} {text}" format
                    text = line['title'] + "\n" + line['text']
                else:
                    text = line['text']
                docid = compute_md5(text)
                corpus_dict[docid] = text
            else:
                if "pos" in line.keys():
                    for text in line['pos']:
                        docid = compute_md5(text)
                        corpus_dict[docid] = text
                if "neg" in line.keys():
                    for text in line['neg']:
                        docid = compute_md5(text)
                        corpus_dict[docid] = text
    return corpus_dict

# get corpus from arrow(new fuction)
def get_corpus_from_arrow(arrow_path: str):
    """
    Load corpus from HuggingFace arrow dataset, return corpus_dict: {md5(text): text}.
    Arrow file must contain either:
        - {"id":..., "text":...}
        - {"text":...}
    """
    ds = Dataset.from_file(arrow_path)
    corpus_dict = {}

    for row in ds:
        if "text" not in row:
            continue

        text = row["text"]
        docid = compute_md5(text)
        corpus_dict[docid] = text

    print(f"[INFO] Loaded {len(corpus_dict)} items from arrow: {arrow_path}")
    return corpus_dict

# ====================================================================
# DataArgs 和 ModelArgs 定义保持不变
# ====================================================================

@dataclass
class DataArgs:
    """
    Data arguments for hard negative mining.
    """
    # 💡 确保非默认参数在前
    task_type: str = field(
        metadata={"help": "The task type.", "required": True}
    )
    input_file: str = field(
        metadata={"help": "The input file for hard negative mining."}
    )
    output_file: str = field(
        metadata={"help": "The output file for hard negative mining."}
    )
    
    # --- 接着是默认参数 ---
    language: str = field(
        default="en", metadata={"help": "The language code."}
    )
    candidate_pool: Optional[List[str]] = field(
        default=None, metadata={"help": "The candidate pool for hard negative mining. If None/empty, defaults to original corpus_path from task_configs."}
    )
    index_save_dir: Optional[str] = field(
        default=None, metadata={"help": "The path to save the index."}
    )
    search_top_k: int = field(
        default=500, metadata={"help": "The top k for searching."}
    )
    range_for_sampling: str = field(
        default="30-200", metadata={"help": "The range to sample negatives."}
    )
    negative_number: int = field(
        default=15, metadata={"help": "The number of negatives."}
    )
    use_gpu_for_searching: bool = field(
        default=False, metadata={"help": "Whether to use faiss-gpu for searching."}
    )
    search_batch_size: int = field(
        default=64, metadata={"help": "The batch size for searching."}
    )
    add_doc_prefix_for_e5: bool = field(
        default=False, metadata={"help": "Whether to add prefix for e5."}
    )


@dataclass
class ModelArgs:
    """
    Model arguments for embedder.
    """
    embedder_name_or_path: str = field(
        metadata={"help": "The embedder name or path.", "required": True}
    )
    embedder_model_class: Optional[str] = field(
        default=None, metadata={"help": "The embedder model class. Available classes: ['encoder-only-base', 'encoder-only-m3', 'decoder-only-base', 'decoder-only-icl']. Default: None. For the custom model, you need to specifiy the model class.", "choices": ["encoder-only-base", "encoder-only-m3", "decoder-only-base", "decoder-only-icl"]}
    )
    normalize_embeddings: bool = field(
        default=True, metadata={"help": "whether to normalize the embeddings"}
    )
    pooling_method: str = field(
        default="cls", metadata={"help": "The pooling method fot the embedder."}
    )
    use_fp16: bool = field(
        default=True, metadata={"help": "whether to use fp16 for inference"}
    )
    devices: Optional[str] = field(
        default=None, metadata={"help": "Devices to use for inference.", "nargs": "+"}
    )
    query_instruction_for_retrieval: Optional[str] = field(
        default=None, metadata={"help": "Instruction for query"}
    )
    query_instruction_format_for_retrieval: str = field(
        default="{}{}", metadata={"help": "Format for query instruction"}
    )
    examples_for_task: Optional[str] = field(
        default=None, metadata={"help": "Examples for task"}
    )
    examples_instruction_format: str = field(
        default="{}{}", metadata={"help": "Format for examples instruction"}
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code"}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Cache directory for models."}
    )
    # ================ for inference ===============
    batch_size: int = field(
        default=512, metadata={"help": "Batch size for inference."}
    )
    embedder_query_max_length: int = field(
        default=512, metadata={"help": "Max length for query."}
    )
    embedder_passage_max_length: int = field(
        default=512, metadata={"help": "Max length for passage."}
    )

# ... (create_index, save_index, load_index, batch_search 保持不变)

# create index
def create_index(embeddings: np.ndarray, use_gpu: bool = False):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


# save index
def save_index(index: faiss.Index, docid: list, index_save_dir: str):
    docid_save_path = os.path.join(index_save_dir, 'docid')
    index_save_path = os.path.join(index_save_dir, 'index')
    with open(docid_save_path, 'w', encoding='utf-8') as f:
        for _id in docid:
            f.write(str(_id) + '\n')
    faiss.write_index(index, index_save_path)


# load index
def load_index(index_save_dir: str, use_gpu: bool = False):
    index_path = os.path.join(index_save_dir, "index")
    docid_path = os.path.join(index_save_dir, "docid")
    
    docids = []
    with open(docid_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            docids.append(line.strip())
    
    index = faiss.read_index(index_path)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    return docids, index


# batch search
def batch_search(
    index: faiss.Index,
    query: np.ndarray,
    topk: int = 200,
    batch_size: int = 64
):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs

# find knn negatives (core function)
def find_knn_neg(
    model: AbsEmbedder,
    input_file: str,
    output_file: str,
    search_top_k: int = 2000,
    candidate_pool: Optional[List[str]] = None,
    index_save_dir: Optional[str] = None,
    sample_range: str = "30-200",
    negative_number: int = 15,
    use_gpu: bool = False,
    add_doc_prefix_for_e5: bool = False,
):
    # 💡 get_corpus_dict 现在已被提前定义，NameError 解决
    corpus_dict = get_corpus_dict(input_file) 
    print (f'corpus size from input file: {len(corpus_dict)}--------------')
    queries = []
    train_data = []
    poses = []
    pos_docids = set()
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            train_data.append(line)
            queries.append(line['query'])
            poses.append(line['pos'][0])
            for text in line['pos']:
                docid = compute_md5(text)
                pos_docids.add(docid)

        if candidate_pool is not None:
        # print(candidate_pool, type(candidate_pool))
            if not isinstance(candidate_pool, list):
                # 💡 确保 candidate_pool 是列表，因为 default_generated_corpus_path 返回的是字符串
                corpus_dict.update(get_corpus_dict(candidate_pool))
            else:
                for cp in candidate_pool:
                    if cp.endswith(".arrow"):
                        # load arrow dataset
                        corpus_dict.update(get_corpus_from_arrow(cp))
                    else:
                        # assume jsonl
                        corpus_dict.update(get_corpus_dict(cp))
    
    # encode queries and poses
    print(f'inferencing embedding for queries (number={len(queries)})--------------')
    q_vecs = model.encode_queries(queries)
    print(f'inferencing embedding for positives (number={len(poses)})--------------')
    if add_doc_prefix_for_e5:
        pos_vecs = model.encode_corpus(["passage: " + text for text in poses])
    else:
        pos_vecs = model.encode_corpus(poses)

    # encode corpus if necessary
    if index_save_dir is not None:
        if os.path.exists(os.path.join(index_save_dir, 'index')):
            print(f'loading index from {index_save_dir}--------------')
            docids, index = load_index(index_save_dir, use_gpu=use_gpu)
            if len(docids) > len(corpus_dict):
                raise ValueError(f'index size mismatch, loading corpus from {input_file}--------------')
        else:
            print(f'creating index and saving to {index_save_dir}--------------')
            os.makedirs(index_save_dir, exist_ok=True)
            docids = list(corpus_dict.keys())
            
            # sample docids to avoid memory overflow
            if len(docids) > 12_000_000:
                docids = random.sample(docids, 12_000_000)
            
            corpus = [corpus_dict[docid] for docid in docids]
            if add_doc_prefix_for_e5:
                p_vecs = model.encode_corpus(["passage: " + text for text in corpus])
            else:
                p_vecs = model.encode_corpus(corpus)
            index = create_index(p_vecs, use_gpu=False)
            save_index(index, docids, index_save_dir)
            docids, index = load_index(index_save_dir, use_gpu=use_gpu)
    else:
        print(f'creating index without saving--------------')
        docids = list(corpus_dict.keys())
        
        # sample docids to avoid memory overflow
        if len(docids) > 12_000_000:
            docids = random.sample(docids, 12_000_000)
        
        corpus = [corpus_dict[docid] for docid in docids]
        if add_doc_prefix_for_e5:
            p_vecs = model.encode_corpus(["passage: " + text for text in corpus])
        else:
            p_vecs = model.encode_corpus(corpus)
        index = create_index(p_vecs, use_gpu=use_gpu)
    
    raw_scores = np.einsum('ij,ij->i', q_vecs, pos_vecs)
    
    results_list = [{} for _ in range(len(queries))]
    
    # search for negatives
    print('searching for negatives--------------')
    all_scores, all_inxs = batch_search(index, q_vecs, topk=search_top_k)
    assert len(all_inxs) == len(train_data)
    
    for idx, (scores, inxs) in enumerate(zip(all_scores, all_inxs)):
        results = {}
        for score, inx in zip(scores, inxs):
            if inx != -1:
                results[docids[inx]] = score
        results_list[idx].update(results)

    # sort the results
    for idx, results in enumerate(results_list):
        results_list[idx] = sorted(list(results.items()), key=lambda x: x[1], reverse=True)

    # fill negatives
    for i, data in enumerate(train_data):
        skip_list = data['pos'] + [data['query']]
        
        results = results_list[i]
        if raw_scores[i] * 0.95 < results[min(200, len(results) - 1)][1]:
            # neg scores are greater than 0.95 * pos score
            range_results = results_list[i][30:200]
            
            neg_list = []
            for docid, _ in range_results:
                if docid not in corpus_dict:
                    print(f"Skip {docid}")
                    continue
                doc = corpus_dict[docid]
                if doc not in skip_list:
                    neg_list.append(docid)

            if len(neg_list) > negative_number:
                neg_list = random.sample(neg_list, negative_number)
        else:
            # neg scores are less than 0.95 * pos score
            neg_list = []
            for docid, score in results:
                if docid not in corpus_dict:
                    print(f"Skip {docid}")
                    continue
                doc = corpus_dict[docid]
                if doc in skip_list:
                    continue
                if len(neg_list) >= negative_number:
                    break
                if score < raw_scores[i] * 0.95:
                    neg_list.append(docid)
        
        if len(neg_list) > negative_number:
            neg_list = random.sample(neg_list, negative_number)
        elif len(neg_list) < negative_number:
            # remove positives from candidates
            docids_rm_pos = list(set(docids) - set(pos_docids))
            # 确保随机采样不超过可用数量
            sample_count = min(negative_number - len(neg_list), len(docids_rm_pos))
            if sample_count > 0:
                neg_list.extend(random.sample(docids_rm_pos, sample_count))
        
        data['neg'] = [corpus_dict[docid] for docid in neg_list]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding="utf-8") as f:
        for data in train_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"Done! The results are saved in {output_file}")


def load_model(model_args: ModelArgs):
    model = FlagAutoModel.from_finetuned(
        model_name_or_path=model_args.embedder_name_or_path,
        model_class=model_args.embedder_model_class,
        normalize_embeddings=model_args.normalize_embeddings,
        pooling_method=model_args.pooling_method,
        use_fp16=model_args.use_fp16,
        query_instruction_for_retrieval=model_args.query_instruction_for_retrieval,
        query_instruction_format=model_args.query_instruction_format_for_retrieval,
        devices=model_args.devices,
        examples_for_task=model_args.examples_for_task,
        examples_instruction_format=model_args.examples_instruction_format,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
        batch_size=model_args.batch_size,
        query_max_length=model_args.embedder_query_max_length,
        passage_max_length=model_args.embedder_passage_max_length,
    )
    return model


def main(data_args: DataArgs, model_args: ModelArgs):
    # 💡 修正 2：清理 task_type 和 language 中的多余引号
    if data_args.task_type.startswith('"') and data_args.task_type.endswith('"'):
        data_args.task_type = data_args.task_type.strip('"')
    if data_args.language.startswith('"') and data_args.language.endswith('"'):
        data_args.language = data_args.language.strip('"')

    # 💡 修正 3：处理 candidate_pool 为空的情况，使用 task_config.corpus_path
    if data_args.candidate_pool == [""]: 
        try:
            # 获取任务配置
            cfg = get_task_config(data_args.task_type)
            # 使用原始语料库路径作为默认候选池
            default_pool_path = cfg.corpus_path 
            
            if default_pool_path is None:
                raise ValueError(f"corpus_path is None for task '{data_args.task_type}' in task_configs.")

            # 设为默认值（用列表包装以兼容 find_knn_neg 的逻辑）
            data_args.candidate_pool = [default_pool_path] 
            print(f"[INFO] Using default candidate pool path (Original Corpus): {default_pool_path}")
        except Exception as e:
            # 捕获 get_task_config 失败或 corpus_path 为 None 的情况
            print(f"[WARNING] Could not determine default candidate pool for task '{data_args.task_type}'. Error: {e}")
            # 如果找不到，将其设为 None
            data_args.candidate_pool = None 

    # 💡 修正 4：处理 query_instruction_for_retrieval 为空的情况
    if model_args.query_instruction_for_retrieval == "":
        try:
            # 从 constant 模块获取默认的任务指令
            task_obj = get_task(data_args.task_type, data_args.language)
            task_instruction = task_obj.task_instruction
            if task_instruction is None:
                 raise ValueError(f"TaskType.{data_args.task_type} 在 constant 中未定义任务说明 (value)")

            model_args.query_instruction_for_retrieval = task_instruction
            print(f"[INFO] Using default query instruction: {task_instruction}")
        except Exception as e:
            # 这里的错误通常是由于 task_type 找不到或 get_task 失败
            print(f"[ERROR] Failed to load default query instruction for task '{data_args.task_type}'. Error: {e}")
            # 如果找不到，将指令设为 None
            model_args.query_instruction_for_retrieval = None
    
    
    model = load_model(model_args)


    if os.path.exists(data_args.output_file):
        print(f'{data_args.output_file} already exists, skip')
        return
    
    # 解析 range_for_sampling 字符串为列表
    try:
        sample_range_list = [int(x) for x in data_args.range_for_sampling.split('-')]
    except ValueError:
        raise ValueError(f"Invalid format for range_for_sampling: {data_args.range_for_sampling}. Expected format like '30-200'.")


    find_knn_neg(
        model=model,
        input_file=data_args.input_file,
        output_file=data_args.output_file,
        search_top_k=data_args.search_top_k,
        candidate_pool=data_args.candidate_pool,
        index_save_dir=data_args.index_save_dir,
        sample_range=data_args.range_for_sampling, 
        negative_number=data_args.negative_number,
        use_gpu=data_args.use_gpu_for_searching,
        add_doc_prefix_for_e5=data_args.add_doc_prefix_for_e5,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((
        DataArgs,
        ModelArgs
    ))
    data_args, model_args = parser.parse_args_into_dataclasses()
    data_args: DataArgs
    model_args: ModelArgs
    main(data_args, model_args)
