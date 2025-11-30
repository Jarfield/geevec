"""
如何使用 MMTEB 里面的代码加载 corpus 和 qrels: https://github.com/embeddings-benchmark/mteb/blob/1.39.7/mteb/abstasks/AbsTaskRetrieval.py#L291
"""
import os
import random
import datasets
from tqdm import tqdm

AILAStatutes_DATA_ROOT = "/share/project/shared_datasets/UK-LEX"
# AILAStatutes_TRAIN_ROOT = "/share/project/psjin/dataset/Qwen2-5-72B-Instruct-llm-data_with_prompt/AILAStatutes"

class CorpusGenerator:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
    
    # --- 原版：依赖 qrels 做过滤，先整体注释掉，方便以后切回 ---
    # def _load_corpus(self, corpus_path: str, qrels_path: str):
    #     assert corpus_path.endswith('.jsonl'), "Invalid file format, please use .jsonl."
    #     assert qrels_path.endswith('.jsonl'), "Invalid file format, please use .jsonl."
    #
    #     # TODO: load corpus
    #     corpus = datasets.load_dataset(
    #         'json', 
    #         data_files=corpus_path, 
    #         cache_dir=self.cache_dir
    #     )['train']
    #
    #     # --- print corpus size ---
    #     print(f"[INFO] Loaded corpus size: {len(corpus)}")
    #
    #
    #     # TODO: load qrels
    #     qrels = datasets.load_dataset(
    #         'json', 
    #         data_files=qrels_path, 
    #         cache_dir=self.cache_dir
    #     )['train']
    #
    #     # --- print qrels size ---
    #     print(f"[INFO] Loaded qrels size: {len(qrels)}")
    #
    #     skip_docids = set()
    #     for qrel in qrels:
    #         score = int(qrel.get("score", 0))
    #         if score > 0:
    #             skip_docids.add(qrel["corpus-id"])
    #
    #     corpus_list = []
    #     for data in tqdm(corpus, desc="Loading corpus"):
    #         if data["_id"] not in skip_docids:
    #             text = data["title"] + "\n" + data["text"]
    #             # filter by length
    #             min_len = 200
    #             if len(text) < min_len:
    #                 continue
    #             # end filter by length
    #             corpus_list.append({"text": text})
    #
    #     # --- print final retained corpus size ---
    #     print(f"[INFO] Final filtered corpus size: {len(corpus_list)}")
    #
    #     return corpus_list

    # --- 新版：只用 corpus，不再用 qrels 过滤 ---
    def _load_corpus(self, corpus_path: str):
        assert corpus_path.endswith('.jsonl'), "Invalid file format, please use .jsonl."

        corpus = datasets.load_dataset(
            'json',
            data_files=corpus_path,
            cache_dir=self.cache_dir
        )['train']

        print(f"[INFO] Loaded corpus size: {len(corpus)}")

        corpus_list = []
        min_len = 200  # 长度过滤阈值，按需改

        for data in tqdm(corpus, desc="Loading corpus"):
            title = (data.get("title") or "")
            text_field = (data.get("text") or "")
            text = title + "\n" + text_field

            if len(text) < min_len:
                continue

            corpus_list.append({"text": text})

        print(f"[INFO] Final filtered corpus size: {len(corpus_list)}")
        return corpus_list
    
    def run(
        self,
        language: str,
        num_samples: int = -1,
    ):
        # --- 原版：MMTEB 的 AILAStatutes 路径 + qrels ---
        # corpus_path = os.path.join(AILAStatutes_DATA_ROOT, "corpus.jsonl")
        # qrels_path = os.path.join(AILAStatutes_DATA_ROOT, "queries.jsonl")
        # corpus_list = self._load_corpus(corpus_path, qrels_path)

        # --- 新版：使用你新造的 corpus，全量可用，不需要 queries/qrels ---
        # 如果你以后想换文件名，只改这里即可
        corpus_path = os.path.join(
            AILAStatutes_DATA_ROOT,
            "uklex_all_ailastatutes_corpus.jsonl",
        )
        corpus_list = self._load_corpus(corpus_path)
        
        if num_samples > 0 and num_samples < len(corpus_list):
            corpus_list = random.sample(corpus_list, num_samples)
        
        return corpus_list
