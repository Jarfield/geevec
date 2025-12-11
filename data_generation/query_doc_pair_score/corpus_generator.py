"""
如何使用 MMTEB 里面的代码加载 corpus 和 qrels: https://github.com/embeddings-benchmark/mteb/blob/1.39.7/mteb/abstasks/AbsTaskRetrieval.py#L291
"""
import os
import random
import datasets
from tqdm import tqdm


TRECCOVID_DATA_ROOT = "/data/share/project/shared_models/datasets--mteb--trec-covid/snapshots/44c0d5a1a986eedaaf740f2aa20922584f0ed045"
TRECCOVID_TRAIN_ROOT = "/data/share/project/shared_models/datasets--mteb--trec-covid/snapshots/bb9466bac8153a0349341eb1b22e06409e78ef4e/qrels"


class CorpusGenerator:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir
    
    def _load_corpus(self, corpus_path: str, qrels_path: str):
        assert corpus_path.endswith('.jsonl'), "Invalid file format, please use .jsonl."
        assert qrels_path.endswith('.jsonl'), "Invalid file format, please use .jsonl."
        
        # TODO: load corpus
        corpus = datasets.load_dataset('json', data_files=corpus_path, cache_dir=self.cache_dir)['train']
        
        # TODO: load qrels
        qrels = datasets.load_dataset('json', data_files=qrels_path, cache_dir=self.cache_dir)['train']
        skip_docids = set()
        for qrel in qrels:
            score = int(qrel.get("score", 0))
            if score > 0:
                skip_docids.add(qrel["corpus-id"])
                
        # load train data
        skip_docs = set()
        # try:
        #     train_data = datasets.load_dataset('json', data_files=train_path, cache_dir=self.cache_dir)['train']
        #     for data in train_data:
        #         for docid in data["neg"]:
        #             skip_docs.add(docid)
        # except:
        #     pass
        
        corpus_list = []
        for data in tqdm(corpus, desc="Loading corpus"):
            if data["_id"] not in skip_docids:
                text = data["title"] + "\n" + data["text"]
                
                if text not in skip_docs and text.strip() not in skip_docs and len(text.strip()) >= 200:
                    corpus_list.append({
                        "text": text,
                    })

        return corpus_list
    
    def run(
        self,
        language: str,
        num_samples: int = -1,
    ):
        corpus_path = os.path.join(TRECCOVID_DATA_ROOT, "corpus.jsonl")
        qrels_path = os.path.join(TRECCOVID_TRAIN_ROOT, "test.jsonl")
        # train_path = os.path.join(MIRACL_TRAIN_ROOT, f"miracl_{language}.jsonl")
        
        corpus_list = self._load_corpus(corpus_path, qrels_path)
        
        if num_samples > 0 and num_samples < len(corpus_list):
            corpus_list = random.sample(corpus_list, num_samples)
        
        return corpus_list
