"""
如何使用 MMTEB 里面的代码加载 corpus 和 qrels: https://github.com/embeddings-benchmark/mteb/blob/1.39.7/mteb/abstasks/AbsTaskRetrieval.py#L291
"""
import os
import random
from datasets import Dataset
from tqdm import tqdm
import json

COVID_ROOT = "shared_datasets/MMTEB/C-MTEB___covid_retrieval/default/0.0.0/1271c7809071a13532e05f25fb53511ffce77117"
COVID_ROOT_qerels = "shared_datasets/MMTEB/C-MTEB___covid_retrieval-qrels/default/0.0.0/a9f41b7cdf24785531d12417ce0d1157ed4b39ca"


class CorpusGenerator:
    def __init__(
        self,
        cache_dir: str = None,
        score_path: str = None,
        score_threshold: float = 3.0,
        score_id_key: str = "docid",
        score_score_key: str = "score",
    ):
        """
        :param score_path: 额外的 score 文件路径（例如 zh-scores.jsonl），可以为 None 表示不做这一步过滤
        :param score_threshold: 保留的最小分数（默认 3.0，条件是 score >= threshold）
        :param score_id_key: score 文件里 docid 的字段名（你的示例是 'docid'）
        :param score_score_key: score 文件里分数字段名（你的示例是 'score'）
        """
        self.cache_dir = cache_dir
        self.score_path = score_path
        self.score_threshold = score_threshold
        self.score_id_key = score_id_key
        self.score_score_key = score_score_key

    def _load_high_score_docids(self):
        """
        从 score 文件中加载 score >= threshold 的 docid 集合。
        如果没有配置 score_path，则返回 None（表示不做这一步过滤）。
        """
        if not getattr(self, "score_path", None):
            print("[INFO] 没有提供 score_path，跳过 score 过滤步骤。")
            return None

        print(f"[INFO] 加载 score 文件: {self.score_path}")
        print(
            f"[INFO] 使用的 id 字段: {self.score_id_key}，"
            f"score 字段: {self.score_score_key}，"
            f"阈值: {self.score_threshold}"
        )

        # zh-scores.jsonl 是 JSONL 文本，用 from_json 读取
        scores_ds = Dataset.from_json(self.score_path)
        print(f"[INFO] score 文件行数: {len(scores_ds)}")

        high_score_docids = set()
        total = 0
        kept = 0

        for d in scores_ds:
            total += 1

            # 1) 先按配置字段取；2) 兜底用 docid / id
            docid = d.get(self.score_id_key) or d.get("docid") or d.get("id")
            score_val = d.get(self.score_score_key)

            if docid is None or score_val is None:
                continue

            try:
                score_val = float(score_val)
            except (TypeError, ValueError):
                continue

            if score_val >= self.score_threshold:
                high_score_docids.add(str(docid))
                kept += 1

        print(
            f"[INFO] 在 score 文件中，共 {total} 条，"
            f"score >= {self.score_threshold} 的 docid 数量: {kept}"
        )
        return high_score_docids

    def _load_corpus(self, corpus_path: str, qrels_path: str):
        # 1. 加载 corpus
        corpus = Dataset.from_file(corpus_path)
        print(f"[INFO] Loaded corpus size: {len(corpus)}")

        # 2. 加载 qrels
        qrels_ds = Dataset.from_file(qrels_path)
        print(f"[INFO] Loaded qrels size: {len(qrels_ds)}")

        # 2.1 先根据 qrels 过滤“相关文档”（score > 0 的 pid）
        skip_docids = set()
        for d in qrels_ds:
            score = int(d["score"])
            if score > 0:
                skip_docids.add(d["pid"])
        print(f"[INFO] 需要跳过（相关文档）的 docid 数量: {len(skip_docids)}")

        # 2.2 再加载 score 文件中的高分 docid（用于“之后”的二次过滤）
        high_score_docids = self._load_high_score_docids()
        if high_score_docids is not None:
            print(f"[INFO] 参与二次过滤的高分 docid 数量: {len(high_score_docids)}")

        corpus_list = []
        min_len = 200  # 长度过滤阈值

        # 3. 依次遍历 corpus，按顺序做三步过滤
        for data in tqdm(corpus, desc="Loading corpus"):
            cid = data["id"]
            text = data["text"]

            # (1) 先过滤掉“相关文档”（qrels 中 score > 0 的 pid）
            if cid in skip_docids:
                continue

            # (2) 再过滤长度
            if len(text) < min_len:
                continue

            # (3) 最后根据 score 文件进行二次过滤（score >= threshold）
            if high_score_docids is not None:
                # 注意统一用 str 对齐
                if str(cid) not in high_score_docids:
                    continue

            # 3 步都通过，才保留
            corpus_list.append({"text": text})

        # --- print final retained corpus size ---
        print(f"[INFO] Final filtered corpus size: {len(corpus_list)}")

        return corpus_list

    def run(
        self,
        language: str,
        num_samples: int = -1,
    ):
        corpus_path = os.path.join(COVID_ROOT, "covid_retrieval-corpus.arrow")
        qrels_path = os.path.join(
            COVID_ROOT_qerels, "covid_retrieval-qrels-dev.arrow"
        )

        corpus_list = self._load_corpus(corpus_path, qrels_path)

        if num_samples > 0 and num_samples < len(corpus_list):
            corpus_list = random.sample(corpus_list, num_samples)

        return corpus_list
