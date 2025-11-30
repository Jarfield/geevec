"""
CorpusGenerator

现在的用途：
- 不再从 MTEB AILA 原始数据 + qrels 中挑“相关条文”；
- 而是直接从我们合成好的 generated_corpus 里读 statute，
  做一次长度过滤后返回，用于后续 query 生成。

注意：
- 这里默认文件名为 {language}_synth_corpus.jsonl（比如 en_synth_corpus.jsonl）。
"""

import os
import random
import datasets
from tqdm import tqdm

# 你的合成法条存放目录
GENERATED_CORPUS_ROOT = (
    "/data/share/project/psjin/data/generated_data/ailastatutes/generation_results/generated_corpus"
)

MIN_LEN = 200  # 长度过滤阈值，按需改


def _get_generated_corpus_path(language: str) -> str:
    """
    根据语言拼出对应的合成 corpus 路径。
    例如 language='en' -> .../generated_corpus/en_synth_corpus.jsonl
    """
    file_name = f"{language}_synth_corpus.jsonl"
    return os.path.join(GENERATED_CORPUS_ROOT, file_name)


class CorpusGenerator:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir

    def _load_corpus(self, corpus_path: str):
        """从指定 jsonl 加载 corpus，并按长度过滤。"""
        assert corpus_path.endswith(".jsonl"), "Invalid file format, please use .jsonl."

        corpus = datasets.load_dataset(
            "json",
            data_files=corpus_path,
            cache_dir=self.cache_dir,
        )["train"]

        print(f"[INFO] Loaded corpus size: {len(corpus)}")

        corpus_list = []
        for data in tqdm(corpus, desc="Loading corpus (generated)"):

            # 对你自己的合成文件结构做一个宽松兼容：
            # - 如果有 'title' / 'text' 字段，就按原来规则拼；
            # - 如果只有 'text'，那就用 text；
            # - 如果只有 'title'，那就用 title。
            title = data.get("title") or ""
            text_field = data.get("text") or ""
            if not text_field and not title:
                # 保险一点：完全没内容就跳过
                continue

            text = title + ("\n" if title and text_field else "") + text_field

            if len(text) < MIN_LEN:
                continue

            # 尽量保留 _id，如果没有，就不填
            doc_id = data.get("_id")
            corpus_list.append({"_id": doc_id, "text": text})

        print(f"[INFO] Final filtered corpus size: {len(corpus_list)}")
        return corpus_list

    def run(
        self,
        language: str,
        num_samples: int = -1,
    ):
        """
        现在 run 只做一件事：
        - 从 generated_corpus/{language}_synth_corpus.jsonl 里加载所有条文；
        - 按长度过滤；
        - 如有需要，再做一次采样。

        返回：
        - positives: List[dict]，每个元素至少有 {"text": "..."}。
        """
        corpus_path = _get_generated_corpus_path(language)
        print(f"[INFO] Using generated corpus from: {corpus_path}")

        corpus_list = self._load_corpus(corpus_path)

        if num_samples > 0 and num_samples < len(corpus_list):
            corpus_list = random.sample(corpus_list, num_samples)

        return corpus_list
