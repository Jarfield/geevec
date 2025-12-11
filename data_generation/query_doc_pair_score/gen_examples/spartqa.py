"""
如果这个数据集已经有训练集了，就从里面采样 10 条出来形成 {"query": str, "pos": list[str]} 这种数据：.jsonl 文件；
否则，利用这个数据集中的测试数据，从里面采样 10 条出来形成 {"query": str, "pos": list[str]} 这种数据：.jsonl 文件；
"""
import os
import json
import random
import pandas as pd
from glob import glob

SPARTQA_TRAIN_ROOT = "/share/project/shared_datasets/mteb/datasets--mteb--SpartQA/snapshots/1c858df377e57725a014a1b7321ebd79d62016b6"
NUM_SAMPLES = 20


def load_miracl_train(language: str):
    # file_path = os.path.join(MIRACL_TRAIN_ROOT, f"queries.jsonl")
    
    corpus_files = glob(os.path.join(SPARTQA_TRAIN_ROOT, "corpus/*.parquet"))
    corpus_df = pd.concat([pd.read_parquet(f) for f in corpus_files], ignore_index=True)
    corpus = {row["_id"]: row for _, row in corpus_df.iterrows()}

    # 2️⃣ 加载 queries
    query_files = glob(os.path.join(SPARTQA_TRAIN_ROOT, "queries/*.parquet"))
    query_df = pd.concat([pd.read_parquet(f) for f in query_files], ignore_index=True)
    queries = {row["_id"]: row["text"] for _, row in query_df.iterrows()}

    # 3️⃣ 加载 qrels
    qrels_files = glob(os.path.join(SPARTQA_TRAIN_ROOT, "qrels/*.parquet"))
    qrels_df = pd.concat([pd.read_parquet(f) for f in qrels_files], ignore_index=True)
    

    data_list = []
    
    for _, item in qrels_df.iterrows():
        if int(item["score"]) > 0:
            docid = item["corpus-id"]
            qid = item["query-id"]

            if docid in corpus and qid in queries:
                doc = corpus[docid]
                text = doc["title"] + "\n" + doc["text"]
                query = queries[qid]
                data_list.append({
                    "input": text,
                    "output": query,
                })
                
    
    random.seed(42)
    data_list = random.sample(data_list, NUM_SAMPLES)
    return data_list


def main():
    language_list = ["en"]
    
    save_dir = "/share/project/tr/mmteb/code/datasets/spartqa_generation_results/examples"
    os.makedirs(save_dir, exist_ok=True)
    
    for language in language_list:
        data_list = load_miracl_train(language)

        save_path = os.path.join(save_dir, f"{language}_sample_examples.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(data_list)} examples for language {language} to {save_path}")
    print("All examples saved successfully!")


if __name__ == "__main__":
    main()
