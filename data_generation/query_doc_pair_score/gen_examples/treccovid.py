"""
如果这个数据集已经有训练集了，就从里面采样 10 条出来形成 {"query": str, "pos": list[str]} 这种数据：.jsonl 文件；
否则，利用这个数据集中的测试数据，从里面采样 10 条出来形成 {"query": str, "pos": list[str]} 这种数据：.jsonl 文件；
"""
import os
import json
import random


MIRACL_TRAIN_ROOT = "/data/share/project/shared_models/datasets--mteb--trec-covid/snapshots/44c0d5a1a986eedaaf740f2aa20922584f0ed045"
NUM_SAMPLES = 20


def load_miracl_train(language: str):
    # file_path = os.path.join(MIRACL_TRAIN_ROOT, f"queries.jsonl")
    
    corpus_path = os.path.join(MIRACL_TRAIN_ROOT, "corpus.jsonl")
    query_path = os.path.join(MIRACL_TRAIN_ROOT, "queries.jsonl")
    qrels_path = os.path.join(MIRACL_TRAIN_ROOT, "qrels/test.jsonl")
    
    
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"File {corpus_path} not found.")
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"File {query_path} not found.")
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"File {qrels_path} not found.")
    
    
    corpus = {json.loads(l)["_id"]: json.loads(l) for l in open(corpus_path, "r", encoding="utf-8")}
    queries = {json.loads(l)["_id"]: json.loads(l)["text"] for l in open(query_path, "r", encoding="utf-8")}
    
    data_list = []
    
    for line in open(qrels_path, "r", encoding="utf-8"):
        item = json.loads(line)
        if int(item["score"]) == 2:
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
    print(f"Total {len(data_list)} examples found.")
    data_list = random.sample(data_list, NUM_SAMPLES)
    return data_list


def main():
    language_list = ["en"]
    
    save_dir = "/data/share/project/tr/mmteb/code/datasets/teccovid_generation_results/examples_new"
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
