"""
如果这个数据集已经有训练集了，就从里面采样 10 条出来形成 {"query": str, "pos": list[str]} 这种数据：.jsonl 文件；
否则，利用这个数据集中的测试数据，从里面采样 10 条出来形成 {"query": str, "pos": list[str]} 这种数据：.jsonl 文件；
"""
import os
import json
import random


AILAStatutes_TRAIN_ROOT = "/share/project/shared_models/datasets--mteb--AILA_statutes/snapshots/ac23c06b6894334dd025491c6abc96ef516aad2b"
NUM_SAMPLES = 20


def load_ailastatutes_train(language: str):
    #file_path = os.path.join(AILAStatutes_TRAIN_ROOT, f"ailastatutes_{language}.jsonl")
    corpus_path = os.path.join(AILAStatutes_TRAIN_ROOT, "corpus.jsonl")
    query_path = os.path.join(AILAStatutes_TRAIN_ROOT, "queries.jsonl")
    qrels_path = os.path.join(AILAStatutes_TRAIN_ROOT, "qrels/test.jsonl")

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
    print("Starting to generate examples...")
    save_dir = "/share/project/psjin/dataset/ailastatutes/generation_results/examples"
    os.makedirs(save_dir, exist_ok=True)
    
    for language in language_list:
        data_list = load_ailastatutes_train(language)

        save_path = os.path.join(save_dir, f"{language}_sample_examples.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(data_list)} examples for language {language} to {save_path}")
    print("All examples saved successfully!")


if __name__ == "__main__":
    main()
