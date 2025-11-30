"""
如果这个数据集已经有训练集了，就从里面采样 10 条出来形成 {"query": str, "pos": list[str]} 这种数据：.jsonl 文件；
否则，利用这个数据集中的测试数据，从里面采样 10 条出来形成 {"query": str, "pos": list[str]} 这种数据：.jsonl 文件；
"""
import os
import json
import random
from datasets import Dataset

COVID_ROOT = "shared_datasets/MMTEB/C-MTEB___covid_retrieval/default/0.0.0/1271c7809071a13532e05f25fb53511ffce77117"
COVID_ROOT_qerels = "shared_datasets/MMTEB/C-MTEB___covid_retrieval-qrels/default/0.0.0/a9f41b7cdf24785531d12417ce0d1157ed4b39ca"
NUM_SAMPLES = 20


def load_covidretrieval_train(language: str):
    #file_path = os.path.join(CovidRetrieval_TRAIN_ROOT, f"covidretrieval_{language}.jsonl")
    corpus_path = os.path.join(COVID_ROOT, "covid_retrieval-corpus.arrow")
    query_path = os.path.join(COVID_ROOT, "covid_retrieval-queries.arrow")
    qrels_path = os.path.join(COVID_ROOT_qerels, "covid_retrieval-qrels-dev.arrow")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"File {corpus_path} not found.")
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"File {query_path} not found.")
    if not os.path.exists(qrels_path):
        raise FileNotFoundError(f"File {qrels_path} not found.")
    
    # 加载 Arrow 数据
    corpus_ds = Dataset.from_file(corpus_path)
    queries_ds = Dataset.from_file(query_path)
    qrels_ds = Dataset.from_file(qrels_path)

    corpus = {d["id"]: d["text"] for d in corpus_ds}
    queries = {d["id"]: d["text"] for d in queries_ds}

    data_list = []
    for item in qrels_ds:
        if int(item["score"]) > 0:
            qid = item["qid"]
            pid = item["pid"]

            if qid in queries and pid in corpus:
                query_text = queries[qid]
                doc_text = corpus[pid]

                data_list.append({
                    "query": query_text,
                    "pos": [doc_text]
                })
    
    random.seed(42)
    data_list = random.sample(data_list, NUM_SAMPLES)

    return data_list


def main():
    language_list = ["zh"]
    print("Starting to generate examples...")
    save_dir = "/share/project/psjin/dataset/examples/covidretrieval"
    os.makedirs(save_dir, exist_ok=True)
    
    for language in language_list:
        data_list = load_covidretrieval_train(language)

        save_path = os.path.join(save_dir, f"{language}_sample_examples.json")

        with open(save_path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data_list)} examples for language {language} to {save_path}")
    print("All examples saved successfully!")

def debug_print():
    corpus_path = os.path.join(COVID_ROOT, "covid_retrieval-corpus.arrow")
    query_path = os.path.join(COVID_ROOT, "covid_retrieval-queries.arrow")
    qrels_path = os.path.join(COVID_ROOT_qerels, "covid_retrieval-qrels-dev.arrow")

    corpus_ds = Dataset.from_file(corpus_path)
    queries_ds = Dataset.from_file(query_path)
    qrels_ds = Dataset.from_file(qrels_path)

    print("=== CORPUS COLUMNS ===")
    print(corpus_ds.column_names)
    print("First item:", corpus_ds[0])

    print("\n=== QUERIES COLUMNS ===")
    print(queries_ds.column_names)
    print("First item:", queries_ds[0])

    print("\n=== QRELS COLUMNS ===")
    print(qrels_ds.column_names)
    print("First item:", qrels_ds[0])

if __name__ == "__main__":
    main()
