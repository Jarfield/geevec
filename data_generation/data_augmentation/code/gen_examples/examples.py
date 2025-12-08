"""
示例采样脚本，用于从 AILA Statutes 训练集生成少量 few-shot 样例。

默认从 ``AILAStatutes_TRAIN_ROOT`` 读取语料，采样 ``NUM_SAMPLES`` 条记录并
保存到 ``save_dir``。所有参数均可通过命令行覆盖。
"""
import argparse
import json
import os
import random


AILAStatutes_TRAIN_ROOT = os.environ.get(
    "AILA_TRAIN_ROOT",
    "/share/project/shared_models/datasets--mteb--AILA_statutes/snapshots/ac23c06b6894334dd025491c6abc96ef516aad2b",
)
NUM_SAMPLES = int(os.environ.get("AILA_EXAMPLE_SAMPLES", 20))


def load_ailastatutes_train(language: str, dataset_root: str, num_samples: int):
    #file_path = os.path.join(AILAStatutes_TRAIN_ROOT, f"ailastatutes_{language}.jsonl")
    corpus_path = os.path.join(dataset_root, "corpus.jsonl")
    query_path = os.path.join(dataset_root, "queries.jsonl")
    qrels_path = os.path.join(dataset_root, "qrels/test.jsonl")

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
    data_list = random.sample(data_list, num_samples)
    return data_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample few-shot examples for AILA Statutes")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to store sampled examples")
    parser.add_argument(
        "--train_root",
        type=str,
        default=AILAStatutes_TRAIN_ROOT,
        help="Root directory of the AILA Statutes dataset (contains corpus.jsonl, queries.jsonl, qrels/)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of examples to sample",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en",
        help="Comma-separated list of languages to sample",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    language_list = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    print("Starting to generate examples...")
    os.makedirs(args.save_dir, exist_ok=True)

    for language in language_list:
        data_list = load_ailastatutes_train(language, args.train_root, args.num_samples)

        save_path = os.path.join(args.save_dir, f"{language}_sample_examples.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(data_list)} examples for language {language} to {save_path}")
    print("All examples saved successfully!")


if __name__ == "__main__":
    main()
