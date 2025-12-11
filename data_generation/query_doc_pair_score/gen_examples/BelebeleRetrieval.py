import os
import json
import random
import pyarrow.parquet as pq
import pyarrow as pa
import datasets 

BELEBELE_ROOT = "/share/project/shared_datasets/mteb/facebook___belebele/default/0.0.0/75b399394a9803252cfec289d103de462763db7c"

NUM_SAMPLES = 20

SAVE_DIR = "/share/project/tr/mmteb/code/datasets/belebeleretrieval_generation_results/examples"
os.makedirs(SAVE_DIR, exist_ok=True)


def sample_belebele_arrow(file_path: str, num_samples: int = 10):
    dataset = datasets.Dataset.from_file(file_path)
    n_total = len(dataset)
    if n_total == 0:
        return []

    random.seed(42)
    sample_indices = random.sample(range(n_total), min(num_samples, n_total))

    samples = []
    for i in sample_indices:
        item = dataset[i]
        query = item.get("question") or item.get("input") or ""
        context = item.get("context") or item.get("passage") or ""
        if not query or not context:
            continue
        samples.append({"query": query.strip(), "pos": [context.strip()]})
    return samples


def main():
    arrow_files = sorted([
        os.path.join(BELEBELE_ROOT, f)
        for f in os.listdir(BELEBELE_ROOT)
        if f.startswith("belebele-") and f.endswith(".arrow")
    ])

    print(f"[INFO] Found {len(arrow_files)} language files in {BELEBELE_ROOT}")

    for file_path in arrow_files:
        lang_name = os.path.basename(file_path).replace(".arrow", "")
        try:
            samples = sample_belebele_arrow(file_path, NUM_SAMPLES)
            if not samples:
                print(f"[WARN] No valid samples found in {lang_name}")
                continue

            save_path = os.path.join(SAVE_DIR, f"{lang_name}_samples.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=4, ensure_ascii=False)

            print(f"[INFO] Saved {len(samples)} samples for {lang_name} → {save_path}")

        except Exception as e:
            print(f"[ERROR] Failed on {lang_name}: {e}")

    print("[DONE] All languages processed.")


if __name__ == "__main__":
    main()
