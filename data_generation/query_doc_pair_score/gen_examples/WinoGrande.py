import os
import json
import random
import datasets


WINOGRANDE_DATASET_ID = "mteb/WinoGrande"
NUM_SAMPLES = 20


def load_winogrande_train(language: str):
    try:
        dataset = datasets.load_dataset(WINOGRANDE_DATASET_ID)
        test_data = dataset['test']
        
        queries = test_data['queries']
        corpus = test_data['corpus']
        qrels = test_data['qrels']
        
        corpus_dict = {item['id']: item['text'] for item in corpus}
        
        data_list = []
        for qrel in qrels:
            if qrel.get('score', 0) > 0:
                query_id = qrel['query-id']
                corpus_id = qrel['corpus-id']
                
                query_text = None
                for q in queries:
                    if q['id'] == query_id:
                        query_text = q['text']
                        break
                
                if query_text and corpus_id in corpus_dict:
                    data_list.append({
                        "input": corpus_dict[corpus_id],  
                        "output": query_text,           
                    })
        
        random.seed(42)
        if len(data_list) > NUM_SAMPLES:
            data_list = random.sample(data_list, NUM_SAMPLES)
        
        return data_list
        
    except Exception as e:
        print(f"Error loading WinoGrande dataset: {e}")
        return []


def main():
    language_list = ["en"]
    
    save_dir = "/share/project/tr/mmteb/code/datasets/winogrande_generation_results/examples"
    os.makedirs(save_dir, exist_ok=True)
    
    for language in language_list:
        data_list = load_winogrande_train(language)

        save_path = os.path.join(save_dir, f"{language}_sample_examples.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(data_list)} examples for language {language} to {save_path}")
    print("All examples saved successfully!")


if __name__ == "__main__":
    main()