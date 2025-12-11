import json
import os
from pathlib import Path
import random

input_file = "/data/share/project/tr/mmteb/code/datasets/teccovid_generation_results/treccovid/11-27-score-generation/en/treccovid/en-triplets.jsonl"
output_file = "/data/share/project/tr/mmteb/code/datasets/teccovid_generation_results/treccovid/11-27-score-generation/en/treccovid/en-split-triplets.jsonl"

pos_len_dict = []
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        item = json.loads(line)
        pos = item.get("pos", [])
        pos_len_dict.append(len(pos))


pos_len_dict.sort()
median_index = len(pos_len_dict) // 2
median_length = pos_len_dict[median_index]
print("media_index:", median_index)
print("max length:", pos_len_dict[-1])
print("中位数长度:", median_length)


split_data_dict = []
with open(output_file, "w", encoding="utf-8") as outfile:
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            item = json.loads(line)
            poses = item.get("pos", [])
            neg = item.get("neg", [])
            query = item.get("query", "")
            prompt = item.get("prompt", "")
            
            
            for pos in poses[:median_length]:
                    new_item = {
                        "prompt": prompt,
                        "query": query,
                        "pos": pos,
                        "neg": neg,
                    }
                    split_data_dict.append(new_item)
    
    random.seed(42)
    random.shuffle(split_data_dict)
    for item in split_data_dict:
        json_line = json.dumps(item, ensure_ascii=False)
        outfile.write(json_line + "\n")
    

print("合并完成！结果保存在:", output_file)