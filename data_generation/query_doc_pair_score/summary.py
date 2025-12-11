import json
import os
from pathlib import Path

input_dir = "/data/share/project/tr/mmteb/code/datasets/teccovid_generation_results/treccovid/11-30-score-generation/en/treccovid/gen_cache_dir"
output_file = "/data/share/project/tr/mmteb/code/datasets/teccovid_generation_results/treccovid/11-30-score-generation/en/treccovid/en-triplets.jsonl"

with open(output_file, "w", encoding="utf-8") as outfile:
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            file_path = Path(input_dir) / file_name
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # 每个文件是一个数组，只取里面的对象
                for item in data:
                    json_line = json.dumps(item, ensure_ascii=False)
                    outfile.write(json_line + "\n")

print("合并完成！结果保存在:", output_file)