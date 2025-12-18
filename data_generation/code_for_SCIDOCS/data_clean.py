import json
import os
from tqdm import tqdm  # 引入进度条库

# 原始文件路径
input_file = '/data/share/project/psjin/data/generated_data/scidocs/scirep_train/en_scirep_base_filtered.jsonl'
# 输出文件路径
output_file = '/data/share/project/psjin/data/generated_data/scidocs/scirep_train/en_scirep_base_filtered_cleaned.jsonl'

# 要添加的 Prompt 内容
prompt_text = "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper."

def process_data():
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    # 1. 先计算总行数，以便 tqdm 能显示剩余时间和百分比
    print("正在计算文件总行数（以便显示进度条）...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"总行数: {total_lines}")

    print(f"正在处理数据并写入: {output_file} ...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            # 使用 tqdm 包装文件迭代器
            for i, line in tqdm(enumerate(f_in), total=total_lines, desc="Processing", unit="lines"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 打印第一行的 Keys (为了不打断进度条，只在刚开始打印一次)
                if i == 0:
                    tqdm.write("-" * 30)
                    tqdm.write(f"发现的原始 Keys: {list(data.keys())}")
                    tqdm.write("-" * 30)

                # 构建新的字典，prompt 放最前
                new_item = {
                    "prompt": prompt_text
                }
                
                # 只保留指定的字段
                target_keys = ['query', 'pos', 'neg']
                for key in target_keys:
                    if key in data:
                        new_item[key] = data[key]
                
                # 写入到输出文件
                f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')

        print(f"\n处理完成！结果已保存至: {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    process_data()