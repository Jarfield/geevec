import json
import argparse
import os

def convert_format(input_path, output_path):
    count = 0
    empty_query_count = 0  # 记录空 query 的数量
    missing_pos_count = 0   # 记录缺失正例的数量
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                query_text = data.get("rewritten_query")
                pos_text = data.get("original_corpus")
                
                # 详细检查：如果 query 为空或不存在
                if not query_text or not str(query_text).strip():
                    empty_query_count += 1
                    print(f"[Warning] Line {line_idx}: rewritten_query is empty.")
                    continue
                
                # 详细检查：如果 pos 为空或不存在
                if not pos_text or not str(pos_text).strip():
                    missing_pos_count += 1
                    print(f"[Warning] Line {line_idx}: original_corpus is empty.")
                    continue

                # 格式转换
                new_data = {
                    "query": str(query_text).strip(),
                    "pos": [str(pos_text).strip()],
                    "neg": [] 
                }
                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                count += 1
                
            except Exception as e:
                print(f"[Error] Line {line_idx} failed to parse: {e}")
                
    # 打印最终统计结果
    print("-" * 30)
    print(f"数据转换统计结果:")
    print(f"✅ 成功转换样本数: {count}")
    print(f"❌ 空 Query 样本数 : {empty_query_count}")
    print(f"⚠️ 缺失正例样本数  : {missing_pos_count}")
    print(f"💾 输出文件路径    : {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    convert_format(args.input, args.output)