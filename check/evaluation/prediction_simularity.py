import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载标准答案和两个模型的预测结果
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# 路径
path_best_model = "/data/share/project/psjin/result/evaluation/Qwen3-Embedding-8B/ArguAna/ArguAna_default_predictions.json"
path_poor_model = "/data/share/project/psjin/result/evaluation/geevec-qwen3-8b-v3/arguana-ori+syn/ArguAna/ArguAna_default_predictions.json"

best_model_data = load_json(path_best_model)
poor_model_data = load_json(path_poor_model)

# 计算 nDCG@K（或者是类似的加权评分）
def calculate_relative_scores(query_id, k=10):
    """
    计算标准答案 Top-K 文档与较差模型中的相对得分（计算 nDCG-like score）
    """
    # 1. 取出优秀模型对每个query的预测文档，并按得分排序
    best_model_ranking = sorted(best_model_data[query_id].items(), key=lambda x: x[1], reverse=True)[:k]
    
    # 2. 从较差模型获取排名
    poor_model_ranking = sorted(poor_model_data[query_id].items(), key=lambda x: x[1], reverse=True)[:k]

    # 3. 计算相对得分（较差模型的得分 / 优秀模型的得分）
    relative_scores = []
    for doc_id, best_score in best_model_ranking:
        # 在差模型中找到相同的文档
        poor_score = dict(poor_model_ranking).get(doc_id, 0)  # 如果找不到就是 0
        # 计算相对分数
        if best_score > 0:  # 防止除零错误
            relative_scores.append(poor_score / best_score)
        else:
            relative_scores.append(0)

    # 4. 计算类似 nDCG@K 的加权和
    dcg = sum([rel_score / np.log2(i + 2) for i, rel_score in enumerate(relative_scores)])
    idcg = sum([1 / np.log2(i + 2) for i in range(k)])  # 理想情况下每个文档都是完美相关的
    return dcg / idcg if idcg > 0 else 0
    # return np.mean(relative_scores)  # 简单返回平均相对分数

# 计算所有query的平均nDCG-like得分
def evaluate_model(data, k=10):
    nDCG_scores = []
    for query_id in data:
        score = calculate_relative_scores(query_id, k)
        nDCG_scores.append(score)
    return nDCG_scores  # 返回所有查询的nDCG得分

# 获取所有查询的nDCG-like得分
nDCG_scores = evaluate_model(best_model_data)

# 保存结果到指定路径
output_path = "/data/share/project/psjin/result/check/evaluation/prediction_similarity"
os.makedirs(output_path, exist_ok=True)

# 打点：横坐标是 query_id，纵坐标是 nDCG_score
query_ids = list(best_model_data.keys())

# 创建图表
plt.figure(figsize=(10, 6))
plt.scatter(query_ids, nDCG_scores, color='b', label='nDCG Scores', alpha=0.7)

# 添加标题和标签
plt.title("nDCG@10-like Scores for Best Model", fontsize=14)
plt.xlabel("Query ID", fontsize=12)
plt.ylabel("nDCG@10-like Score", fontsize=12)

plt.xticks([])

# 保存图表到指定路径
plt.tight_layout()
plt.savefig(os.path.join(output_path, "model_comparison_arguana_v3.png"))
plt.close()

print(f"Results saved to {output_path}")
