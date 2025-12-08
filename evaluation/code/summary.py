# 调用方法:
# python /data/share/project/psjin/code/evaluation/mmteb/code/summary.py \
#   /data/share/project/psjin/code/evaluation/mmteb/results_output_folder/geevec-qwen3-8b-v1-test-w-syn \
import mteb
import json
import os
import sys

path = sys.argv[1]
benchmark = "MTEB(Multilingual, v2)"
if len(sys.argv) > 2:
    benchmark = sys.argv[2]

results = {}

def get_tasks(names: list[str] | None, languages: list[str] | None = None, benchmark: str | None = None):
    if benchmark:
        tasks = []
        for task in mteb.get_benchmark(benchmark).tasks:
            if task.metadata.type == "Retrieval":
                print("Including task:", task.metadata.name)
                tasks.append(task)
    else:
        tasks = mteb.get_tasks(languages=languages, tasks=names)
    return tasks

tasks_list = get_tasks(names=None, languages=None, benchmark=benchmark)
names = [t.metadata.name for t in tasks_list]
tasks = {name: task for name, task in zip(names, tasks_list)}

split_tasks = {}

# -------- 判断目录结构 --------
entries = os.listdir(path)
root_json_files = [f for f in entries if f.endswith(".json")]

json_file_infos = []  # (task_name, full_path)

if root_json_files:
    # 旧结构：path 里直接放了一堆 json
    for fname in root_json_files:
        task_name = fname.split(".json")[0]
        full_path = os.path.join(path, fname)
        json_file_infos.append((task_name, full_path))
else:
    # 新结构：path 下每个子目录是一个任务，例如 AILAStatutes / SCIDOCS / ...
    for task_name in entries:
        task_dir = os.path.join(path, task_name)
        if not os.path.isdir(task_dir):
            continue
        if task_name not in names:
            # 不在当前 benchmark 的任务列表里就跳过
            continue

        # 在该任务目录下递归寻找第一个包含 "scores" 的 json
        json_path = None
        for root, _, files in os.walk(task_dir):
            for f in files:
                if not f.endswith(".json"):
                    continue
                candidate = os.path.join(root, f)
                try:
                    with open(candidate, "r") as fp:
                        data = json.load(fp)
                    if "scores" in data:
                        json_path = candidate
                        break
                except Exception:
                    continue
            if json_path is not None:
                break

        if json_path is None:
            print(f"[WARN] No result json found for task {task_name}, skip.")
            continue

        json_file_infos.append((task_name, json_path))

# -------- 统计分数 --------
for task_name, json_path in json_file_infos:
    if task_name not in names:
        # 不属于当前 benchmark 的任务，跳过
        continue

    meta = tasks[task_name].metadata
    with open(json_path, "r") as f:
        result = json.load(f)

    eval_split = list(result["scores"].keys())[0]
    score_list = result["scores"][eval_split]
    score = sum([ele["main_score"] for ele in score_list]) / len(score_list)

    results[task_name] = round(score * 100, 4)

    task_type = meta.type
    if task_type not in split_tasks:
        split_tasks[task_type] = []
    split_tasks[task_type].append(score)

# -------- 输出汇总 --------
final_scores = sum(results.values()) / len(results) if results else 0.0
missed_tasks = [name for name in names if name not in results]

print("========================")
print("missed tasks", missed_tasks)
print("final score", len(results), final_scores)

scores = []
for task_type, value_list in split_tasks.items():
    mean_score = sum(value_list) / len(value_list)
    print(task_type, len(value_list), mean_score)
    scores.append(mean_score)

if scores:
    print("Mean(Type)", sum(scores) / len(scores))
else:
    print("Mean(Type) N/A")
print("========================")

sorted_results = dict(sorted(results.items(), key=lambda item: item[0], reverse=False))
for name in sorted_results:
    print("------------------------")
    print(name, sorted_results[name])
