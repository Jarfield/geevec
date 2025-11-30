from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict


class TaskType(Enum):
    miracl = "Given a question, retrieve Wikipedia passages that answer the question."
    arguana = "Given a claim, find documents that refute the claim."

class Language(Enum):
    en = 'English'  # 英语
    zh = 'Simplified Chinese'  # 简体中文
    ar = 'Arabic'  # 阿拉伯语
    bn = 'Bengali'  # 孟加拉语
    es = 'Spanish'  # 西班牙语
    fa = 'Persian'  # 波斯语
    fi = 'Finnish'  # 芬兰语
    fr = 'French'  # 法语
    hi = 'Hindi'  # 印地语
    id = 'Indonesian'  # 印度尼西亚语
    ja = 'Japanese'  # 日语
    ko = 'Korean'  # 韩语
    ru = 'Russian'  # 俄语
    sw = 'Swahili'  # 斯瓦希里语
    te = 'Telugu'  # 泰卢固语
    th = 'Thai'  # 泰语
    de = 'German'  # 德语
    yo = 'Yoruba'  # 约鲁巴语


@dataclass
class Task:
    task_type: TaskType
    language: Language
    task_instruction: str = None


def get_task(
    task_type: str,
    language: str,
):
    task_instruction = TaskType[task_type].value

    task = Task(
        task_type=TaskType[task_type],
        language=Language[language],
        task_instruction=task_instruction,
    )
    return task


def get_generation_prompt(
    task: Task,
    text: str,
    examples: Optional[List[dict]] = None,
) -> str:
    task_to_gen_instruction: Dict[TaskType, str] = {
        TaskType.miracl: "Given a Wikipedia passage in {language}, generate a question in {language} that the passage can answer.",
        TaskType.arguana: "Given a document in {language}, generate a claim in {language} that the document would refute.",
    }
    
    task_to_gen_output: Dict[TaskType, str] = {
        TaskType.miracl: "the generated question in {language}",
        TaskType.arguana: "the generated claim in {language}",
    }
    
    gen_instruction = task_to_gen_instruction[task.task_type]
    gen_output = task_to_gen_output[task.task_type]
    
    gen_instruction = gen_instruction.replace("{language}", task.language.value)
    gen_output = gen_output.replace("{language}", task.language.value)
    
    prefix = "The given content:"
    
    gen_prompt = f"""\
{gen_instruction}

{prefix}
[Begin of Content]
{text}
[End of Content]

- Your output must always be a string, only containing {gen_output}.
- Your output should be independent of the given passage, which means that it should not contain the pronouns such as "it", "this", "that", "the given", "the provided", etc.

"""

    if examples is not None:
        examples_str_list = [f"""\
- Example {i + 1}:
    {prefix}
    [Begin of Content]
    {example['input']}
    [End of Content]
    
    Expected Output ({gen_output}): {example['output']}

""" for i, example in enumerate(examples)]
        
        gen_prompt += f"""\
Here are a few examples for your reference:
{''.join(examples_str_list)}
"""

    gen_prompt += "Remember do not explain your output or output anything else. Your output:"
    
    return gen_prompt


def get_quality_control_prompt(
    task: Task,
    query: str,
    pos: str,
) -> str:
    task_to_qc_mission: Dict[TaskType, str] = {
        TaskType.miracl: (
            "judge whether the Wikipedia passage can answer the question",
            "the question",
            "the Wikipedia passage",
            [
                "Yes, the Wikipedia passage can answer the question.",
                "No, the Wikipedia passage cannot answer the question.",
            ]
        )
    }
    
    qc_mission, query_type, doc_type, qc_options = task_to_qc_mission[task.task_type]
    
    pos_option = qc_options[0]
    neg_option = qc_options[1]
    
    qc_prompt = f"""\
Given a code retrieval task (Task), a query (Query), and a document (Document), your mission is to {qc_mission}.

Task: {task.task_instruction}

Query ({query_type}):
```
{query}
```

Document ({doc_type}):
```
{pos}
```

Your output must be one of the following options:
- 0: {neg_option}
- 1: {pos_option}

Do not explain your answer in the output. Your output must be a single number (0 or 1).

Your output:"""
    
    return qc_prompt
