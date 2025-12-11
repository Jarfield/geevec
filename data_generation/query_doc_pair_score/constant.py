from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict


class TaskType(Enum):
    # miracl = "Given a question, retrieve Wikipedia passages that answer the question."
    treccovid = "Given a query on COVID-19, retrieve documents that answer the query."
    belebeleretrieval  = "Retrieval the relevant passage for the given query."
    winogrande = "Given a commonsense reasoning question, retrieve the correct answer option."


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
    data_triplets: dict,
    examples: Optional[List[dict]] = None,
) -> str:
    task_to_gen_instruction: Dict[TaskType, str] = {
        # TaskType.miracl: "Given a Wikipedia passage in {language}, generate a question in {language} that the passage can answer.",
        TaskType.treccovid: "Given a document in {language}, generate a query on COVID-19 in {language} that the document can answer.",
        TaskType.belebeleretrieval: "Given a passage in {language}, generate a query in {language} that the passage can answer.",
        TaskType.winogrande: "Given an answer option in {language}, generate a commonsense reasoning question in {language} that this option can answer correctly.",
    }
    
    task_to_gen_output: Dict[TaskType, str] = {
        # TaskType.miracl: "the generated question in {language}",
        TaskType.treccovid: "the generated query on COVID-19 in {language}",
        TaskType.belebeleretrieval: "the generated query in {language}",
        TaskType.winogrande: "the generated commonsense reasoning question in {language}",
    }
    
    gen_instruction = task_to_gen_instruction[task.task_type]
    gen_output = task_to_gen_output[task.task_type]
    
    gen_instruction = gen_instruction.replace("{language}", task.language.value)
    gen_output = gen_output.replace("{language}", task.language.value)
    
    prefix = "The given content:"
    Query_Type = f"about COVID-19"
    Doc_Type = f"about the Medical topic"
    Query = data_triplets["query"]
    Doc = data_triplets["neg"][0] if len(data_triplets["neg"]) > 0 else text
    
    gen_prompt = f"""\
Now given a query ({Query_Type}) and a document ({Doc_Type}) in this retrieval task, your mission is to perform the
following steps to determine the relevance between the query
and the document.
1. Query Analysis: Think to reason and describe what
information would be most helpful in answering the query.
2. Document Analysis: Discuss how the information provided by the document fulfills or fails to fulfill the requirements implied by the query.
3. Relevance Annotation: Based on the relevance definition
and the insights from the previous two steps, clearly justify
your final relevance annotation result and annotate an integer
score from a scale of 1 to 5. Please use the following guide:
- 5 (Highly Relevant): The document is directly and fully
responsive to the query, providing comprehensive, accurate,
and specific information that completely addresses all aspects of the query.
- 4 (Relevant): The document is largely relevant and
provides most of the information needed, but may have
minor omissions, slight inaccuracies, or not be perfectly
aligned with the query’s intent.
- 3 (Moderately Relevant): The document has some relevance and offers partial information, but it may be incomplete, vague, or include some irrelevant content. It provides
a basic connection but lacks depth or precision.
- 2 (Slightly Relevant): The document has minimal relevance, with only a small portion of content tangentially
related to the query. The majority of the document is offtopic or provides little value.
- 1 (Irrelevant): The document is completely unrelated
to the query and provides no useful information. There is no
discernible connection or value for answering the query.
After providing your detailed analysis and justification for
all the steps above, conclude your entire response with the
final relevance score. The score must be placed strictly
between the <score> tags. There should be no other text or
explanation inside the tags:
<score>
[From a scale of 1 to 5, annotate the degree of relevance
between the query and the document.]
</score>
Note: The whole response should be as concise as possible
while covering all the necessary details, and not exceeding
512 words in total.
Query ({Query_Type}):
[Begin of Query]
{Query}
[End of Query]
Document ({Doc_Type}):
[Begin of Document]
{Doc}
[End of Document]

"""

#     if examples is not None:
#         examples_str_list = [f"""\
# - Example {i + 1}:
#     {prefix}
#     [Begin of Content]
#     {example['input']}
#     [End of Content]
    
#     Expected Output ({gen_output}): {example['output']}

# """ for i, example in enumerate(examples)]
        
#         gen_prompt += f"""\
# Here are a few examples for your reference:
# {''.join(examples_str_list)}
# """

#     gen_prompt += "Remember do not explain your output or output anything else. Your output:"
    
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
