import os
import time
import openai
import random
import tiktoken
import threading
from openai import OpenAI, AzureOpenAI
from typing import Tuple, List, Optional, Dict, Any


class LLM:
    def __init__(
        self,
        model: str = "Qwen2-5-Coder-32B-Instruct",
        model_type: str = "open-source",   # ["open-source", "azure", "openai"]
        port: int = 8000,
    ):
        self.model = model
        self.model_type = model_type

        if model_type == "open-source":
            # vLLM / LM Studio 等 OpenAI-compatible 本地部署
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1/"
            )
        elif model_type == "azure":
            self.client = AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo"),
            )
        elif model_type == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", None)
            )
        else:
            raise ValueError("model_type must be one of ['open-source', 'azure', 'openai']")

        # 只用于 split_text，和模型不强绑定；你要是不用 split_text，也可以删掉
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    def split_text(self, text: str, anchor_points: Tuple[float, float] = (0.4, 0.7)):
        """可选：随机把文本按 token 长度切成前/后两段."""
        token_ids = self.tokenizer.encode(text)
        anchor_point = random.uniform(anchor_points[0], anchor_points[1])
        split_index = int(len(token_ids) * anchor_point)
        return self.tokenizer.decode(token_ids[:split_index]), self.tokenizer.decode(token_ids[split_index:])

    def chat(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        remove_thinking: bool = True,
        timeout: int = 60,
        stop: Optional[List[str]] = None,
    ) -> List[Optional[str]]:
        """
        通用聊天接口：
        - 对“只打 1~5 分”的清洗任务，建议：
          max_tokens 很小（比如 4），temperature=0, top_p=1, stop=["\n"]。
        - 返回长度为 n 的列表，每个元素是一个 string 或 None。
        """
        # 对于只输出 1 个数字的任务，默认 max_tokens 不需要 8k
        if max_tokens is None:
            max_tokens = 8

        endure_time = 0
        endure_time_limit = timeout * 2  # 最多重试/等待 2 个 timeout 周期

        def create_completion(results: Dict[str, Any]):
            try:
                extra_body: Dict[str, Any] = {}
                # repetition_penalty 通常只在本地 / vLLM 里有用，云端模型可能不支持
                if self.model_type == "open-source" and repetition_penalty != 1.0:
                    extra_body["repetition_penalty"] = repetition_penalty

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    logit_bias=logit_bias or {},
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    extra_body=extra_body or None,
                    timeout=timeout,
                )
                results["content"] = [choice.message.content for choice in completion.choices[:n]]
            except openai.BadRequestError as e:
                # 响应被内容安全过滤
                results["content"] = [None for _ in range(n)]
            except openai.APIConnectionError as e:
                results["error"] = f"APIConnectionError({e})"
            except openai.RateLimitError as e:
                results["error"] = f"RateLimitError({e})"
            except Exception as e:
                results["error"] = f"Error: {e}"

        while True:
            results: Dict[str, Any] = {"content": None, "error": None}
            completion_thread = threading.Thread(target=create_completion, args=(results,))
            completion_thread.start()

            start_time = time.time()
            while completion_thread.is_alive():
                elapsed_time = time.time() - start_time
                if elapsed_time > endure_time_limit:
                    print("Completion timeout exceeded. Aborting...")
                    return [None for _ in range(n)]
                time.sleep(1)

            # 如果请求过程中出错，做有限次数重试
            if results["error"]:
                if endure_time >= endure_time_limit:
                    print(f'{results["error"]} - Skip this prompt.')
                    return [None for _ in range(n)]
                print(f"{results['error']} - Waiting for 5 seconds...")
                endure_time += 5
                time.sleep(5)
                continue

            content_list: List[Optional[str]] = results["content"]
            if remove_thinking:
                processed = []
                for x in content_list:
                    if x is None:
                        processed.append(None)
                    else:
                        # 兼容 <think>...</think> 风格模型，取后半部分
                        processed.append(x.split("</think>")[-1].strip())
                content_list = processed

            return content_list
