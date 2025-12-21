import asyncio
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple, Union

import openai
import tiktoken
from openai import AzureOpenAI, OpenAI


class LLM:
    """Unified LLM wrapper that supports OpenAI, Azure OpenAI, and vLLM endpoints.

    The class keeps the original retry / timeout protections while adding a
    thread-pool powered path for high-concurrency workloads. Callers can either
    pass a single prompt or an iterable of prompts; an async interface is also
    provided for asyncio-based orchestration.
    """

    def __init__(
        self,
        model: str = "Qwen2-5-Coder-32B-Instruct",
        model_type: str = "open-source",
        port: int = 8000,
        max_workers: int = 32,
    ):
        self.model = model
        self.model_type = model_type
        self.client = self._create_client(model_type, port)
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _create_client(self, model_type: str, port: int):
        if model_type == "open-source":
            return OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1/",
            )
        if model_type == "azure":
            return AzureOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                azure_deployment=os.getenv("OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo"),
            )
        if model_type == "openai":
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", None),
            )
        raise ValueError("model_type must be one of ['open-source', 'azure', 'openai']")

    def split_text(self, text: str, anchor_points: Tuple[float, float] = (0.4, 0.7)):
        token_ids = self.tokenizer.encode(text)
        anchor_point = random.uniform(anchor_points[0], anchor_points[1])
        split_index = int(len(token_ids) * anchor_point)
        return self.tokenizer.decode(token_ids[:split_index]), self.tokenizer.decode(token_ids[split_index:])

    def _chat_single(
        self,
        prompt: str,
        *,
        max_tokens: int = 8192,
        logit_bias: Optional[Dict] = None,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.6,
        repetition_penalty: float = 1.0,
        remove_thinking: bool = True,
        timeout: int = 60,
    ) -> List[Optional[str]]:
        endure_time = 0
        endure_time_limit = timeout * 2

        def create_completion(results: Dict[str, Optional[Union[List[str], str]]]):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    logit_bias=logit_bias or {},
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={"repetition_penalty": repetition_penalty},
                    timeout=timeout,
                )
                results["content"] = [x.message.content for x in completion.choices[:n]]
            except openai.BadRequestError:
                results["content"] = [None for _ in range(n)]
            except openai.APIConnectionError as e:
                results["error"] = f"APIConnectionError({e})"
            except openai.RateLimitError as e:
                results["error"] = f"RateLimitError({e})"
            except Exception as e:  # pragma: no cover - defensive path
                results["error"] = f"Error: {e}"

        while True:
            results: Dict[str, Optional[Union[List[str], str]]] = {"content": None, "error": None}
            completion_thread = threading.Thread(target=create_completion, args=(results,))
            completion_thread.start()

            start_time = time.time()
            while completion_thread.is_alive():
                elapsed_time = time.time() - start_time
                if elapsed_time > endure_time_limit:
                    print("Completion timeout exceeded. Aborting...")
                    return [None for _ in range(n)]
                time.sleep(1)

            if results["error"]:
                if endure_time >= endure_time_limit:
                    print(f'{results["error"]} - Skip this prompt.')
                    return [None for _ in range(n)]
                print(f"{results['error']} - Waiting for 5 seconds...")
                endure_time += 5
                time.sleep(5)
                continue

            content_list: List[Optional[str]] = results["content"]  # type: ignore[assignment]
            if remove_thinking:
                content_list = [x.split("</think>")[-1].strip("\n").strip() if x is not None else None for x in content_list]
            return content_list

    def chat(
        self,
        prompt: Union[str, Iterable[str]],
        *,
        max_tokens: int = 8192,
        logit_bias: Optional[Dict] = None,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.6,
        repetition_penalty: float = 1.0,
        remove_thinking: bool = True,
        timeout: int = 60,
    ) -> List[Optional[str]] | List[List[Optional[str]]]:
        """Synchronous chat API with optional multi-prompt fan-out."""

        if isinstance(prompt, str):
            return self._chat_single(
                prompt,
                max_tokens=max_tokens,
                logit_bias=logit_bias,
                n=n,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                remove_thinking=remove_thinking,
                timeout=timeout,
            )

        futures = [
            self._executor.submit(
                self._chat_single,
                p,
                max_tokens=max_tokens,
                logit_bias=logit_bias,
                n=n,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                remove_thinking=remove_thinking,
                timeout=timeout,
            )
            for p in prompt
        ]
        return [f.result() for f in futures]

    async def achat(
        self,
        prompt: Union[str, Iterable[str]],
        *,
        max_tokens: int = 8192,
        logit_bias: Optional[Dict] = None,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.6,
        repetition_penalty: float = 1.0,
        remove_thinking: bool = True,
        timeout: int = 60,
    ) -> List[Optional[str]] | List[List[Optional[str]]]:
        """Async wrapper around :meth:`chat` for asyncio pipelines."""

        loop = asyncio.get_running_loop()
        if isinstance(prompt, str):
            return await loop.run_in_executor(
                self._executor,
                lambda: self.chat(
                    prompt,
                    max_tokens=max_tokens,
                    logit_bias=logit_bias,
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    remove_thinking=remove_thinking,
                    timeout=timeout,
                ),
            )

        tasks = [
            loop.run_in_executor(
                self._executor,
                lambda p=p: self.chat(
                    p,
                    max_tokens=max_tokens,
                    logit_bias=logit_bias,
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    remove_thinking=remove_thinking,
                    timeout=timeout,
                ),
            )
            for p in prompt
        ]
        results = await asyncio.gather(*tasks)
        return results
