import mteb
import torch
import gc
from sentence_transformers import SentenceTransformer
from typing import Optional


# MultiGPUModel: 修复了较旧版本中 prompt 传递失效的问题，并支持显式显存清理
class MultiGPUModel:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        # 启动多进程池
        self.gpu_pool = self.model.start_multi_process_pool()

    def encode(self, sentences, batch_size=32, **kwargs):
        # 显式传递参数，确保 multi_process 模式下配置生效
        return self.model.encode_multi_process(
            sentences, 
            self.gpu_pool, 
            batch_size=batch_size, 
            **kwargs
        )

    def stop_pool(self):
        """评测结束或切换大版本时建议手动关闭，释放显存"""
        if hasattr(self, "gpu_pool"):
            self.model.stop_multi_process_pool(self.gpu_pool)


class SentenceTransformerEncoder:
    def __init__(
        self,
        model_name_or_path: str,
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        use_bf16: bool = True,
        normalize_embeddings: bool = True,
        max_length: int = 512,
        batch_size: int = 32, # 对于 8B 模型，32-64 是 A100/H800 的稳健选择
        prompt_template: str = "Instruct: {}\nQuery: ",
        prompts_dict: Optional[dict] = None,
        assert_prompts_exist: bool = True,
        only_download_data: bool = False,
        **kwargs
    ):
        _dtype = torch.float32
        if use_fp16:
            _dtype = torch.float16
        elif use_bf16:
            _dtype = torch.bfloat16

        # --- 优化点 1: 主进程 CPU 加载，避免 GPU 0 出现双模型 OOM ---
        # Nemotron-8b 约 16GB，若主进程占 16G，子进程又占 16G，24G/40G 卡必崩
        self.model_obj = SentenceTransformer(
            model_name_or_path, 
            device="cpu", # 关键：主进程不占 GPU 显存
            trust_remote_code=trust_remote_code, 
            model_kwargs={
                "torch_dtype": _dtype,
                "attn_implementation": "flash_attention_2"
            },
            tokenizer_kwargs={"padding_side": "left"}, # Nemotron 官方推荐左填充
            **kwargs
        )
        
        # 预设模型参数
        self.max_length = max_length
        self.model_obj.max_seq_length = max_length
        
        self.model_card_data = self.model_obj.model_card_data
        self.similarity_fn_name = self.model_obj.similarity_fn_name
        
        # --- 优化点 2: 启动 GPU 进程池 ---
        self.model = MultiGPUModel(self.model_obj)
        
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.prompt_template = prompt_template
        self.prompts_dict = prompts_dict or {}
        self.assert_prompts_exist = assert_prompts_exist
        self.only_download_data = only_download_data

    def get_prompt(
        self,
        task_name: str,
    ):
        if task_name in self.prompts_dict:
            instruction = self.prompts_dict[task_name]
        else:
            _prompt_value = mteb.get_task(task_name).metadata.prompt
            if _prompt_value:
                if isinstance(_prompt_value, str):
                    instruction = _prompt_value
                else:
                    instruction = _prompt_value["query"]
            else:
                if self.assert_prompts_exist:
                    raise ValueError(f"Prompt for task '{task_name}' not found in prompts_dict or task metadata.")
                instruction = "Given the following text, generate its embedding."
        
        prompt = (
            self.prompt_template.format(instruction)
            if self.prompt_template
            else instruction
        )
        return prompt
    
    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type = None, # "query" or "document"
        **kwargs
    ):
        if self.only_download_data:
            raise RuntimeError("Model is set to only download data.")

        # --- 优化点 4: 手动拼接 Prompt，彻底规避多进程下 prompt 参数丢失的 Bug ---
        if prompt_type == "document":
            # Nemotron 官方：Document 端严禁加任何指令或前缀
            input_sentences = sentences
        else:
            # Query 端：手动拼接指令和 Query 标记
            prompt_prefix = self.get_prompt(task_name)
            input_sentences = [prompt_prefix + s for s in sentences]
            
        # 调用 MultiGPU 编码
        return self.model.encode(
            input_sentences,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True
        )

    def close(self):
        """释放所有 GPU 显存"""
        self.model.stop_pool()
        del self.model_obj
        torch.cuda.empty_cache()
        gc.collect()