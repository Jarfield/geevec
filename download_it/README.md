# Download helpers

两个轻量脚本，方便在已经配置好环境变量的前提下拉取 Hugging Face 模型和数据集。

## 1. 模型下载

```bash
cd download_it
bash download_model/script/download_model.sh --repo-id Qwen/Qwen2.5-3B --revision main
```

参数要点：

- `--repo-id`：必填，模型仓库 ID。
- `--revision`：可选，分支/Tag/Commit。
- `--local-dir`：可选，自定义落盘目录（默认写入 `/data/share/project/shared_models/<repo>`）。
- `--allow-patterns/--ignore-patterns`：可选，控制要下/不要下的文件。
- `--token`：可选，默认读取 `HF_TOKEN`。

## 2. 数据集下载

```bash
cd download_it
bash download_data/script/download_dataset.sh --dataset wikitext --subset wikitext-2-raw-v1 --splits train validation test
```

参数要点：

- `--dataset`：必填，数据集名称。
- `--subset`：可选，子集/配置名。
- `--splits`：可选，指定要下载的 split（默认 `train`）。
- `--cache-dir`：可选，默认读取 `HF_DATASETS_CACHE`，若未设置则写入 `/data/share/project/shared_datasets`。
- `--token`：可选，默认读取 `HF_TOKEN`。

## 3. 环境与镜像

脚本会自动读取你已经设置好的环境变量（如 `HF_ENDPOINT`、`HF_HUB_ENABLE_HF_TRANSFER`、`HUGGINGFACE_HUB_CACHE`、`HF_DATASETS_CACHE` 等），无需重复配置。只需保证依赖安装完成：

```bash
pip install huggingface_hub datasets hf_transfer
```

> 已经在系统中创建好的缓存目录会被直接复用，模型默认落到 `/data/share/project/shared_models`，数据集默认落到 `/data/share/project/shared_datasets`，缓存位置则遵循环境变量设置。
