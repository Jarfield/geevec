# Contamination Check Query Rewriter

This module rewrites queries for contamination checking with a vLLM endpoint. It reads items from `/data/share/project/psjin/data/generated_data/task/preparation/en_pair_filtered.jsonl`, rewrites the `query` field while preserving intent, and saves results to `/data/share/project/psjin/data/generated_data/task/contamination_check`.

## Structure
- `code/`: Python utilities, including the prompt builder and the rewrite driver.
- `script/`: Runnable scripts with default paths and parameters.
- `README.md`: This guide.

## Usage

1. Make sure a vLLM server is running (see other repositories for startup commands).
2. Run the helper script (overriding paths or model settings as needed):

```bash
bash data_generation/contamination_check/script/run_rewrite.sh
```

Environment variables you can override:
- `INPUT_PATH`: Source JSONL path (default: `/data/share/project/psjin/data/generated_data/task/preparation/en_pair_filtered.jsonl`).
- `OUTPUT_DIR`: Output directory (default: `/data/share/project/psjin/data/generated_data/task/contamination_check`).
- `MODEL_NAME`, `MODEL_TYPE`, `PORT`, `TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `LIMIT`.

The script writes `rewrite_results.jsonl` with the original sample plus `rewrite_prompt` and `rewritten_query` fields.
