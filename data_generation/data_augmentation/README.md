# General data augmentation toolkit

This folder is a task-agnostic copy of the CovidRetrieval data_augmentation code
with extensions for SCIDOCS, AILA, and ArguAna. The goal is to control every
path and prompt through a small set of scripts/configs instead of editing
multiple files per task.

## Key files
- `task_configs.py`: single place to point each task to its corpus/qrels and
  optional few-shot example roots. Override the environment variables
  `DATA_AUG_ROOT` and `DATA_AUG_GENERATED_ROOT` or edit the constants once to
  redirect all default paths.
- `constant.py`: shared task definitions and prompt templates for query
  generation and quality control.
- `attributes_config.py`: attribute samplers and prompt snippets for each task
  (Covid, AILA, SCIDOCS, ArguAna). Extend or modify the samplers to tune the
  style of synthetic corpora.
- `corpus_generator.py`: task-aware loader that filters the source corpus (and
  qrels if provided) before query generation or corpus synthesis.
- `triplet_generator.py`: uses the prompts in `constant.py` to turn seed
  documents into (query, pos) training triples.
- `doc_synthesis_generator.py`: uses the attribute samplers to rewrite or extend
  the corpus from seed documents.
- `run_generation.py`: entry point for generating query–document triplets.
- `run_corpus_generation.py`: entry point for synthesising new corpus documents
  from seeds.

## Typical workflow
1. **Configure datasets** once in `task_configs.py` (or pass `--corpus_path` / `--qrels_path` at runtime).
   - CovidRetrieval comes with prefilled arrow paths; other tasks default to
     `None` so you can plug in your own files.
2. **Generate triplets** for a task:
   ```bash
   python run_generation.py \
     --task_type covidretrieval \
     --language zh \
     --save_dir /path/to/save \
     --corpus_path /custom/corpus.arrow \  # optional override
     --qrels_path /custom/qrels.arrow    # optional override
   ```
   - Few-shot examples are discovered via `task_configs.py` unless you set
     `--examples_dir` explicitly.
   - Multi-round generation (`--num_rounds`) will auto-cycle narrative focuses
     for CovidRetrieval and AILA.
3. **Synthesize corpus documents** (optional pre-step for query generation):
   ```bash
   python run_corpus_generation.py \
     --task_type arguana \
     --language en \
     --corpus_path /path/to/arguana.jsonl \
     --save_path /path/to/generated/arguana_synth.jsonl
   ```
   - The synthesiser samples task-specific attributes from
     `attributes_config.py` to diversify outputs.

## How things fit together
- `run_generation.py` loads a corpus via `CorpusGenerator`, optionally filters
  with qrels, and fans out `TripletGenerator` workers to create training
  triplets. Outputs are written to `<save_dir>/<language>/<task>/...`.
- `run_corpus_generation.py` reuses the same corpus loader to get seed
  documents, then `DocSynthesisGenerator` rewrites each seed with task-aware
  attributes and format hints before saving a synthetic corpus JSONL file.
- Both scripts are controlled by CLI flags so you can swap tasks, paths, or
  model endpoints without touching internal modules.

## Extending to new tasks
- Add a `TaskType` entry in `constant.py` and update the generation/QC prompt
  dictionaries.
- Register corpus defaults in `task_configs.py`.
- Provide an attribute sampler + hint renderer in `attributes_config.py` so the
  synthesiser can steer style and content.
