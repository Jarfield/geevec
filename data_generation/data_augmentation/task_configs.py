from __future__ import annotations

"""Centralised task configuration for data augmentation pipelines.

This module keeps dataset paths, filtering rules, and example locations in one
place so that callers no longer need to edit individual scripts when switching
between tasks. Update the default paths to match your local layout or override
values via CLI arguments in ``run_generation.py`` and ``run_corpus_generation.py``.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TaskDatasetConfig:
    """Dataset settings for query generation.

    Attributes:
        corpus_path: Path to the corpus file to sample positives from.
        qrels_path: Optional qrels file used to keep only positive documents.
        min_len: Minimum length (characters) to keep a document.
        id_key: Field name for document id in the corpus file.
        text_key: Field name for document text in the corpus file.
        qrels_pid_key: Field name for passage id inside the qrels file.
        qrels_score_key: Field name for the relevance score inside the qrels file.
        examples_dir: Optional directory that stores few-shot examples.
    """

    corpus_path: Optional[str]
    qrels_path: Optional[str] = None
    min_len: int = 200
    id_key: str = "id"
    text_key: str = "text"
    qrels_pid_key: str = "pid"
    qrels_score_key: str = "score"
    examples_dir: Optional[str] = None


# You can change this root once to redirect all default paths.
DEFAULT_DATA_ROOT = os.environ.get("DATA_AUG_ROOT", "/data/share/project/shared_datasets")
DEFAULT_GENERATED_ROOT = os.environ.get(
    "DATA_AUG_GENERATED_ROOT",
    "/data/share/project/psjin/data/generated_data",
)


TASK_DATASETS: Dict[str, TaskDatasetConfig] = {
    "covidretrieval": TaskDatasetConfig(
        corpus_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "MMTEB/C-MTEB___covid_retrieval/default/0.0.0/1271c7809071a13532e05f25fb53511ffce77117/covid_retrieval-corpus.arrow",
        ),
        qrels_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "MMTEB/C-MTEB___covid_retrieval-qrels/default/0.0.0/a9f41b7cdf24785531d12417ce0d1157ed4b39ca/covid_retrieval-qrels-dev.arrow",
        ),
        min_len=200,
        text_key="text",
        id_key="id",
        examples_dir=os.path.join(DEFAULT_DATA_ROOT, "data_generation/examples"),
    ),
    # Leave corpus/qrels as None for tasks where the source varies; provide them
    # via CLI or by editing this config once.
    "scidocs": TaskDatasetConfig(
        corpus_path=None,
        qrels_path=None,
        min_len=120,
        text_key="text",
        id_key="id",
    ),
    "arguana": TaskDatasetConfig(
        corpus_path=None,
        qrels_path=None,
        min_len=80,
        text_key="text",
        id_key="id",
    ),
    "ailastatutes": TaskDatasetConfig(
        corpus_path=os.path.join(
            DEFAULT_GENERATED_ROOT,
            "ailastatutes/generation_results/generated_corpus/en_synth_corpus.jsonl",
        ),
        qrels_path=None,
        min_len=200,
        text_key="text",
        id_key="_id",
    ),
}


def get_task_config(task_type: str) -> TaskDatasetConfig:
    try:
        return TASK_DATASETS[task_type]
    except KeyError:
        raise ValueError(
            f"Unknown task_type '{task_type}'. Please update TASK_DATASETS in task_configs.py."
        )
