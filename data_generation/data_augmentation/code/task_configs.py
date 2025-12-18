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
        queries_path: Path to the query file that pairs with the qrels file.
        qrels_path: Optional qrels file used to keep only positive documents.
        min_len: Minimum length (characters) to keep a document.
        id_key: Field name for document id in the corpus file.
        text_key: Field name for document text in the corpus file.
        title_key: Optional field name for document title in the corpus file.
        query_id_key: Field name for query id in the queries file.
        query_text_key: Field name for query text in the queries file.
        qrels_qid_key: Field name for query id inside the qrels file.
        qrels_pid_key: Field name for passage id inside the qrels file.
        qrels_score_key: Field name for the relevance score inside the qrels file.
        examples_dir: Optional directory that stores few-shot examples.
    """

    corpus_path: Optional[str]
    queries_path: Optional[str] = None
    qrels_path: Optional[str] = None
    min_len: int = 200
    id_key: str = "id"
    text_key: str = "text"
    title_key: Optional[str] = None
    query_id_key: str = "id"
    query_text_key: str = "text"
    qrels_qid_key: str = "qid"
    qrels_pid_key: str = "pid"
    qrels_score_key: str = "score"
    examples_dir: Optional[str] = None


# You can change this root once to redirect all default paths.
DEFAULT_DATA_ROOT = os.environ.get("DATA_AUG_ROOT", "/data/share/project/shared_datasets/.cache")
DEFAULT_GENERATED_ROOT = os.environ.get(
    "DATA_AUG_GENERATED_ROOT",
    "/data/share/project/psjin/data/generated_data",
)
DEFAULT_ORIGINAL_ROOT = os.environ.get(
    "DATA_AUG_ORIGINAL_ROOT",
    "/data/share/project/psjin/data/exported_original",
)

ARGUANA_DATA_ROOT = os.path.join(
    DEFAULT_DATA_ROOT,
    "shared_models/datasets--mteb--arguana/snapshots/c22ab2a51041ffd869aaddef7af8d8215647e41a",
)

def default_generated_corpus_path(task_type: str, language: str) -> str:
    """Return the expected path for the synthesized corpus of a task.

    The corpus synthesis step writes to
    ``<DEFAULT_GENERATED_ROOT>/<task>/generation_results/generated_corpus/<lang>_synth_corpus.jsonl``.
    Query generation should read from the same location by default to avoid
    accidentally sampling the original corpus.
    """

    return os.path.join(
        DEFAULT_GENERATED_ROOT,
        task_type,
        "generation_results",
        "generated_corpus",
        f"{language}_synth_corpus.jsonl",
    )


TASK_DATASETS: Dict[str, TaskDatasetConfig] = {
    "covidretrieval": TaskDatasetConfig(
        corpus_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "MMTEB/C-MTEB___covid_retrieval/default/0.0.0/1271c7809071a13532e05f25fb53511ffce77117/covid_retrieval-corpus.arrow",
        ),
        queries_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "MMTEB/C-MTEB___covid_retrieval/default/0.0.0/1271c7809071a13532e05f25fb53511ffce77117/covid_retrieval-queries.arrow",
        ),
        qrels_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "MMTEB/C-MTEB___covid_retrieval-qrels/default/0.0.0/a9f41b7cdf24785531d12417ce0d1157ed4b39ca/covid_retrieval-qrels-dev.arrow",
        ),
        min_len=200,
        text_key="text",
        id_key="id",
        query_id_key="id",
        query_text_key="text",
        qrels_qid_key="qid",
        qrels_pid_key="pid",
        examples_dir=os.path.join(DEFAULT_DATA_ROOT, "data_generation/examples"),
    ),
    "scidocs": TaskDatasetConfig(
        corpus_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "mteb___scidocs/corpus/0.0.0/955fa095d8dfece60ea5b5d8a1377a6e8b6c8b93/scidocs-corpus.arrow"
        ),
        queries_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "mteb___scidocs/queries/0.0.0/955fa095d8dfece60ea5b5d8a1377a6e8b6c8b93/scidocs-queries.arrow",
        ),
        qrels_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "mteb___scidocs/default/0.0.0/955fa095d8dfece60ea5b5d8a1377a6e8b6c8b93/scidocs-test.arrow",
        ),
        min_len=200,
        text_key="text",
        title_key="title",
        id_key="_id",
        query_id_key="_id",
        query_text_key="text",
        qrels_qid_key="query-id",
        qrels_pid_key="corpus-id",
        qrels_score_key="score",
    ),
    "arguana": TaskDatasetConfig(
        corpus_path=os.path.join(ARGUANA_DATA_ROOT, "corpus.jsonl"),
        queries_path=os.path.join(ARGUANA_DATA_ROOT, "queries.jsonl"),
        qrels_path=os.path.join(ARGUANA_DATA_ROOT, "qrels/test.jsonl"),
        min_len=200,
        text_key="text",
        title_key="title",
        id_key="_id",
        query_id_key="_id",
        query_text_key="text",
        qrels_qid_key="query-id",
        qrels_pid_key="corpus-id",
        qrels_score_key="score",
    ),
    "ailastatutes": TaskDatasetConfig(
        corpus_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "MMTEB/mteb___aila_statutes/corpus/0.0.0/ac23c06b6894334dd025491c6abc96ef516aad2b/aila_statutes-corpus.arrow",
        ),
        queries_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "MMTEB/mteb___aila_statutes/queries/0.0.0/ac23c06b6894334dd025491c6abc96ef516aad2b/aila_statutes-queries.arrow",
        ),
        qrels_path=os.path.join(
            DEFAULT_DATA_ROOT,
            "MMTEB/mteb___aila_statutes/default/0.0.0/ac23c06b6894334dd025491c6abc96ef516aad2b/aila_statutes-test.arrow",
        ),
        min_len=200,
        text_key="text",
        title_key="title",
        id_key="_id",
        query_id_key="_id",
        query_text_key="text",
        qrels_qid_key="query-id",
        qrels_pid_key="corpus-id",
        qrels_score_key="score",
    ),
}


def get_task_config(task_type: str) -> TaskDatasetConfig:
    try:
        return TASK_DATASETS[task_type]
    except KeyError:
        raise ValueError(
            f"Unknown task_type '{task_type}'. Please update TASK_DATASETS in task_configs.py."
        )


__all__ = [
    "TaskDatasetConfig",
    "TASK_DATASETS",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_GENERATED_ROOT",
    "DEFAULT_ORIGINAL_ROOT",
    "default_generated_corpus_path",
    "get_task_config",
]
