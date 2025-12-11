"""
Stage A: Build SCIDOCS-like training triples from the allenai/scirepeval dataset.

The script keeps the JSONL layout used by data_augmentation (query + pos/neg text lists)
while attaching doc_id metadata for downstream SciNCL sampling.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from datasets import Dataset, load_dataset
from tqdm import tqdm


@dataclass
class DocRecord:
    doc_id: str
    text: str
    fos: Optional[Sequence[str]] = None
    year: Optional[int] = None


def _first_non_empty(row: dict, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        val = row.get(key)
        if val:
            return val
    return None


def _extract_doc_id(row: dict) -> Optional[str]:
    return _first_non_empty(row, ("corpus_id", "paper_id", "id", "mag_id", "pid"))


def _extract_text(row: dict, *, title_keys: Sequence[str], text_keys: Sequence[str]) -> Optional[str]:
    title = _first_non_empty(row, title_keys)
    body = _first_non_empty(row, text_keys)
    if not body:
        return None
    if title:
        return f"{title}\n{body}".strip()
    return str(body).strip()


def _load_doc_metadata(min_len: int = 0) -> Dict[str, DocRecord]:
    """Load SciDocs MAG/Mesh metadata into a doc_id -> DocRecord lookup."""

    try:
        scidocs_ds = load_dataset("allenai/scirepeval", "scidocs_mag_mesh", split="train")
    except Exception as err:  # pragma: no cover - safety for offline environments
        raise RuntimeError(
            "Failed to load 'scidocs_mag_mesh' split from allenai/scirepeval. "
            "Please check that the datasets package and network are available."
        ) from err

    doc_store: Dict[str, DocRecord] = {}
    for row in tqdm(scidocs_ds, desc="Loading SciDocs metadata"):
        doc_id = _extract_doc_id(row)
        if not doc_id:
            continue

        text = _extract_text(
            row,
            title_keys=("title",),
            text_keys=("abstract", "text", "body"),
        )
        if text and len(text) < min_len:
            text = None

        fos = row.get("fos") or row.get("field_of_study")
        year = row.get("pub_year") or row.get("year")
        year = int(year) if year not in (None, "") else None

        doc_store[doc_id] = DocRecord(doc_id=doc_id, text=text or "", fos=fos, year=year)

    # Optional enrichment: merge FOS and publication year from dedicated configs when available.
    for cfg, attr in (("fos", "fos"), ("pub_year", "year")):
        try:
            extra_ds = load_dataset("allenai/scirepeval", cfg, split="train")
        except Exception:
            continue
        for row in tqdm(extra_ds, desc=f"Merging {cfg}"):
            doc_id = _extract_doc_id(row)
            if not doc_id or doc_id not in doc_store:
                continue
            if attr == "fos":
                doc_store[doc_id].fos = row.get("fos") or row.get("field_of_study")
            elif attr == "year":
                yr = row.get("pub_year") or row.get("year")
                doc_store[doc_id].year = int(yr) if yr not in (None, "") else doc_store[doc_id].year

    return doc_store


def _load_scidocs_eval_ids() -> Set[str]:
    """Collect doc_ids that belong to SciDocs evaluation to avoid leakage."""
    eval_ids: Set[str] = set()
    # The "scidocs" split carries the evaluation pairs used by MTEB-SCIDOCS.
    try:
        ds = load_dataset("allenai/scirepeval", "scidocs", split="train")
    except Exception:
        return eval_ids

    for row in tqdm(ds, desc="Collecting ids from scidocs"):
        for key in ("corpus_id", "paper_id", "source_id", "target_id", "pid"):
            val = row.get(key)
            if val:
                eval_ids.add(str(val))
    return eval_ids


def _normalize_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCIDOCS-like citation triples from SciRepEval")
    parser.add_argument("--save_root", default=os.environ.get("DATA_AUG_ORIGINAL_ROOT", "./original_data"))
    parser.add_argument(
        "--output_path",
        default=None,
        help="Full path to save JSONL. Defaults to <save_root>/scidocs/scirep_citation_train/en_scirep.jsonl",
    )
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of triples to sample; -1 keeps all")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed")
    parser.add_argument("--min_year", type=int, default=2015, help="Minimum publication year to keep")
    parser.add_argument(
        "--fos",
        default="Computer Science",
        help="Required field-of-study tag (case-insensitive substring match); set empty to disable",
    )
    parser.add_argument(
        "--min_text_len", type=int, default=30, help="Drop documents whose text is shorter than this many chars"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    return Path(args.save_root) / "scidocs" / "scirep_citation_train" / "en_scirep.jsonl"


def _passes_filters(doc: DocRecord, *, fos_filter: Optional[str], min_year: Optional[int]) -> bool:
    if fos_filter:
        fos_values = [v.lower() for v in doc.fos or []]
        if fos_values and all(fos_filter.lower() not in v for v in fos_values):
            return False
    if min_year and doc.year and doc.year < min_year:
        return False
    if not doc.text:
        return False
    return True


def build_samples(
    *,
    citation_ds: Dataset,
    doc_store: Dict[str, DocRecord],
    scidocs_eval_ids: Set[str],
    num_samples: int,
    seed: int,
    min_year: Optional[int],
    fos_filter: Optional[str],
    min_text_len: int,
) -> List[dict]:
    rng = random.Random(seed)
    samples: List[dict] = []

    all_rows = list(citation_ds)
    rng.shuffle(all_rows)

    for row in tqdm(all_rows, desc="Filtering citation triples"):
        citing_id = _first_non_empty(row, ("citing_id", "source_id", "source", "qid", "query_id"))
        cited_id = _first_non_empty(row, ("cited_id", "target_id", "positive_id", "pid", "pos_id"))
        neg_ids = _normalize_list(
            _first_non_empty(row, ("neg_id", "neg_ids", "negative_ids", "hard_negative_ids", "neg"))
        )

        if not citing_id or not cited_id:
            continue
        if citing_id in scidocs_eval_ids or cited_id in scidocs_eval_ids:
            continue

        citing_doc = doc_store.get(str(citing_id))
        cited_doc = doc_store.get(str(cited_id))

        # Fallback: build missing docs directly from the citation row if metadata is absent.
        if not citing_doc:
            text = _extract_text(
                row,
                title_keys=("citing_title", "source_title", "query_title"),
                text_keys=("citing_abstract", "source_abstract", "query", "query_text"),
            )
            if text and len(text) >= min_text_len:
                citing_doc = DocRecord(doc_id=str(citing_id), text=text)
                doc_store[str(citing_id)] = citing_doc
        if not cited_doc:
            text = _extract_text(
                row,
                title_keys=("cited_title", "target_title"),
                text_keys=("cited_abstract", "target_abstract", "positive"),
            )
            if text and len(text) >= min_text_len:
                cited_doc = DocRecord(doc_id=str(cited_id), text=text)
                doc_store[str(cited_id)] = cited_doc
        if not citing_doc or not cited_doc:
            continue

        if not _passes_filters(citing_doc, fos_filter=fos_filter, min_year=min_year):
            continue
        if not _passes_filters(cited_doc, fos_filter=fos_filter, min_year=min_year):
            continue

        neg_doc = None
        neg_id_chosen = None
        for nid in neg_ids:
            if nid in scidocs_eval_ids:
                continue
            candidate = doc_store.get(str(nid))
            if candidate and _passes_filters(candidate, fos_filter=fos_filter, min_year=min_year):
                neg_doc = candidate
                neg_id_chosen = str(nid)
                break
        if not neg_doc:
            continue

        query_text = _extract_text(
            row,
            title_keys=("citing_title", "source_title", "query_title"),
            text_keys=("citing_abstract", "source_abstract", "query", "query_text"),
        )
        query_text = query_text or citing_doc.text

        pos_text = _extract_text(
            row,
            title_keys=("cited_title", "target_title"),
            text_keys=("cited_abstract", "target_abstract", "positive"),
        )
        pos_text = pos_text or cited_doc.text

        neg_text = neg_doc.text

        if not query_text or not pos_text or not neg_text:
            continue

        item = {
            "query": query_text,
            "pos": [pos_text],
            "neg": [neg_text],
            "source_ids": {
                "query_id": str(citing_id),
                "pos_ids": [str(cited_id)],
                "neg_ids": [neg_id_chosen],
            },
            "doc_pool": {
                str(citing_id): citing_doc.text,
                str(cited_id): pos_text,
                str(neg_id_chosen): neg_text,
            },
        }
        samples.append(item)

        if 0 < num_samples == len(samples):
            break

    return samples


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace.")

    doc_store = _load_doc_metadata(min_len=args.min_text_len)
    scidocs_eval_ids = _load_scidocs_eval_ids()

    try:
        citation_ds = load_dataset("allenai/scirepeval", "cite_prediction_new", split="train")
    except Exception as err:  # pragma: no cover - safety for offline environments
        raise RuntimeError(
            "Failed to load 'cite_prediction_new' split from allenai/scirepeval. "
            "Please verify your datasets installation and network connectivity."
        ) from err

    samples = build_samples(
        citation_ds=citation_ds,
        doc_store=doc_store,
        scidocs_eval_ids=scidocs_eval_ids,
        num_samples=args.num_samples,
        seed=args.seed,
        min_year=args.min_year,
        fos_filter=args.fos,
        min_text_len=args.min_text_len,
    )

    save_jsonl(samples, output_path)
    print(f"Saved {len(samples)} triples to {output_path}")


if __name__ == "__main__":
    main()
