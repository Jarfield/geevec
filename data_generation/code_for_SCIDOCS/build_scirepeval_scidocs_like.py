"""
Stage A: Build SCIDOCS-like training triples from the allenai/scirepeval dataset.

Output JSONL layout:
- query: citing paper title
- pos: list of cited paper abstracts
- neg: list of negative paper abstracts
- source_ids/doc_pool: attach doc_id metadata for downstream sampling
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from datasets import Dataset, load_dataset, load_dataset_builder
from tqdm import tqdm


@dataclass
class DocRecord:
    doc_id: str
    text: str  # for docs: abstract; for query: title
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


def _safe_int(x: Any) -> Optional[int]:
    if x in (None, ""):
        return None
    try:
        return int(x)
    except Exception:
        return None


def _load_scirepeval(cfg: str, preferred: str = "train") -> Dataset:
    """Load scirepeval config with an available split (e.g., scidocs_mag_mesh only has evaluation)."""
    splits = list(load_dataset_builder("allenai/scirepeval", cfg).info.splits.keys())
    if not splits:
        raise RuntimeError(f"No splits found for config={cfg}")
    split = preferred if preferred in splits else splits[0]
    return load_dataset("allenai/scirepeval", cfg, split=split)


def _norm(x: Optional[str]) -> str:
    return (x or "").strip()


def _get_title_abs(struct_obj: Any) -> tuple[str, str, Optional[str]]:
    """
    cite_prediction_new row fields are structs:
      {"title": ..., "abstract": ..., "corpus_id": ...}
    """
    if not isinstance(struct_obj, dict):
        return "", "", None
    return _norm(struct_obj.get("title")), _norm(struct_obj.get("abstract")), (
        str(struct_obj.get("corpus_id")) if struct_obj.get("corpus_id") is not None else None
    )


def _load_doc_metadata(min_len: int = 0) -> Dict[str, DocRecord]:
    """Load SciDocs MAG/Mesh metadata into a doc_id -> DocRecord lookup.
    NOTE: Here we store abstract-only in DocRecord.text to match your training target (doc=abstract).
    """
    try:
        scidocs_ds = _load_scirepeval("scidocs_mag_mesh", preferred="train")  # will auto-pick evaluation
    except Exception as err:
        raise RuntimeError(
            "Failed to load 'scidocs_mag_mesh' from allenai/scirepeval. "
            "Please check that the datasets package and network are available."
        ) from err

    doc_store: Dict[str, DocRecord] = {}
    for row in tqdm(scidocs_ds, desc="Loading SciDocs metadata"):
        doc_id = _extract_doc_id(row)
        if not doc_id:
            continue

        # abstract-only
        text = _extract_text(
            row,
            title_keys=(),  # <- empty => don't prepend title
            text_keys=("abstract", "text", "body"),
        )
        if text and len(text) < min_len:
            text = None

        fos = row.get("fos") or row.get("field_of_study")
        year = _safe_int(row.get("pub_year") or row.get("year"))

        doc_store[str(doc_id)] = DocRecord(doc_id=str(doc_id), text=text or "", fos=fos, year=year)

    # Optional enrichment: merge FOS and publication year from dedicated configs when available.
    for cfg, attr in (("fos", "fos"), ("pub_year", "year")):
        try:
            extra_ds = _load_scirepeval(cfg, preferred="train")
        except Exception:
            continue
        for row in tqdm(extra_ds, desc=f"Merging {cfg}"):
            doc_id = _extract_doc_id(row)
            if not doc_id:
                continue
            doc_id = str(doc_id)
            if doc_id not in doc_store:
                continue

            if attr == "fos":
                doc_store[doc_id].fos = row.get("fos") or row.get("field_of_study")
            elif attr == "year":
                yr = _safe_int(row.get("pub_year") or row.get("year"))
                if yr is not None:
                    doc_store[doc_id].year = yr

    return doc_store


def _load_scidocs_eval_ids() -> Set[str]:
    """Collect doc_ids that belong to SciDocs evaluation to avoid leakage."""
    eval_ids: Set[str] = set()
    try:
        ds = _load_scirepeval("scidocs", preferred="train")
    except Exception:
        return eval_ids

    for row in tqdm(ds, desc="Collecting ids from scidocs"):
        for key in ("corpus_id", "paper_id", "source_id", "target_id", "pid"):
            val = row.get(key)
            if val:
                eval_ids.add(str(val))
    return eval_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCIDOCS-like citation triples from SciRepEval")
    parser.add_argument("--save_root", default=os.environ.get("DATA_AUG_ORIGINAL_ROOT", "./original_data"))
    parser.add_argument(
        "--output_path",
        default=None,
        help="Full path to save JSONL. Defaults to <save_root>/scidocs/scirep_citation_train/en_scirep.jsonl",
    )
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of queries to output; -1 keeps all")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed")
    parser.add_argument("--min_year", type=int, default=0, help="Minimum publication year to keep (0 disables)")
    parser.add_argument(
        "--fos",
        default="",
        help="Required field-of-study tag (case-insensitive substring match); empty disables",
    )
    parser.add_argument(
        "--min_text_len",
        type=int,
        default=200,
        help="Drop pos/neg abstracts shorter than this many chars (query title is NOT filtered by this).",
    )
    parser.add_argument("--pos_per_query", type=int, default=5, help="How many positives per query")
    parser.add_argument("--neg_per_query", type=int, default=10, help="How many negatives per query")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument(
        "--mteb_qrels_dir",
        default="",
        help="Optional: path to MTEB-SCIDOCS qrels dir; if set, all ids in qrels will be excluded to prevent leakage.",
    )
    parser.add_argument(
        "--topic_summary_model",
        default="",
        help=(
            "Optional: model name served by a vLLM/OpenAI-compatible endpoint to summarize SCIDOCS topics. "
            "If set, anchor vocabulary will be derived from the generated keywords instead of raw frequency counts."
        ),
    )
    parser.add_argument(
        "--topic_summary_endpoint",
        default=os.environ.get("VLLM_ENDPOINT", ""),
        help="Optional: base_url for the vLLM/OpenAI-compatible endpoint (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--topic_summary_max_docs",
        type=int,
        default=200,
        help=(
            "How many SCIDOCS evaluation documents to summarize for topic anchors; "
            "0 or negative values will use the full evaluation set. "
            "A random sample of titles+abstracts will be sent to the vLLM endpoint."
        ),
    )
    parser.add_argument(
        "--topic_keywords_per_chunk",
        type=int,
        default=48,
        help="How many keywords to request from the vLLM summarizer per prompt chunk.",
    )
    parser.add_argument(
        "--min_anchor_overlap",
        type=int,
        default=0,
        help=(
            "Filter out queries with fewer than this many shared tokens inside the domain anchor vocabulary. "
            "0 disables anchor filtering."
        ),
    )
    parser.add_argument(
        "--anchor_vocab_size",
        type=int,
        default=8000,
        help="Number of most common tokens from SciDocs metadata to keep for domain anchoring; 0 disables the vocabulary.",
    )
    parser.add_argument(
        "--anchor_min_token_len",
        type=int,
        default=3,
        help="Minimum token length when constructing and applying the anchor vocabulary.",
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


def _tokenize(text: str, *, min_len: int) -> List[str]:
    return [tok for tok in re.findall(r"[A-Za-z]+", text.lower()) if len(tok) >= min_len]


def _build_anchor_vocab(
    doc_store: Dict[str, DocRecord],
    *,
    vocab_size: int,
    min_token_len: int,
) -> Set[str]:
    """
    Build a SciDocs-aware anchor vocabulary from non-eval abstracts/titles to filter out out-of-domain pairs.
    """
    counter: Counter[str] = Counter()
    if vocab_size <= 0:
        return set()

    for _, rec in doc_store.items():
        counter.update(_tokenize(rec.text, min_len=min_token_len))

    return {tok for tok, _ in counter.most_common(vocab_size)}


def _ensure_openai() -> "openai":
    import importlib

    if importlib.util.find_spec("openai") is None:
        raise RuntimeError(
            "The 'openai' package is required for vLLM keyword summarization. Install with `pip install openai>=1.0`."
        )
    import openai

    return openai


def _summarize_scidocs_topics_with_vllm(
    *,
    model: str,
    endpoint: str,
    max_docs: int,
    keywords_per_chunk: int,
    anchor_min_token_len: int,
    seed: int,
) -> Set[str]:
    """
    Use a vLLM/OpenAI-compatible endpoint to summarize SCIDOCS evaluation topics into a keyword anchor set.
    """
    if not model:
        return set()
    if not endpoint:
        raise RuntimeError("--topic_summary_endpoint (or VLLM_ENDPOINT env var) is required when --topic_summary_model is set")

    openai = _ensure_openai()
    client = openai.OpenAI(base_url=endpoint, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    eval_ds = _load_scirepeval("scidocs", preferred="train")
    rows = list(eval_ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:max_docs] if max_docs > 0 else rows

    payloads: List[str] = []
    for row in rows:
        title, abstract, _ = _get_title_abs(row.get("source") or row.get("doc") or row)
        snippet = f"Title: {title}".strip()
        if abstract:
            snippet += f" | Abstract: {abstract}"
        payloads.append(snippet[:800])

    anchor_tokens: Set[str] = set()
    chunk_size = 12
    for i in range(0, len(payloads), chunk_size):
        chunk = "\n".join(payloads[i : i + chunk_size])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert scientific curator. Extract distinct topical keywords (nouns/noun phrases) that best "
                    "summarize the research themes across the provided SCIDOCS papers. "
                    "Return ONLY a comma-separated list without numbering or extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paper snippets:\n{chunk}\n\nRespond with {keywords_per_chunk} keywords, concise and lower-case."
                ),
            },
        ]

        resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, max_tokens=128)
        text = resp.choices[0].message.content or ""
        for tok in re.split(r"[,\n]+", text):
            tok = tok.strip().lower()
            if len(tok) >= anchor_min_token_len:
                anchor_tokens.add(tok)

    return anchor_tokens


def _passes_anchor(text: str, anchor_vocab: Set[str], *, min_overlap: int, min_token_len: int) -> bool:
    if min_overlap <= 0 or not anchor_vocab:
        return True
    overlap = anchor_vocab.intersection(_tokenize(text, min_len=min_token_len))
    return len(overlap) >= min_overlap


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
    pos_per_query: int,
    neg_per_query: int,
    anchor_vocab: Set[str],
    min_anchor_overlap: int,
    anchor_min_token_len: int,
) -> List[dict]:
    """
    Desired output:
      query = title
      pos/neg = abstracts
      each query has exactly pos_per_query positives and neg_per_query negatives (otherwise skipped)
    """
    rng = random.Random(seed)

    # qid -> {"query_title": str, "pos": {pid: abs}, "neg": {nid: abs}}
    buckets: Dict[str, Dict[str, Any]] = {}

    rows = list(citation_ds)
    rng.shuffle(rows)

    for row in tqdm(rows, desc="Grouping cite_prediction_new by query_id"):
        q_title, q_abs, qid = _get_title_abs(row.get("query"))
        p_title, p_abs, pid = _get_title_abs(row.get("pos"))
        n_title, n_abs, nid = _get_title_abs(row.get("neg"))

        if not qid or not pid or not nid:
            continue
        # Only block evaluation IDs when they are used as queries; allow pos/neg to keep distributional coverage.
        if qid in scidocs_eval_ids:
            continue
        if pid == nid:
            continue

        # query: title only
        if not q_title:
            continue

        if not _passes_anchor(q_title, anchor_vocab, min_overlap=min_anchor_overlap, min_token_len=anchor_min_token_len):
            continue

        # pos/neg: abstract (fallback to title if abstract missing)
        p_text = p_abs or p_title
        n_text = n_abs or n_title
        if not p_text or not n_text:
            continue
        if len(p_text) < min_text_len or len(n_text) < min_text_len:
            continue

        # Ensure doc_store has entries (keep fos/year if exists)
        # - query store title
        # - docs store abstract/text
        if qid not in doc_store:
            doc_store[qid] = DocRecord(doc_id=qid, text=q_title)
        else:
            if not doc_store[qid].text:
                doc_store[qid].text = q_title

        for did, txt in ((pid, p_text), (nid, n_text)):
            if did not in doc_store:
                doc_store[did] = DocRecord(doc_id=did, text=txt)
            else:
                if not doc_store[did].text:
                    doc_store[did].text = txt

        # Apply fos/year filters if metadata exists (missing fos/year will pass)
        if not _passes_filters(doc_store[qid], fos_filter=fos_filter, min_year=min_year):
            continue
        if not _passes_filters(doc_store[pid], fos_filter=fos_filter, min_year=min_year):
            continue
        if not _passes_filters(doc_store[nid], fos_filter=fos_filter, min_year=min_year):
            continue

        b = buckets.setdefault(qid, {"query_title": q_title, "pos": {}, "neg": {}})
        # keep a stable query_title (first non-empty wins)
        if not b["query_title"]:
            b["query_title"] = q_title
        b["pos"][pid] = p_text
        b["neg"][nid] = n_text

    # sample fixed-size pos/neg per query
    qids = list(buckets.keys())
    rng.shuffle(qids)

    out: List[dict] = []
    for qid in tqdm(qids, desc="Sampling 5 pos + 10 neg per query"):
        b = buckets[qid]
        pos_items = list(b["pos"].items())
        neg_items = list(b["neg"].items())

        if len(pos_items) < pos_per_query or len(neg_items) < neg_per_query:
            continue

        chosen_pos = rng.sample(pos_items, pos_per_query)
        chosen_neg = rng.sample(neg_items, neg_per_query)

        pos_ids = [pid for pid, _ in chosen_pos]
        neg_ids = [nid for nid, _ in chosen_neg]
        pos_texts = [txt for _, txt in chosen_pos]
        neg_texts = [txt for _, txt in chosen_neg]

        doc_pool: Dict[str, str] = {qid: b["query_title"]}
        doc_pool.update({pid: txt for pid, txt in chosen_pos})
        doc_pool.update({nid: txt for nid, txt in chosen_neg})

        out.append(
            {
                "query": b["query_title"],  # title only
                "pos": pos_texts,           # abstracts
                "neg": neg_texts,           # abstracts
                "source_ids": {
                    "query_id": qid,
                    "pos_ids": pos_ids,
                    "neg_ids": neg_ids,
                },
                "doc_pool": doc_pool,
            }
        )

        if 0 < num_samples == len(out):
            break

    return out


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _load_mteb_qrels_ids(qrels_dir: str) -> Set[str]:
    """
    Read all qrels files under a directory; collect both query_id and corpus_id.
    Compatible with common BEIR/MTEB qrels formats (tab/space separated).
    """
    ids: Set[str] = set()
    if not qrels_dir:
        return ids

    p = Path(qrels_dir)
    if not p.exists():
        raise FileNotFoundError(f"qrels_dir not found: {qrels_dir}")

    # try common patterns
    files = list(p.rglob("*.tsv")) + list(p.rglob("*.txt")) + list(p.rglob("*.qrels"))
    if not files:
        # still try all files in dir (last resort)
        files = [x for x in p.rglob("*") if x.is_file()]

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.lower().startswith("query"):  # header-like
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        parts = line.split()
                    if len(parts) < 2:
                        continue
                    qid, did = parts[0].strip(), parts[1].strip()
                    if qid:
                        ids.add(str(qid))
                    if did:
                        ids.add(str(did))
        except Exception:
            # ignore unreadable files
            continue

    return ids

def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace.")

    # doc_store will be used for fos/year enrichment (if any) + fallback text store
    doc_store = _load_doc_metadata(min_len=args.min_text_len)
    scidocs_eval_ids = _load_scidocs_eval_ids()
    if args.mteb_qrels_dir:
        scidocs_eval_ids |= _load_mteb_qrels_ids(args.mteb_qrels_dir)

    topic_anchor = _summarize_scidocs_topics_with_vllm(
        model=args.topic_summary_model,
        endpoint=args.topic_summary_endpoint,
        max_docs=args.topic_summary_max_docs,
        keywords_per_chunk=args.topic_keywords_per_chunk,
        anchor_min_token_len=args.anchor_min_token_len,
        seed=args.seed,
    )

    anchor_vocab = topic_anchor or _build_anchor_vocab(
        doc_store,
        vocab_size=args.anchor_vocab_size,
        min_token_len=args.anchor_min_token_len,
    )


    try:
        citation_ds = load_dataset("allenai/scirepeval", "cite_prediction_new", split="train")
    except Exception as err:
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
        pos_per_query=args.pos_per_query,
        neg_per_query=args.neg_per_query,
        anchor_vocab=anchor_vocab,
        min_anchor_overlap=args.min_anchor_overlap,
        anchor_min_token_len=args.anchor_min_token_len,
    )

    save_jsonl(samples, output_path)
    print(f"Saved {len(samples)} queries (each {args.pos_per_query} pos + {args.neg_per_query} neg) to {output_path}")


if __name__ == "__main__":
    main()
