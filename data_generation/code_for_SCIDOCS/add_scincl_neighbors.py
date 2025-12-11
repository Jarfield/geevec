"""
Stage B: Add SciNCL-style neighbors on top of the Stage A JSONL.

The script expects Stage A output with `doc_pool` + `source_ids` so we can expand
pos/neg lists while keeping compatibility with data_augmentation's downstream
scoring pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append SciNCL-style neighbors to triplets")
    parser.add_argument("--input_path", required=True, help="Stage A JSONL path")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Where to save the augmented JSONL; defaults to <input> with _scincl suffix",
    )
    parser.add_argument("--num_pos_neighbors", type=int, default=2, help="How many extra positives per sample")
    parser.add_argument("--num_hard_negatives", type=int, default=2, help="How many hard negatives per sample")
    parser.add_argument(
        "--search_depth",
        type=int,
        default=50,
        help="Top-k neighbors to inspect when sampling positives/negatives",
    )
    parser.add_argument(
        "--index_factory",
        default="FlatIP",
        help="FAISS index factory string (embeddings are L2-normalized beforehand)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding documents into vectors",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_path if it already exists",
    )
    return parser.parse_args()


def _default_output_path(input_path: str) -> Path:
    stem = Path(input_path).with_suffix("")
    return Path(f"{stem}_scincl.jsonl")


def load_jsonl(path: str) -> List[dict]:
    data: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def encode_batch(texts: List[str]) -> np.ndarray:
    """
    TODO: Replace this stub with your embedding model.

    Returns an array of shape (len(texts), dim). Remember to L2-normalize if
    you change the FAISS index to IP/FlatIP.
    """
    raise NotImplementedError(
        "Please implement encode_batch(texts) with your embedding model (e.g., SentenceTransformers)."
    )


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_doc_pool(data: List[dict]) -> Tuple[Dict[str, str], Dict[str, int], Dict[int, str]]:
    doc_map: Dict[str, str] = {}
    for item in data:
        pool = item.get("doc_pool", {})
        for doc_id, text in pool.items():
            doc_map[doc_id] = text
    # fallback to hashed text if doc_ids are missing
    if not doc_map:
        for item in data:
            for text in item.get("pos", []) + item.get("neg", []):
                doc_id = _hash_text(text)
                doc_map[doc_id] = text
    id_to_idx: Dict[str, int] = {}
    idx_to_id: Dict[int, str] = {}
    for idx, doc_id in enumerate(sorted(doc_map.keys())):
        id_to_idx[doc_id] = idx
        idx_to_id[idx] = doc_id
    return doc_map, id_to_idx, idx_to_id


def build_index(doc_map: Dict[str, str], id_to_idx: Dict[str, int], batch_size: int, index_factory: str):
    doc_ids = [doc_id for doc_id, _ in sorted(id_to_idx.items(), key=lambda kv: kv[1])]
    texts = [doc_map[doc_id] for doc_id in doc_ids]

    embeddings: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding documents"):
        batch = texts[start : start + batch_size]
        vecs = encode_batch(batch)
        embeddings.append(vecs)
    all_embeddings = np.concatenate(embeddings, axis=0)
    faiss.normalize_L2(all_embeddings)

    dim = all_embeddings.shape[1]
    index = faiss.index_factory(dim, index_factory)
    if index.is_trained:
        pass
    else:
        index.train(all_embeddings)
    index.add(all_embeddings)
    return index, all_embeddings


def _search_neighbors(index, all_embeddings, anchor_idx: int, k: int) -> List[int]:
    anchor = all_embeddings[anchor_idx : anchor_idx + 1]
    _, indices = index.search(anchor, k + 1)
    # drop self at position 0
    return [int(i) for i in indices[0] if i != anchor_idx]


def _append_unique(target_list: List[str], candidates: Iterable[str]) -> None:
    existing = set(target_list)
    for cand in candidates:
        if cand not in existing:
            target_list.append(cand)
            existing.add(cand)


def augment_sample(
    item: dict,
    *,
    doc_map: Dict[str, str],
    id_to_idx: Dict[str, int],
    idx_to_id: Dict[int, str],
    index,
    all_embeddings: np.ndarray,
    num_pos_neighbors: int,
    num_hard_negatives: int,
    search_depth: int,
) -> dict:
    pos_ids = item.get("source_ids", {}).get("pos_ids") or []
    neg_ids = set(item.get("source_ids", {}).get("neg_ids") or [])
    query_id = item.get("source_ids", {}).get("query_id")

    pos_indices = [id_to_idx[pid] for pid in pos_ids if pid in id_to_idx]
    if not pos_indices:
        return item

    pos_candidates: List[str] = []
    neg_candidates: List[str] = []

    for pos_idx in pos_indices:
        neighbor_indices = _search_neighbors(index, all_embeddings, pos_idx, search_depth)
        for idx in neighbor_indices:
            doc_id = idx_to_id[idx]
            text = doc_map.get(doc_id)
            if not text:
                continue
            if doc_id in pos_ids:
                continue
            if query_id and doc_id == query_id:
                continue
            if len(pos_candidates) < num_pos_neighbors:
                pos_candidates.append(text)
            elif len(neg_candidates) < num_hard_negatives and doc_id not in neg_ids:
                neg_candidates.append(text)
            if len(pos_candidates) >= num_pos_neighbors and len(neg_candidates) >= num_hard_negatives:
                break
        if len(pos_candidates) >= num_pos_neighbors and len(neg_candidates) >= num_hard_negatives:
            break

    augmented = dict(item)
    augmented["pos"] = list(item.get("pos", []))
    augmented["neg"] = list(item.get("neg", []))
    _append_unique(augmented["pos"], pos_candidates)
    _append_unique(augmented["neg"], neg_candidates)
    return augmented


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path) if args.output_path else _default_output_path(args.input_path)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace.")

    data = load_jsonl(args.input_path)
    doc_map, id_to_idx, idx_to_id = build_doc_pool(data)

    if not doc_map:
        raise ValueError("No documents found in doc_pool/pos/neg fields.")

    index, all_embeddings = build_index(
        doc_map=doc_map,
        id_to_idx=id_to_idx,
        batch_size=args.batch_size,
        index_factory=args.index_factory,
    )

    augmented: List[dict] = []
    for item in tqdm(data, desc="Augmenting with neighbors"):
        augmented.append(
            augment_sample(
                item,
                doc_map=doc_map,
                id_to_idx=id_to_idx,
                idx_to_id=idx_to_id,
                index=index,
                all_embeddings=all_embeddings,
                num_pos_neighbors=args.num_pos_neighbors,
                num_hard_negatives=args.num_hard_negatives,
                search_depth=args.search_depth,
            )
        )

    save_jsonl(augmented, output_path)
    print(f"Saved augmented triples to {output_path}")


if __name__ == "__main__":
    main()
