# backend/agents/dedupe_rank.py
"""
Dedupe & ranking utilities.

Strategy:
1) Exact dedupe using text_hash metadata (fast, deterministic).
2) Embedding-similarity dedupe: for candidate chunks without exact hash match,
   run a Chroma nearest-neighbor search using the chunk embedding and mark
   duplicates if cosine similarity >= threshold.

APIs:
- dedupe_chunks(chunk_metas, embeddings, chroma_collection="snippets", similarity_threshold=0.87)
    chunk_metas: list of metadata dicts (must include chunk_id and text_hash)
    embeddings: list parallel to chunk_metas (list of vectors)
    returns: (unique_chunks, duplicates_map)
       unique_chunks: list of chunk_metas that survive dedupe
       duplicates_map: dict duplicate_chunk_id -> canonical_chunk_id
"""

from __future__ import annotations
import logging
from typing import List, Dict, Tuple, Any
import numpy as np

from backend.services.chroma_client import query_by_embedding, exists_text_hash, get_or_create_collection

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def dedupe_chunks(chunk_metas: List[Dict[str, Any]],
                  embeddings: List[List[float]],
                  chroma_collection: str = "snippets",
                  similarity_threshold: float = 0.87) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    if not chunk_metas:
        return [], {}

    if len(chunk_metas) != len(embeddings):
        raise ValueError("chunk_metas and embeddings must be same length")

    unique_chunks: List[Dict[str, Any]] = []
    duplicates_map: Dict[str, str] = {}
    # canonical_index_map maps canonical_chunk_id -> its embedding vector
    canonical_embeddings: Dict[str, List[float]] = {}

    for i, meta in enumerate(chunk_metas):
        chunk_id = meta.get("chunk_id")
        text_hash = meta.get("text_hash")
        emb = embeddings[i]

        # 1) exact text_hash check against existing collection
        try:
            if text_hash and exists_text_hash(chroma_collection, text_hash):
                # If found in DB, treat this chunk as duplicate of external item.
                # We don't have the canonical id here; mark as duplicate of text_hash placeholder.
                # Caller can later query DB by text_hash to find canonical id.
                duplicates_map[chunk_id] = f"EXISTING_TEXTHASH:{text_hash}"
                logger.debug("Chunk %s deduped by exact text_hash", chunk_id)
                continue
        except Exception:
            # on error, proceed to embedding dedupe
            logger.debug("exists_text_hash lookup failed for %s; falling back to embedding dedupe", chunk_id)

        # 2) compare to already-accepted (in-batch) canonical embeddings
        found_dup = False
        for canon_id, canon_emb in canonical_embeddings.items():
            sim = _cosine_sim(emb, canon_emb)
            if sim >= similarity_threshold:
                duplicates_map[chunk_id] = canon_id
                found_dup = True
                logger.debug("Chunk %s deduped to in-batch canonical %s (sim=%.3f)", chunk_id, canon_id, sim)
                break
        if found_dup:
            continue

        # 3) query Chroma for similar existing documents (to dedupe against DB)
        try:
            qres = query_by_embedding(chroma_collection, emb, top_k=3)
            # qres may have 'ids', 'distances', 'documents', depending on chroma version
            ids = qres.get("ids", [[]])
            distances = qres.get("distances", [[]])
            # unify access pattern
            nearest_ids = ids[0] if isinstance(ids, list) and ids else []
            nearest_dists = distances[0] if isinstance(distances, list) and distances else []
            # some Chroma returns distance where lower is better (e.g., L2), others return similarity; assume distance -> convert
            # We'll compute embedding similarity against returned documents if possible; else, use distance heuristic
            # Here, conservatively check if any nearest exists — then compute cosine sim by fetching their embeddings is heavy.
            # Instead: if nearest_dists provided and nearest_dists[0] close to 0 -> duplicate.
            if nearest_ids:
                # If the API returns distances as floats where smaller means closer, treat small distance as duplicate.
                # We'll use a heuristic threshold: distance < 0.2 (tunable). This is a best-effort fallback.
                try:
                    first_dist = nearest_dists[0] if nearest_dists else None
                    if first_dist is not None and float(first_dist) < 0.22:
                        duplicates_map[chunk_id] = nearest_ids[0]
                        logger.debug("Chunk %s deduped to existing %s via chroma distance %.4f", chunk_id, nearest_ids[0], first_dist)
                        continue
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Chroma query_by_embedding failed for chunk %s: %s", chunk_id, e)

        # 4) no duplicates found -> accept as canonical
        unique_chunks.append(meta)
        canonical_embeddings[chunk_id] = emb

    logger.info("Dedupe: %d unique chunks, %d duplicates", len(unique_chunks), len(duplicates_map))
    return unique_chunks, duplicates_map
