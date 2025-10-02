# backend/agents/ingest.py
"""
Ingest agent (refactored to use backend.services.chroma_client + dedupe_rank).

Responsibilities:
- Chunk each source (prefer full_text, fallback to snippet/title)
- Compute stable text_hash for each chunk
- Generate embeddings (local sentence-transformers by default, or OpenAI embeddings optionally)
- Deduplicate chunks (exact hash + embedding similarity) via dedupe_rank.dedupe_chunks
- Upsert unique chunks into Chroma via backend.services.chroma_client.upsert
- Return (upserted_chunk_metas, duplicates_map)

Usage:
  from backend.agents.ingest import ingest_sources
  chunks_meta, duplicates = ingest_sources(sources, job_id="abc123", use_openai_embeddings=False)
"""

from __future__ import annotations
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple

from backend.config import cfg
from backend.services import chroma_client
from backend.agents.dedupe_rank import dedupe_chunks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Local embedder - lazy init
_local_embedder = None


def _init_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("sentence-transformers is required for local embeddings") from e
        model_name = cfg.EMBED_MODEL
        logger.info("Initializing local embedder: %s", model_name)
        _local_embedder = SentenceTransformer(model_name)
    return _local_embedder


def _embed_texts_local(texts: List[str]) -> List[List[float]]:
    m = _init_local_embedder()
    embs = m.encode(texts, show_progress_bar=False)
    # ensure list-of-lists
    if hasattr(embs, "tolist"):
        embs = embs.tolist()
    return [list(map(float, e)) for e in embs]


def _embed_texts_openai(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    try:
        import openai
    except Exception:
        raise RuntimeError("openai package required for cloud embeddings")
    if not getattr(cfg, "OPENAI_API_KEY", None):
        raise RuntimeError("OPENAI_API_KEY not configured in cfg")
    openai.api_key = cfg.OPENAI_API_KEY
    embeddings = []
    batch = 16
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        resp = openai.Embedding.create(input=chunk, model=model)
        for item in resp["data"]:
            embeddings.append(item["embedding"])
    return embeddings


# chunking utils
DEFAULT_CHUNK_SIZE = 750
DEFAULT_OVERLAP = 100
MIN_CHUNK_WORDS = 10


def _compute_text_hash(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _chunk_text_by_words(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
        if end == n:
            break
        # move start with overlap
        start = end - overlap if (end - overlap) > start else end
    return chunks


def _select_text_from_source(src: Dict[str, Any]) -> str:
    """
    Prefer full_text when available (premium); otherwise snippet then title.
    """
    if src.get("fetched") and src.get("full_text"):
        return str(src.get("full_text", ""))
    if src.get("snippet"):
        return str(src.get("snippet", ""))
    return str(src.get("title", ""))


def ingest_sources(sources: List[Dict[str, Any]],
                   job_id: Optional[str] = None,
                   use_openai_embeddings: bool = False,
                   chunk_size: int = DEFAULT_CHUNK_SIZE,
                   overlap: int = DEFAULT_OVERLAP,
                   collection_name: str = "snippets"
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Ingest sources into Chroma (deduped).

    Args:
      sources: list of source dicts from scout()
      job_id: optional identifier for this ingest run (saved into metadata)
      use_openai_embeddings: if True use OpenAI embeddings; otherwise local sentence-transformers
      chunk_size, overlap: chunking params
      collection_name: Chroma collection name

    Returns:
      upserted_chunk_metas: list of metadata dicts for chunks that were upserted
      duplicates_map: mapping duplicate_chunk_id -> canonical_chunk_id or EXISTING_TEXTHASH:...
    """
    start = time.time()
    if not sources:
        return [], {}

    # 1) Build chunks and metadata
    chunk_metas: List[Dict[str, Any]] = []
    chunk_texts: List[str] = []

    for src in sources:
        source_id = src.get("id") or _compute_text_hash(src.get("url", "") + str(time.time()))
        url = src.get("url", "")
        title = src.get("title", "") or ""
        published = src.get("published", None)
        fetched = bool(src.get("fetched", False))

        text = _select_text_from_source(src)
        if not text or len(text.strip().split()) < MIN_CHUNK_WORDS:
            logger.debug("Skipping source %s: insufficient text", source_id)
            continue

        chunks = _chunk_text_by_words(text, chunk_size=chunk_size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            chunk_id = f"{source_id}_{idx}"
            text_hash = _compute_text_hash(ch)
            meta = {
                "chunk_id": chunk_id,
                "source_id": source_id,
                "url": url,
                "title": title,
                "published": published,
                "chunk_idx": idx,
                "text_hash": text_hash,
                "fetched": fetched,
                "job_id": job_id
            }
            chunk_metas.append(meta)
            chunk_texts.append(ch)

    if not chunk_texts:
        logger.info("No chunkable text found across sources")
        return [], {}

    # 2) Generate embeddings
    logger.info("Generating embeddings for %d chunks (openai=%s)", len(chunk_texts), bool(use_openai_embeddings))
    try:
        if use_openai_embeddings:
            embeddings = _embed_texts_openai_wrapper(chunk_texts)
        else:
            embeddings = _embed_texts_local_wrapper(chunk_texts)
    except Exception as e:
        logger.exception("Failed to compute embeddings: %s", e)
        raise

    # 3) Dedupe via dedupe_rank (exact hash + embedding similarity + chroma NN)
    logger.info("Running dedupe on %d chunks", len(chunk_metas))
    unique_chunk_metas, duplicates_map = dedupe_chunks(chunk_metas, embeddings, chroma_collection=collection_name, similarity_threshold=0.87)

    if not unique_chunk_metas:
        logger.info("All chunks deduped or already exist; returning duplicates_map")
        return [], duplicates_map

    # Build lists for upsert aligned to unique_chunk_metas
    docs_to_upsert = []
    ids_to_upsert = []
    metas_to_upsert = []
    # We need to map unique meta order back to embeddings/texts
    text_by_chunk_id = {m["chunk_id"]: txt for m, txt in zip(chunk_metas, chunk_texts)}
    emb_by_chunk_id = {m["chunk_id"]: emb for m, emb in zip(chunk_metas, embeddings)}

    for meta in unique_chunk_metas:
        cid = meta["chunk_id"]
        txt = text_by_chunk_id.get(cid)
        emb = emb_by_chunk_id.get(cid)
        if txt is None or emb is None:
            logger.debug("Missing text/embedding for %s, skipping", cid)
            continue
        ids_to_upsert.append(cid)
        docs_to_upsert.append(txt)
        metas_to_upsert.append(meta)

    # 4) Upsert into Chroma via chroma_client
    try:
        chroma_client.upsert(collection_name, ids=ids_to_upsert, embeddings=[list(map(float, e)) for e in [emb_by_chunk_id[cid] for cid in ids_to_upsert]], metadatas=metas_to_upsert, documents=docs_to_upsert)
        logger.info("Upserted %d unique chunks into collection=%s", len(ids_to_upsert), collection_name)
    except Exception as e:
        logger.exception("Chroma upsert failed: %s", e)
        # In case of failure, raise to surface the error to caller
        raise

    # 5) Add text content to metadata for extract agent
    chunks_with_text = []
    for meta in metas_to_upsert:
        chunk_id = meta["chunk_id"]
        text = text_by_chunk_id.get(chunk_id, "")
        meta_with_text = meta.copy()
        meta_with_text["text"] = text
        chunks_with_text.append(meta_with_text)

    elapsed = time.time() - start
    logger.info("Ingest finished: prepared=%d, upserted=%d, time=%.2fs", len(chunk_metas), len(ids_to_upsert), elapsed)

    # Return the metadata with text for upserted chunks and the duplicates map
    return chunks_with_text, duplicates_map


# small wrappers for embedding functions to avoid NameError if openai not required
def _embed_texts_local_wrapper(texts: List[str]) -> List[List[float]]:
    return _embed_texts_local(texts)


def _embed_texts_openai_wrapper(texts: List[str]) -> List[List[float]]:
    # This wrapper imports openai lazily to avoid hard dependency when not used.
    try:
        return _embed_texts_openai(texts)
    except NameError:
        # define the function here lazily if not found
        def _embed_texts_openai_fallback(texts_inner):
            import openai
            if not getattr(cfg, "OPENAI_API_KEY", None):
                raise RuntimeError("OPENAI_API_KEY not configured")
            openai.api_key = cfg.OPENAI_API_KEY
            embeddings = []
            batch = 16
            for i in range(0, len(texts_inner), batch):
                chunk = texts_inner[i : i + batch]
                resp = openai.Embedding.create(input=chunk, model="text-embedding-3-small")
                for item in resp["data"]:
                    embeddings.append(item["embedding"])
            return embeddings
        return _embed_texts_openai_fallback(texts)

