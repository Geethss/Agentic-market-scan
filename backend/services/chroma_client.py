# backend/services/chroma_client.py
"""
Central Chroma client wrapper.

Provides:
- init_client()              -> returns chroma client instance
- get_or_create_collection(name, metadata) -> collection
- upsert(ids, embeddings, metadatas, documents)
- query_by_embedding(embedding, top_k)
- query_by_text(text, top_k)
- get_by_ids(ids)
- exists_text_hash(text_hash) -> bool
- delete(ids)

Notes:
- This wrapper assumes a local Chroma instance (default).
- It keeps one collection handle per-name.
- It does not compute embeddings; caller supplies embeddings (ingest.py handles embeddings).
"""

from __future__ import annotations
import logging
import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from backend.config import cfg

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# module-level client and collections cache
_CLIENT: Optional[chromadb.Client] = None
_COLLECTIONS: Dict[str, chromadb.api.models.Collection.Collection] = {}

# Default collection name
DEFAULT_COLLECTION = "snippets"


def init_client() -> chromadb.Client:
    """
    Initialize (or return cached) Chroma client.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    try:
        # If you run an external Chroma server, you can adapt Settings accordingly.
        settings = ChromaSettings()
        _CLIENT = chromadb.Client(settings)
        logger.info("Chroma client initialized")
    except Exception as e:
        logger.exception("Failed to initialize Chroma client: %s", e)
        raise
    return _CLIENT


def get_or_create_collection(name: str = DEFAULT_COLLECTION, metadata: Optional[Dict[str, Any]] = None):
    """
    Return an existing collection or create it.
    """
    global _COLLECTIONS
    client = init_client()
    if name in _COLLECTIONS:
        return _COLLECTIONS[name]
    try:
        coll = client.get_collection(name)
    except Exception:
        # ChromaDB requires non-empty metadata, so provide a default
        collection_metadata = metadata if metadata else {"description": "Market scan snippets collection"}
        coll = client.create_collection(name, metadata=collection_metadata)
    _COLLECTIONS[name] = coll
    return coll


def upsert(collection_name: str,
           ids: List[str],
           embeddings: List[List[float]],
           metadatas: List[Dict[str, Any]],
           documents: Optional[List[str]] = None):
    """
    Upsert vectors into the collection.
    """
    coll = get_or_create_collection(collection_name)
    try:
        coll.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents or [])
        logger.debug("Upserted %d docs into collection=%s", len(ids), collection_name)
    except Exception as e:
        logger.exception("Chroma upsert failed: %s", e)
        # bubble up to caller to decide
        raise


def query_by_embedding(collection_name: str, embedding: List[float], top_k: int = 5) -> Dict[str, Any]:
    coll = get_or_create_collection(collection_name)
    try:
        res = coll.query(query_embeddings=[embedding], n_results=top_k)
        return res
    except Exception as e:
        logger.exception("Chroma query_by_embedding failed: %s", e)
        return {}


def query_by_text(collection_name: str, text: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Convenience: embed locally/given embedder is not used here.
    Use this if collection supports text queries (may rely on underlying engine).
    For robust similarity, caller should pass an embedding and call query_by_embedding.
    """
    coll = get_or_create_collection(collection_name)
    try:
        res = coll.query(query_texts=[text], n_results=top_k)
        return res
    except Exception as e:
        logger.exception("Chroma query_by_text failed: %s", e)
        return {}


def get_by_ids(collection_name: str, ids: List[str]) -> Dict[str, Any]:
    coll = get_or_create_collection(collection_name)
    try:
        return coll.get(ids=ids)
    except Exception as e:
        logger.exception("Chroma get_by_ids failed: %s", e)
        return {}


def exists_text_hash(collection_name: str, text_hash: str) -> bool:
    """
    Heuristic existence check: search collection for metadata.text_hash == text_hash.
    Note: requires that you stored 'text_hash' as metadata during upsert.
    """
    coll = get_or_create_collection(collection_name)
    try:
        # Many chroma versions support a 'get' with where filter:
        try:
            res = coll.get(where={"text_hash": text_hash}, limit=1)
            ids = res.get("ids") or []
            return len(ids) > 0
        except Exception:
            # Fallback: query by text_hash as text (not ideal but keep compatibility)
            res = coll.query(query_texts=[text_hash], n_results=1)
            # if distances returned, treat as non-existent (fallback)
            return False
    except Exception as e:
        logger.exception("Chroma exists_text_hash check failed: %s", e)
        return False


def delete(collection_name: str, ids: List[str]):
    coll = get_or_create_collection(collection_name)
    try:
        coll.delete(ids=ids)
        logger.debug("Deleted %d ids from collection=%s", len(ids), collection_name)
    except Exception as e:
        logger.exception("Chroma delete failed: %s", e)
        raise
