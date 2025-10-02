import pytest
import numpy as np

from backend.agents.dedupe_rank import dedupe_chunks

# Helpers to build synthetic chunk metas and embeddings
def make_meta(chunk_id: str, text_hash: str) -> dict:
    return {"chunk_id": chunk_id, "text_hash": text_hash}

def vec(v):
    # ensure plain python list
    return list(np.array(v, dtype=float))


def test_dedupe_exact_text_hash(monkeypatch):
    """
    If exists_text_hash returns True for a chunk, dedupe_chunks should mark it as duplicate.
    Note: In-batch similarity check happens first, so we need embeddings that are NOT similar
    to avoid in-batch deduplication.
    """
    # Two chunks with very different embeddings to avoid in-batch deduplication
    metas = [
        make_meta("c1", "hash_a"),
        make_meta("c2", "hash_existing"),
    ]
    embeddings = [vec([1, 0, 0]), vec([0, 1, 0])]  # orthogonal vectors, similarity = 0

    # Monkeypatch exists_text_hash to return True only for "hash_existing"
    def fake_exists_text_hash(collection, text_hash):
        return text_hash == "hash_existing"
    monkeypatch.setattr("backend.agents.dedupe_rank.exists_text_hash", fake_exists_text_hash)

    unique, duplicates = dedupe_chunks(metas, embeddings, chroma_collection="snippets", similarity_threshold=0.87)

    # Expect only the first chunk considered unique, the second mapped to EXISTING_TEXTHASH:hash_existing
    unique_ids = {m["chunk_id"] for m in unique}
    assert "c1" in unique_ids
    assert "c2" not in unique_ids
    assert "c2" in duplicates
    assert duplicates["c2"].startswith("EXISTING_TEXTHASH:") and "hash_existing" in duplicates["c2"]


def test_dedupe_inbatch_cosine(monkeypatch):
    """
    Two new chunks with near-identical embeddings should dedupe in-batch:
    c2 should be deduped to c1 by cosine similarity.
    """
    metas = [
        make_meta("c1", "h1"),
        make_meta("c2", "h2"),
        make_meta("c3", "h3"),
    ]
    # c1 and c2 are identical; c3 is different
    embeddings = [
        vec([1.0, 0.0, 0.0]),
        vec([1.0, 0.0, 0.0]),
        vec([0.0, 1.0, 0.0]),
    ]

    # Ensure exists_text_hash always returns False
    monkeypatch.setattr("backend.agents.dedupe_rank.exists_text_hash", lambda collection, text_hash: False)
    # Also stub query_by_embedding to return no nearby ids
    monkeypatch.setattr("backend.agents.dedupe_rank.query_by_embedding", lambda collection, emb, top_k=3: {"ids": [[]], "distances": [[]]})

    unique, duplicates = dedupe_chunks(metas, embeddings, chroma_collection="snippets", similarity_threshold=0.99)
    # With threshold 0.99 and identical vectors, c2 should be deduped to c1
    unique_ids = [m["chunk_id"] for m in unique]
    assert "c1" in unique_ids
    assert "c3" in unique_ids
    assert "c2" not in unique_ids
    assert duplicates.get("c2") == "c1"


def test_dedupe_chroma_nn_fallback(monkeypatch):
    """
    If exists_text_hash is False and in-batch dedupe doesn't catch it,
    but Chroma query_by_embedding returns a close neighbor (distance < threshold),
    the chunk should be deduped to the returned Chroma id.
    Note: In-batch similarity check happens first, so we need embeddings that are NOT similar
    to avoid in-batch deduplication.
    """
    metas = [
        make_meta("c1", "h1"),
        make_meta("c2", "h2"),
    ]
    embeddings = [
        vec([1.0, 0.0, 0.0]),  # orthogonal to c2 to avoid in-batch deduplication
        vec([0.0, 1.0, 0.0]),  # orthogonal to c1 to avoid in-batch deduplication
    ]

    # exists_text_hash -> False for both
    monkeypatch.setattr("backend.agents.dedupe_rank.exists_text_hash", lambda collection, text_hash: False)

    # Simulate chroma returning nearest id 'existing_42' with a small distance (0.1) for c2 only
    def fake_query_by_embedding(collection, emb, top_k=3):
        # Only return a close match for the second embedding (c2)
        if emb == vec([0.0, 1.0, 0.0]):
            return {"ids": [["existing_42"]], "distances": [[0.1]]}
        else:
            return {"ids": [[]], "distances": [[]]}
    monkeypatch.setattr("backend.agents.dedupe_rank.query_by_embedding", fake_query_by_embedding)

    unique, duplicates = dedupe_chunks(metas, embeddings, chroma_collection="snippets", similarity_threshold=0.95)

    # c2 should be deduped to 'existing_42' via chroma fallback
    unique_ids = [m["chunk_id"] for m in unique]
    assert "c1" in unique_ids  # c1 accepted
    assert "c2" not in unique_ids
    assert duplicates.get("c2") == "existing_42"
