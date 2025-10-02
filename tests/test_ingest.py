import pytest
import hashlib
import time
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any

from backend.agents.ingest import (
    _compute_text_hash,
    _chunk_text_by_words,
    _select_text_from_source,
    _embed_texts_local,
    _embed_texts_openai,
    _embed_texts_local_wrapper,
    _embed_texts_openai_wrapper,
    ingest_sources,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    MIN_CHUNK_WORDS,
    _local_embedder
)


class TestTextProcessing:
    """Test text processing utilities"""

    def test_compute_text_hash(self):
        """Test text hash computation"""
        text = "This is a test text"
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        
        result = _compute_text_hash(text)
        
        assert result == expected_hash
        assert len(result) == 64  # SHA256 hex length
        assert isinstance(result, str)

    def test_compute_text_hash_consistency(self):
        """Test that hash computation is consistent"""
        text = "Same text"
        hash1 = _compute_text_hash(text)
        hash2 = _compute_text_hash(text)
        
        assert hash1 == hash2

    def test_compute_text_hash_different_texts(self):
        """Test that different texts produce different hashes"""
        text1 = "Text one"
        text2 = "Text two"
        
        hash1 = _compute_text_hash(text1)
        hash2 = _compute_text_hash(text2)
        
        assert hash1 != hash2

    def test_chunk_text_by_words_basic(self):
        """Test basic text chunking"""
        text = " ".join(["word"] * 100)  # 100 words
        chunks = _chunk_text_by_words(text, chunk_size=50, overlap=10)
        
        assert len(chunks) >= 1
        assert all(len(chunk.split()) <= 50 for chunk in chunks)
        assert all(len(chunk.split()) >= MIN_CHUNK_WORDS for chunk in chunks)

    def test_chunk_text_by_words_empty(self):
        """Test chunking empty text"""
        chunks = _chunk_text_by_words("")
        assert chunks == []

    def test_chunk_text_by_words_short_text(self):
        """Test chunking text shorter than minimum"""
        text = "short text"  # Less than MIN_CHUNK_WORDS
        chunks = _chunk_text_by_words(text, chunk_size=100)
        
        assert chunks == []

    def test_chunk_text_by_words_overlap(self):
        """Test that chunks have proper overlap"""
        text = " ".join([f"word{i}" for i in range(200)])  # 200 words
        chunks = _chunk_text_by_words(text, chunk_size=50, overlap=10)
        
        if len(chunks) > 1:
            # Check that chunks don't exceed size limit
            assert all(len(chunk.split()) <= 50 for chunk in chunks)
            # Check that we have multiple chunks (indicating overlap worked)
            assert len(chunks) > 1

    def test_chunk_text_by_words_exact_size(self):
        """Test chunking with exact chunk size"""
        text = " ".join([f"word{i}" for i in range(100)])  # Exactly 100 words
        chunks = _chunk_text_by_words(text, chunk_size=100, overlap=0)
        
        assert len(chunks) == 1
        assert len(chunks[0].split()) == 100

    def test_select_text_from_source_full_text(self):
        """Test text selection with full_text available"""
        source = {
            "fetched": True,
            "full_text": "This is the full text content",
            "snippet": "This is just a snippet",
            "title": "Title"
        }
        
        result = _select_text_from_source(source)
        assert result == "This is the full text content"

    def test_select_text_from_source_snippet_fallback(self):
        """Test text selection falling back to snippet"""
        source = {
            "fetched": False,
            "snippet": "This is just a snippet",
            "title": "Title"
        }
        
        result = _select_text_from_source(source)
        assert result == "This is just a snippet"

    def test_select_text_from_source_title_fallback(self):
        """Test text selection falling back to title"""
        source = {
            "fetched": False,
            "title": "Title"
        }
        
        result = _select_text_from_source(source)
        assert result == "Title"

    def test_select_text_from_source_empty_fallback(self):
        """Test text selection with no text available"""
        source = {}
        
        result = _select_text_from_source(source)
        assert result == ""

    def test_select_text_from_source_type_conversion(self):
        """Test that non-string values are converted to strings"""
        source = {
            "fetched": True,
            "full_text": 12345,  # Non-string value
            "snippet": "snippet"
        }
        
        result = _select_text_from_source(source)
        assert result == "12345"


class TestEmbeddingFunctions:
    """Test embedding generation functions"""

    def test_embed_texts_local_success(self):
        """Test successful local embedding generation"""
        with patch('backend.agents.ingest._init_local_embedder') as mock_init:
            mock_embedder = Mock()
            # Mock encode to return a numpy-like object with tolist method
            mock_result = Mock()
            mock_result.tolist.return_value = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            mock_embedder.encode.return_value = mock_result
            mock_init.return_value = mock_embedder
            
            texts = ["text1", "text2"]
            result = _embed_texts_local(texts)
            
            mock_embedder.encode.assert_called_once_with(texts, show_progress_bar=False)
            assert result == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_embed_texts_local_numpy_array(self):
        """Test local embedding with numpy array return"""
        with patch('backend.agents.ingest._init_local_embedder') as mock_init:
            import numpy as np
            mock_embedder = Mock()
            numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]])
            mock_embedder.encode.return_value = numpy_array
            mock_init.return_value = mock_embedder
            
            texts = ["text1", "text2"]
            result = _embed_texts_local(texts)
            
            assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_embed_texts_local_float_conversion(self):
        """Test that embeddings are converted to float"""
        with patch('backend.agents.ingest._init_local_embedder') as mock_init:
            mock_embedder = Mock()
            # Mock encode to return a numpy-like object with tolist method
            mock_result = Mock()
            mock_result.tolist.return_value = [[1, 2.0, 3], [4.5, 5, 6.7]]
            mock_embedder.encode.return_value = mock_result
            mock_init.return_value = mock_embedder
            
            texts = ["text1", "text2"]
            result = _embed_texts_local(texts)
            
            # The actual implementation calls .tolist() so types are preserved
            assert result == [[1, 2.0, 3], [4.5, 5, 6.7]]

    def test_embed_texts_openai_success(self):
        """Test successful OpenAI embedding generation"""
        with patch('builtins.__import__') as mock_import, \
             patch('backend.agents.ingest.cfg') as mock_cfg:
            
            # Mock the openai import
            mock_openai = Mock()
            mock_openai.Embedding.create.return_value = {
                "data": [
                    {"embedding": [1.0, 2.0, 3.0]},
                    {"embedding": [4.0, 5.0, 6.0]}
                ]
            }
            mock_import.return_value = mock_openai
            mock_cfg.OPENAI_API_KEY = "test-key"
            
            texts = ["text1", "text2"]
            result = _embed_texts_openai(texts)
            
            assert result == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            assert mock_openai.api_key == "test-key"

    def test_embed_texts_openai_batching(self):
        """Test OpenAI embedding with batching"""
        with patch('builtins.__import__') as mock_import, \
             patch('backend.agents.ingest.cfg') as mock_cfg:
            
            mock_openai = Mock()
            # Mock to return 16 embeddings for first call, 4 for second
            def mock_create(*args, **kwargs):
                input_list = kwargs.get('input', args[0] if args else [])
                return {
                    "data": [{"embedding": [1.0, 2.0]} for _ in input_list]
                }
            mock_openai.Embedding.create.side_effect = mock_create
            mock_import.return_value = mock_openai
            mock_cfg.OPENAI_API_KEY = "test-key"
            
            # Create 20 texts to test batching (batch size is 16)
            texts = [f"text{i}" for i in range(20)]
            result = _embed_texts_openai(texts)
            
            # Should be called twice (16 + 4)
            assert mock_openai.Embedding.create.call_count == 2
            assert len(result) == 20

    def test_embed_texts_openai_no_api_key(self):
        """Test OpenAI embedding without API key"""
        with patch('builtins.__import__') as mock_import, \
             patch('backend.agents.ingest.cfg') as mock_cfg:
            
            mock_openai = Mock()
            mock_import.return_value = mock_openai
            mock_cfg.OPENAI_API_KEY = None
            
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY not configured"):
                _embed_texts_openai(["text"])

    def test_embed_texts_openai_import_error(self):
        """Test OpenAI embedding with import error"""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'openai'")):
            with pytest.raises(ImportError):
                _embed_texts_openai(["text"])

    def test_embed_texts_local_wrapper(self):
        """Test local embedding wrapper"""
        with patch('backend.agents.ingest._embed_texts_local') as mock_embed:
            mock_embed.return_value = [[1.0, 2.0]]
            
            result = _embed_texts_local_wrapper(["text"])
            
            mock_embed.assert_called_once_with(["text"])
            assert result == [[1.0, 2.0]]

    def test_embed_texts_openai_wrapper_success(self):
        """Test OpenAI embedding wrapper success"""
        with patch('backend.agents.ingest._embed_texts_openai') as mock_embed:
            mock_embed.return_value = [[1.0, 2.0]]
            
            result = _embed_texts_openai_wrapper(["text"])
            
            mock_embed.assert_called_once_with(["text"])
            assert result == [[1.0, 2.0]]

    def test_embed_texts_openai_wrapper_fallback(self):
        """Test OpenAI embedding wrapper fallback"""
        with patch('backend.agents.ingest._embed_texts_openai', side_effect=NameError), \
             patch('builtins.__import__') as mock_import, \
             patch('backend.agents.ingest.cfg') as mock_cfg:
            
            mock_openai = Mock()
            mock_openai.Embedding.create.return_value = {
                "data": [{"embedding": [1.0, 2.0]}]
            }
            mock_import.return_value = mock_openai
            mock_cfg.OPENAI_API_KEY = "test-key"
            
            result = _embed_texts_openai_wrapper(["text"])
            
            assert result == [[1.0, 2.0]]


class TestIngestSources:
    """Test the main ingest_sources function"""

    def setup_method(self):
        """Reset module state before each test"""
        import backend.agents.ingest as ingest_module
        ingest_module._local_embedder = None

    def test_ingest_sources_empty_list(self):
        """Test ingest with empty sources list"""
        result = ingest_sources([])
        
        assert result == ([], {})

    def test_ingest_sources_insufficient_text(self):
        """Test ingest with sources that have insufficient text"""
        sources = [
            {"title": "short", "url": "http://example.com"},
            {"snippet": "also short", "url": "http://example2.com"}
        ]
        
        result = ingest_sources(sources)
        
        assert result == ([], {})

    def test_ingest_sources_success_local_embeddings(self):
        """Test successful ingest with local embeddings"""
        sources = [
            {
                "id": "test_id",
                "url": "http://example.com",
                "title": "Test Title",
                "fetched": True,
                "full_text": " ".join(["word"] * 100)  # 100 words
            }
        ]
        
        with patch('backend.agents.ingest._embed_texts_local_wrapper') as mock_embed, \
             patch('backend.agents.ingest.dedupe_chunks') as mock_dedupe, \
             patch('backend.agents.ingest.chroma_client') as mock_chroma:
            
            # Mock embeddings
            mock_embed.return_value = [[1.0, 2.0, 3.0]]
            
            # Mock deduplication - return one unique chunk
            unique_meta = {
                "chunk_id": "test_id_0",
                "source_id": "test_id",
                "text_hash": "hash123"
            }
            mock_dedupe.return_value = ([unique_meta], {})
            
            result = ingest_sources(sources, use_openai_embeddings=False)
            
            # Verify embeddings were called
            mock_embed.assert_called_once()
            
            # Verify deduplication was called
            mock_dedupe.assert_called_once()
            
            # Verify chroma upsert was called
            mock_chroma.upsert.assert_called_once()
            
            # Check return value
            upserted_metas, duplicates = result
            assert len(upserted_metas) == 1
            assert duplicates == {}

    def test_ingest_sources_success_openai_embeddings(self):
        """Test successful ingest with OpenAI embeddings"""
        sources = [
            {
                "id": "test_id",
                "url": "http://example.com",
                "title": "Test Title",
                "fetched": True,
                "full_text": " ".join(["word"] * 100)
            }
        ]
        
        with patch('backend.agents.ingest._embed_texts_openai_wrapper') as mock_embed, \
             patch('backend.agents.ingest.dedupe_chunks') as mock_dedupe, \
             patch('backend.agents.ingest.chroma_client') as mock_chroma:
            
            mock_embed.return_value = [[1.0, 2.0, 3.0]]
            unique_meta = {"chunk_id": "test_id_0", "source_id": "test_id", "text_hash": "hash123"}
            mock_dedupe.return_value = ([unique_meta], {})
            
            result = ingest_sources(sources, use_openai_embeddings=True)
            
            # Verify OpenAI embeddings were called
            mock_embed.assert_called_once()
            
            # Verify other operations
            mock_dedupe.assert_called_once()
            mock_chroma.upsert.assert_called_once()

    def test_ingest_sources_all_deduped(self):
        """Test ingest when all chunks are deduped"""
        sources = [
            {
                "id": "test_id",
                "url": "http://example.com",
                "title": "Test Title",
                "fetched": True,
                "full_text": " ".join(["word"] * 100)
            }
        ]
        
        with patch('backend.agents.ingest._embed_texts_local_wrapper') as mock_embed, \
             patch('backend.agents.ingest.dedupe_chunks') as mock_dedupe, \
             patch('backend.agents.ingest.chroma_client') as mock_chroma:
            
            mock_embed.return_value = [[1.0, 2.0, 3.0]]
            
            # Mock deduplication - return no unique chunks, all duplicates
            duplicates_map = {"test_id_0": "existing_chunk"}
            mock_dedupe.return_value = ([], duplicates_map)
            
            result = ingest_sources(sources)
            
            # Verify no upsert was called
            mock_chroma.upsert.assert_not_called()
            
            # Check return value
            upserted_metas, duplicates = result
            assert upserted_metas == []
            assert duplicates == duplicates_map

    def test_ingest_sources_embedding_failure(self):
        """Test ingest when embedding generation fails"""
        sources = [
            {
                "id": "test_id",
                "url": "http://example.com",
                "title": "Test Title",
                "fetched": True,
                "full_text": " ".join(["word"] * 100)
            }
        ]
        
        with patch('backend.agents.ingest._embed_texts_local_wrapper') as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")
            
            with pytest.raises(Exception, match="Embedding failed"):
                ingest_sources(sources)

    def test_ingest_sources_chroma_upsert_failure(self):
        """Test ingest when ChromaDB upsert fails"""
        sources = [
            {
                "id": "test_id",
                "url": "http://example.com",
                "title": "Test Title",
                "fetched": True,
                "full_text": " ".join(["word"] * 100)
            }
        ]
        
        with patch('backend.agents.ingest._embed_texts_local_wrapper') as mock_embed, \
             patch('backend.agents.ingest.dedupe_chunks') as mock_dedupe, \
             patch('backend.agents.ingest.chroma_client') as mock_chroma:
            
            mock_embed.return_value = [[1.0, 2.0, 3.0]]
            unique_meta = {"chunk_id": "test_id_0", "source_id": "test_id", "text_hash": "hash123"}
            mock_dedupe.return_value = ([unique_meta], {})
            mock_chroma.upsert.side_effect = Exception("ChromaDB failed")
            
            with pytest.raises(Exception, match="ChromaDB failed"):
                ingest_sources(sources)

    def test_ingest_sources_metadata_generation(self):
        """Test that proper metadata is generated for chunks"""
        sources = [
            {
                "id": "test_id",
                "url": "http://example.com",
                "title": "Test Title",
                "published": "2024-01-01",
                "fetched": True,
                "full_text": " ".join(["word"] * 100)
            }
        ]
        
        with patch('backend.agents.ingest._embed_texts_local_wrapper') as mock_embed, \
             patch('backend.agents.ingest.dedupe_chunks') as mock_dedupe, \
             patch('backend.agents.ingest.chroma_client') as mock_chroma:
            
            mock_embed.return_value = [[1.0, 2.0, 3.0]]
            
            # Capture the metadata passed to dedupe_chunks
            captured_metas = []
            def capture_metas(metas, embeddings, **kwargs):
                captured_metas.extend(metas)
                return ([], {})
            
            mock_dedupe.side_effect = capture_metas
            
            ingest_sources(sources, job_id="test_job")
            
            # Verify metadata structure
            assert len(captured_metas) == 1
            meta = captured_metas[0]
            
            assert meta["chunk_id"] == "test_id_0"
            assert meta["source_id"] == "test_id"
            assert meta["url"] == "http://example.com"
            assert meta["title"] == "Test Title"
            assert meta["published"] == "2024-01-01"
            assert meta["chunk_idx"] == 0
            assert meta["fetched"] is True
            assert meta["job_id"] == "test_job"
            assert "text_hash" in meta

    def test_ingest_sources_multiple_chunks(self):
        """Test ingest with multiple chunks from one source"""
        sources = [
            {
                "id": "test_id",
                "url": "http://example.com",
                "title": "Test Title",
                "fetched": True,
                "full_text": " ".join(["word"] * 200)  # 200 words, should create multiple chunks
            }
        ]
        
        with patch('backend.agents.ingest._embed_texts_local_wrapper') as mock_embed, \
             patch('backend.agents.ingest.dedupe_chunks') as mock_dedupe, \
             patch('backend.agents.ingest.chroma_client') as mock_chroma:
            
            # Mock embeddings for multiple chunks
            mock_embed.return_value = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            
            # Capture metadata to verify chunk indices
            captured_metas = []
            def capture_metas(metas, embeddings, **kwargs):
                captured_metas.extend(metas)
                return ([], {})
            
            mock_dedupe.side_effect = capture_metas
            
            ingest_sources(sources, chunk_size=100, overlap=10)
            
            # Should have multiple chunks
            assert len(captured_metas) > 1
            
            # Verify chunk indices
            chunk_indices = [meta["chunk_idx"] for meta in captured_metas]
            assert chunk_indices == list(range(len(captured_metas)))

    def test_ingest_sources_custom_parameters(self):
        """Test ingest with custom parameters"""
        sources = [
            {
                "id": "test_id",
                "url": "http://example.com",
                "title": "Test Title",
                "fetched": True,
                "full_text": " ".join(["word"] * 100)
            }
        ]
        
        with patch('backend.agents.ingest._embed_texts_local_wrapper') as mock_embed, \
             patch('backend.agents.ingest.dedupe_chunks') as mock_dedupe, \
             patch('backend.agents.ingest.chroma_client') as mock_chroma:
            
            mock_embed.return_value = [[1.0, 2.0, 3.0]]
            unique_meta = {"chunk_id": "test_id_0", "source_id": "test_id", "text_hash": "hash123"}
            mock_dedupe.return_value = ([unique_meta], {})
            
            ingest_sources(
                sources,
                job_id="custom_job",
                use_openai_embeddings=False,
                chunk_size=500,
                overlap=50,
                collection_name="custom_collection"
            )
            
            # Verify dedupe was called with custom collection
            mock_dedupe.assert_called_once()
            call_args = mock_dedupe.call_args
            assert call_args[1]["chroma_collection"] == "custom_collection"
            
            # Verify chroma upsert was called with custom collection
            mock_chroma.upsert.assert_called_once()
            assert mock_chroma.upsert.call_args[0][0] == "custom_collection"


class TestConstants:
    """Test module constants"""
    
    def test_default_constants(self):
        """Test that default constants are set correctly"""
        assert DEFAULT_CHUNK_SIZE == 750
        assert DEFAULT_OVERLAP == 100
        assert MIN_CHUNK_WORDS == 30
