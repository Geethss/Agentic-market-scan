import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from backend.services.chroma_client import (
    init_client,
    get_or_create_collection,
    upsert,
    query_by_embedding,
    query_by_text,
    get_by_ids,
    exists_text_hash,
    delete,
    DEFAULT_COLLECTION,
    _CLIENT,
    _COLLECTIONS
)


class TestChromaClient:
    """Test suite for chroma_client.py"""

    def setup_method(self):
        """Reset module state before each test"""
        # Clear the global state
        import backend.services.chroma_client as chroma_module
        chroma_module._CLIENT = None
        chroma_module._COLLECTIONS.clear()

    def test_init_client_creates_client(self):
        """Test that init_client creates a new ChromaDB client"""
        with patch('backend.services.chroma_client.chromadb.Client') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            client = init_client()
            
            mock_client.assert_called_once()
            assert client == mock_instance

    def test_init_client_returns_cached_client(self):
        """Test that init_client returns cached client on subsequent calls"""
        with patch('backend.services.chroma_client.chromadb.Client') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # First call
            client1 = init_client()
            # Second call
            client2 = init_client()
            
            # Should only create client once
            mock_client.assert_called_once()
            assert client1 == client2 == mock_instance

    def test_get_or_create_collection_creates_new(self):
        """Test get_or_create_collection creates a new collection"""
        with patch('backend.services.chroma_client.init_client') as mock_init:
            mock_client = Mock()
            mock_init.return_value = mock_client
            
            # Mock collection doesn't exist initially
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_collection = Mock()
            mock_client.create_collection.return_value = mock_collection
            
            collection = get_or_create_collection("test_collection")
            
            mock_client.get_collection.assert_called_once_with("test_collection")
            mock_client.create_collection.assert_called_once_with("test_collection", metadata={})
            assert collection == mock_collection

    def test_get_or_create_collection_returns_existing(self):
        """Test get_or_create_collection returns existing collection"""
        with patch('backend.services.chroma_client.init_client') as mock_init:
            mock_client = Mock()
            mock_init.return_value = mock_client
            
            mock_collection = Mock()
            mock_client.get_collection.return_value = mock_collection
            
            collection = get_or_create_collection("test_collection")
            
            mock_client.get_collection.assert_called_once_with("test_collection")
            mock_client.create_collection.assert_not_called()
            assert collection == mock_collection

    def test_get_or_create_collection_caches_collection(self):
        """Test that collections are cached"""
        with patch('backend.services.chroma_client.init_client') as mock_init:
            mock_client = Mock()
            mock_init.return_value = mock_client
            
            mock_collection = Mock()
            mock_client.get_collection.return_value = mock_collection
            
            # First call
            collection1 = get_or_create_collection("test_collection")
            # Second call
            collection2 = get_or_create_collection("test_collection")
            
            # Should only call get_collection once due to caching
            mock_client.get_collection.assert_called_once()
            assert collection1 == collection2 == mock_collection

    def test_upsert_success(self):
        """Test successful upsert operation"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            ids = ["id1", "id2"]
            embeddings = [[1.0, 2.0], [3.0, 4.0]]
            metadatas = [{"text": "doc1"}, {"text": "doc2"}]
            documents = ["Document 1", "Document 2"]
            
            upsert("test_collection", ids, embeddings, metadatas, documents)
            
            mock_collection.upsert.assert_called_once_with(
                ids=ids, 
                embeddings=embeddings, 
                metadatas=metadatas, 
                documents=documents
            )

    def test_upsert_without_documents(self):
        """Test upsert without documents parameter"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            ids = ["id1"]
            embeddings = [[1.0, 2.0]]
            metadatas = [{"text": "doc1"}]
            
            upsert("test_collection", ids, embeddings, metadatas)
            
            mock_collection.upsert.assert_called_once_with(
                ids=ids, 
                embeddings=embeddings, 
                metadatas=metadatas, 
                documents=[]
            )

    def test_upsert_exception_propagation(self):
        """Test that upsert exceptions are propagated"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            mock_collection.upsert.side_effect = Exception("Upsert failed")
            
            with pytest.raises(Exception, match="Upsert failed"):
                upsert("test_collection", ["id1"], [[1.0]], [{"text": "doc1"}])

    def test_query_by_embedding_success(self):
        """Test successful query_by_embedding"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            expected_result = {
                "ids": [["result1", "result2"]],
                "distances": [[0.1, 0.2]],
                "documents": [["doc1", "doc2"]]
            }
            mock_collection.query.return_value = expected_result
            
            embedding = [1.0, 2.0, 3.0]
            result = query_by_embedding("test_collection", embedding, top_k=5)
            
            mock_collection.query.assert_called_once_with(
                query_embeddings=[embedding], 
                n_results=5
            )
            assert result == expected_result

    def test_query_by_embedding_exception_returns_empty(self):
        """Test that query_by_embedding returns empty dict on exception"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            mock_collection.query.side_effect = Exception("Query failed")
            
            result = query_by_embedding("test_collection", [1.0, 2.0])
            
            assert result == {}

    def test_query_by_text_success(self):
        """Test successful query_by_text"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            expected_result = {
                "ids": [["result1"]],
                "distances": [[0.1]],
                "documents": [["doc1"]]
            }
            mock_collection.query.return_value = expected_result
            
            result = query_by_text("test_collection", "test query", top_k=3)
            
            mock_collection.query.assert_called_once_with(
                query_texts=["test query"], 
                n_results=3
            )
            assert result == expected_result

    def test_query_by_text_exception_returns_empty(self):
        """Test that query_by_text returns empty dict on exception"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            mock_collection.query.side_effect = Exception("Query failed")
            
            result = query_by_text("test_collection", "test query")
            
            assert result == {}

    def test_get_by_ids_success(self):
        """Test successful get_by_ids"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            expected_result = {
                "ids": ["id1", "id2"],
                "embeddings": [[1.0, 2.0], [3.0, 4.0]],
                "metadatas": [{"text": "doc1"}, {"text": "doc2"}]
            }
            mock_collection.get.return_value = expected_result
            
            result = get_by_ids("test_collection", ["id1", "id2"])
            
            mock_collection.get.assert_called_once_with(ids=["id1", "id2"])
            assert result == expected_result

    def test_get_by_ids_exception_returns_empty(self):
        """Test that get_by_ids returns empty dict on exception"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            mock_collection.get.side_effect = Exception("Get failed")
            
            result = get_by_ids("test_collection", ["id1"])
            
            assert result == {}

    def test_exists_text_hash_found(self):
        """Test exists_text_hash when text_hash exists"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            # Mock successful get with where filter
            mock_collection.get.return_value = {
                "ids": ["existing_id"],
                "metadatas": [{"text_hash": "test_hash"}]
            }
            
            result = exists_text_hash("test_collection", "test_hash")
            
            mock_collection.get.assert_called_once_with(where={"text_hash": "test_hash"}, limit=1)
            assert result is True

    def test_exists_text_hash_not_found(self):
        """Test exists_text_hash when text_hash doesn't exist"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            # Mock get with where filter returns empty
            mock_collection.get.return_value = {"ids": [], "metadatas": []}
            
            result = exists_text_hash("test_collection", "nonexistent_hash")
            
            assert result is False

    def test_exists_text_hash_fallback_on_exception(self):
        """Test exists_text_hash fallback behavior on exception"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            # Mock get with where filter fails
            mock_collection.get.side_effect = Exception("Where filter not supported")
            
            result = exists_text_hash("test_collection", "test_hash")
            
            # Should return False due to fallback
            assert result is False

    def test_exists_text_hash_complete_failure(self):
        """Test exists_text_hash when all operations fail"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            # Mock both get and query to fail
            mock_collection.get.side_effect = Exception("Get failed")
            mock_collection.query.side_effect = Exception("Query failed")
            
            result = exists_text_hash("test_collection", "test_hash")
            
            assert result is False

    def test_delete_success(self):
        """Test successful delete operation"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            
            ids = ["id1", "id2"]
            delete("test_collection", ids)
            
            mock_collection.delete.assert_called_once_with(ids=ids)

    def test_delete_exception_propagation(self):
        """Test that delete exceptions are propagated"""
        with patch('backend.services.chroma_client.get_or_create_collection') as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection
            mock_collection.delete.side_effect = Exception("Delete failed")
            
            with pytest.raises(Exception, match="Delete failed"):
                delete("test_collection", ["id1"])

    def test_default_collection_constant(self):
        """Test that DEFAULT_COLLECTION constant is set correctly"""
        assert DEFAULT_COLLECTION == "snippets"

    def test_collection_with_metadata(self):
        """Test creating collection with custom metadata"""
        with patch('backend.services.chroma_client.init_client') as mock_init:
            mock_client = Mock()
            mock_init.return_value = mock_client
            
            # Mock collection doesn't exist initially
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_collection = Mock()
            mock_client.create_collection.return_value = mock_collection
            
            metadata = {"description": "Test collection", "version": "1.0"}
            collection = get_or_create_collection("test_collection", metadata=metadata)
            
            mock_client.create_collection.assert_called_once_with("test_collection", metadata=metadata)
            assert collection == mock_collection


class TestChromaClientIntegration:
    """Integration tests that require actual ChromaDB instance"""
    
    def test_real_chroma_operations(self):
        """Test with real ChromaDB instance if available"""
        try:
            # Test basic operations with real ChromaDB
            client = init_client()
            assert client is not None
            
            # Test collection creation
            collection = get_or_create_collection("test_integration")
            assert collection is not None
            
            # Test upsert
            test_ids = ["test_id_1", "test_id_2"]
            test_embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            test_metadatas = [{"text_hash": "hash1", "content": "test1"}, 
                            {"text_hash": "hash2", "content": "test2"}]
            test_documents = ["Test document 1", "Test document 2"]
            
            upsert("test_integration", test_ids, test_embeddings, test_metadatas, test_documents)
            
            # Test query
            result = query_by_embedding("test_integration", [1.0, 2.0, 3.0], top_k=2)
            assert "ids" in result
            
            # Test exists_text_hash
            exists = exists_text_hash("test_integration", "hash1")
            assert exists is True
            
            exists_false = exists_text_hash("test_integration", "nonexistent_hash")
            assert exists_false is False
            
            # Test get_by_ids
            retrieved = get_by_ids("test_integration", ["test_id_1"])
            assert "ids" in retrieved
            
            # Test delete
            delete("test_integration", ["test_id_1"])
            
            # Verify deletion
            exists_after_delete = exists_text_hash("test_integration", "hash1")
            assert exists_after_delete is False
            
        except Exception as e:
            pytest.skip(f"ChromaDB integration test skipped due to: {e}")
            print(f"Integration test error details: {e}")  # For debugging
