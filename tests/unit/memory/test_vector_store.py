"""Tests for sqlite-vec vector store."""

import pytest
import tempfile
import os
from pathlib import Path


class TestSQLiteVecStore:
    """Test suite for SQLiteVecStore."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_vec.db"
            yield str(db_path)
    
    def test_init_creates_tables(self, temp_db):
        """Test that initialization creates necessary tables."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        store = SQLiteVecStore(temp_db, dimension=384)
        store.close()
        
        # Check that the database file was created
        assert os.path.exists(temp_db)
    
    def test_add_single_document(self, temp_db):
        """Test adding a single document."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        store = SQLiteVecStore(temp_db, dimension=384)
        text = "public void testAdd() { assertEquals(2, calculator.add(1, 1)); }"
        embedding = [0.1] * 384
        metadata = {"source": "CalculatorTest.java", "method": "testAdd"}
        
        store.add([text], [embedding], [metadata])
        
        # Search for the document
        results = store.search(embedding, k=1)
        store.close()
        
        assert len(results) == 1
        assert results[0][0] == text
        assert results[0][2]["source"] == "CalculatorTest.java"
    
    def test_add_multiple_documents(self, temp_db):
        """Test adding multiple documents."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        store = SQLiteVecStore(temp_db, dimension=384)
        texts = [
            "public void testAdd() { }",
            "public void testSubtract() { }",
            "public void testMultiply() { }"
        ]
        embeddings = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ]
        metadatas = [
            {"method": "testAdd"},
            {"method": "testSubtract"},
            {"method": "testMultiply"}
        ]
        
        store.add(texts, embeddings, metadatas)
        
        # Search with first embedding
        results = store.search(embeddings[0], k=3)
        store.close()
        
        assert len(results) == 3
        # First result should be the most similar
        assert results[0][0] == texts[0]
    
    def test_search_returns_k_results(self, temp_db):
        """Test that search returns exactly k results."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        store = SQLiteVecStore(temp_db, dimension=384)
        
        # Add 5 documents
        for i in range(5):
            store.add(
                [f"test {i}"],
                [[i * 0.01] * 384],
                [{"id": i}]
            )
        
        # Search for 3 results
        query = [0.0] * 384
        results = store.search(query, k=3)
        store.close()
        
        assert len(results) == 3
    
    def test_search_similarity_ordering(self, temp_db):
        """Test that results are ordered by similarity."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        store = SQLiteVecStore(temp_db, dimension=384)
        
        # Add documents with different embeddings
        texts = ["exact match", "close match", "different"]
        embeddings = [
            [0.5] * 384,
            [0.4] * 384,
            [0.1] * 384
        ]
        
        for text, emb in zip(texts, embeddings):
            store.add([text], [emb], [{}])
        
        # Search with query close to first document
        query = [0.5] * 384
        results = store.search(query, k=3)
        store.close()
        
        # First result should be the exact match
        assert results[0][0] == "exact match"
    
    def test_persistence(self, temp_db):
        """Test that data persists across store instances."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        # Create store and add data
        store1 = SQLiteVecStore(temp_db, dimension=384)
        text = "persistent test"
        embedding = [0.5] * 384
        store1.add([text], [embedding], [{"key": "value"}])
        store1.close()
        
        # Create new store instance with same database
        store2 = SQLiteVecStore(temp_db, dimension=384)
        results = store2.search(embedding, k=1)
        store2.close()
        
        assert len(results) == 1
        assert results[0][0] == text
    
    def test_empty_search(self, temp_db):
        """Test searching in empty store."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        store = SQLiteVecStore(temp_db, dimension=384)
        query = [0.0] * 384
        results = store.search(query, k=5)
        store.close()
        
        assert len(results) == 0
    
    def test_different_dimensions(self, temp_db):
        """Test creating stores with different dimensions."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        db_256 = temp_db + "_256"
        db_512 = temp_db + "_512"
        
        store_256 = SQLiteVecStore(db_256, dimension=256)
        store_512 = SQLiteVecStore(db_512, dimension=512)
        
        store_256.add(["test"], [[0.1] * 256], [{}])
        store_512.add(["test"], [[0.1] * 512], [{}])
        
        results_256 = store_256.search([0.1] * 256, k=1)
        results_512 = store_512.search([0.1] * 512, k=1)
        
        store_256.close()
        store_512.close()
        
        assert len(results_256) == 1
        assert len(results_512) == 1
    
    def test_count(self, temp_db):
        """Test document count."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        store = SQLiteVecStore(temp_db, dimension=384)
        
        assert store.count() == 0
        
        store.add(["test1"], [[0.1] * 384], [{}])
        assert store.count() == 1
        
        store.add(["test2", "test3"], [[0.2] * 384, [0.3] * 384], [{}, {}])
        assert store.count() == 3
        
        store.close()
    
    def test_clear(self, temp_db):
        """Test clearing all data."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        store = SQLiteVecStore(temp_db, dimension=384)
        
        store.add(["test1", "test2"], [[0.1] * 384, [0.2] * 384], [{}, {}])
        assert store.count() == 2
        
        store.clear()
        assert store.count() == 0
        
        store.close()
    
    def test_context_manager(self, temp_db):
        """Test using store as context manager."""
        from pyutagent.memory.vector_store import SQLiteVecStore
        
        with SQLiteVecStore(temp_db, dimension=384) as store:
            store.add(["test"], [[0.1] * 384], [{}])
            assert store.count() == 1
        
        # After exiting context, connection should be closed
        # But we can create a new connection
        with SQLiteVecStore(temp_db, dimension=384) as store2:
            assert store2.count() == 1
