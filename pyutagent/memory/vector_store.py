"""Vector store implementation using sqlite-vec (pure Python)."""

import sqlite3
import sqlite_vec
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path


class SQLiteVecStore:
    """Pure Python vector store using sqlite-vec.
    
    No C++ compilation required. Stores embeddings and supports
    similarity search using cosine distance.
    """
    
    def __init__(self, db_path: str, dimension: int = 384):
        """Initialize vector store.
        
        Args:
            db_path: Path to SQLite database file
            dimension: Dimension of embedding vectors
        """
        self.db_path = db_path
        self.dimension = dimension
        self._conn: Optional[sqlite3.Connection] = None
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect and initialize
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        if self._conn is not None:
            return
            
        # Connect and load extension
        self._conn = sqlite3.connect(self.db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        
        # Initialize tables
        self._init_tables()
    
    def _init_tables(self):
        """Create necessary tables if they don't exist."""
        # Vector table using sqlite-vec virtual table
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
                embedding float[{self.dimension}]
            )
        """)
        
        # Metadata table for storing text and other info
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS item_metadata (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                method TEXT,
                timestamp TEXT,
                extra_data TEXT
            )
        """)
        
        self._conn.commit()
    
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ):
        """Add documents with embeddings to the store.
        
        Args:
            texts: List of text content
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        for text, emb, meta in zip(texts, embeddings, metadatas):
            # Serialize embedding to bytes
            emb_bytes = sqlite_vec.serialize_float32(emb)
            
            # Insert into vector table
            cursor = self._conn.execute(
                "INSERT INTO vec_items(embedding) VALUES (?)",
                (emb_bytes,)
            )
            row_id = cursor.lastrowid
            
            # Insert metadata
            import json
            from datetime import datetime
            
            self._conn.execute(
                """
                INSERT INTO item_metadata 
                (id, content, source, method, timestamp, extra_data)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    text,
                    meta.get("source", ""),
                    meta.get("method", ""),
                    datetime.now().isoformat(),
                    json.dumps({k: v for k, v in meta.items() 
                               if k not in ["source", "method"]})
                )
            )
        
        self._conn.commit()
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of (content, distance, metadata) tuples
        """
        query_bytes = sqlite_vec.serialize_float32(query_embedding)
        
        # Use proper sqlite-vec KNN query syntax with k constraint
        results = self._conn.execute(
            """
            SELECT 
                m.content,
                v.distance,
                m.source,
                m.method,
                m.timestamp,
                m.extra_data
            FROM vec_items v
            LEFT JOIN item_metadata m ON v.rowid = m.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (query_bytes, k)
        ).fetchall()
        
        import json
        return [
            (
                content or "",
                float(distance) if distance is not None else 0.0,
                {
                    "source": source or "",
                    "method": method or "",
                    "timestamp": timestamp or "",
                    **json.loads(extra_data or "{}")
                }
            )
            for content, distance, source, method, timestamp, extra_data in results
        ]
    
    def delete(self, source: str):
        """Delete all documents from a specific source.
        
        Args:
            source: Source file path or identifier
        """
        # Get IDs to delete
        cursor = self._conn.execute(
            "SELECT id FROM item_metadata WHERE source = ?",
            (source,)
        )
        ids_to_delete = [row[0] for row in cursor.fetchall()]
        
        # Delete from vector table
        for id_ in ids_to_delete:
            self._conn.execute(
                "DELETE FROM vec_items WHERE rowid = ?",
                (id_,)
            )
        
        # Delete from metadata table
        self._conn.execute(
            "DELETE FROM item_metadata WHERE source = ?",
            (source,)
        )
        
        self._conn.commit()
    
    def clear(self):
        """Clear all data from the store."""
        self._conn.execute("DELETE FROM vec_items")
        self._conn.execute("DELETE FROM item_metadata")
        self._conn.commit()
    
    def count(self) -> int:
        """Get total number of documents."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM item_metadata")
        return cursor.fetchone()[0]
    
    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
