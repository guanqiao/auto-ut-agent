"""Enhanced Code Indexer with FAISS/Chroma support.

This module provides optimized code indexing with:
- FAISS integration for fast similarity search
- Chroma integration for persistent vector storage
- Incremental indexing with change detection
- Index statistics and monitoring
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Statistics for code index."""
    total_files: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    index_size_mb: float = 0.0
    last_indexed: Optional[str] = None
    indexing_time_seconds: float = 0.0
    average_chunk_size: float = 0.0
    language_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "index_size_mb": round(self.index_size_mb, 2),
            "last_indexed": self.last_indexed,
            "indexing_time_seconds": round(self.indexing_time_seconds, 2),
            "average_chunk_size": round(self.average_chunk_size, 2),
            "language_distribution": self.language_distribution,
        }


class EnhancedCodeIndexer:
    """Enhanced code indexer with optimized vector storage."""
    
    def __init__(
        self,
        project_path: str,
        embedding_model=None,
        storage_backend: str = "sqlite",
        index_path: str = ".pyutagent/code_index"
    ):
        """Initialize enhanced code indexer.
        
        Args:
            project_path: Path to the project
            embedding_model: Model for generating embeddings
            storage_backend: Storage backend ("sqlite", "faiss", "chroma")
            index_path: Path for index storage
        """
        self.project_path = Path(project_path)
        self.embedding_model = embedding_model
        self.storage_backend = storage_backend
        self.index_path = self.project_path / index_path
        
        self._vector_store = None
        self._file_hashes: Dict[str, str] = {}
        self._stats = IndexStats()
        
        self._init_vector_store()
    
    def _init_vector_store(self):
        """Initialize vector store based on backend."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        if self.storage_backend == "faiss":
            self._init_faiss()
        elif self.storage_backend == "chroma":
            self._init_chroma()
        else:
            self._init_sqlite()
    
    def _init_faiss(self):
        """Initialize FAISS vector store."""
        try:
            import faiss
            import numpy as np
            
            self._faiss_index = None
            self._faiss_metadata: List[Dict[str, Any]] = []
            self._faiss_dimension = 384
            
            faiss_path = self.index_path / "faiss.index"
            if faiss_path.exists():
                self._faiss_index = faiss.read_index(str(faiss_path))
                logger.info(f"[EnhancedIndexer] Loaded FAISS index with {self._faiss_index.ntotal} vectors")
            else:
                self._faiss_index = faiss.IndexFlatIP(self._faiss_dimension)
                logger.info("[EnhancedIndexer] Created new FAISS index")
            
            metadata_path = self.index_path / "faiss_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self._faiss_metadata = json.load(f)
            
            self._vector_store = "faiss"
            
        except ImportError:
            logger.warning("[EnhancedIndexer] FAISS not available, falling back to sqlite")
            self._init_sqlite()
    
    def _init_chroma(self):
        """Initialize Chroma vector store."""
        try:
            import chromadb
            
            chroma_path = self.index_path / "chroma"
            self._chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="code_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            
            self._vector_store = "chroma"
            logger.info(f"[EnhancedIndexer] Chroma collection has {self._chroma_collection.count()} documents")
            
        except ImportError:
            logger.warning("[EnhancedIndexer] Chroma not available, falling back to sqlite")
            self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite vector store."""
        import sqlite3
        
        db_path = self.index_path / "code_index.db"
        self._sqlite_conn = sqlite3.connect(str(db_path))
        
        self._sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                content TEXT,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT
            )
        """)
        
        self._sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path)
        """)
        
        self._sqlite_conn.commit()
        self._vector_store = "sqlite"
        logger.info("[EnhancedIndexer] SQLite vector store initialized")
    
    async def index_project(
        self,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        incremental: bool = True,
        progress_callback: Optional[callable] = None
    ) -> IndexStats:
        """Index project files.
        
        Args:
            file_patterns: File patterns to include
            exclude_patterns: Patterns to exclude
            incremental: Only index changed files
            progress_callback: Progress callback
            
        Returns:
            Index statistics
        """
        start_time = time.time()
        
        if file_patterns is None:
            file_patterns = ["**/*.java", "**/*.py", "**/*.ts", "**/*.js"]
        
        if exclude_patterns is None:
            exclude_patterns = [
                "**/test/**", "**/tests/**", "**/__pycache__/**",
                "**/node_modules/**", "**/target/**", "**/build/**"
            ]
        
        files_to_index = []
        for pattern in file_patterns:
            for file_path in self.project_path.glob(pattern):
                if any(file_path.match(ex) for ex in exclude_patterns):
                    continue
                
                if incremental and not self._has_file_changed(file_path):
                    continue
                
                files_to_index.append(file_path)
        
        logger.info(f"[EnhancedIndexer] Indexing {len(files_to_index)} files")
        
        total_chunks = 0
        total_tokens = 0
        language_dist: Dict[str, int] = {}
        
        for i, file_path in enumerate(files_to_index):
            try:
                chunks, tokens = await self._index_file(file_path)
                total_chunks += chunks
                total_tokens += tokens
                
                lang = self._detect_language(file_path)
                language_dist[lang] = language_dist.get(lang, 0) + chunks
                
                if progress_callback:
                    progress_callback({
                        "file": str(file_path),
                        "current": i + 1,
                        "total": len(files_to_index),
                        "chunks": chunks,
                    })
                
            except Exception as e:
                logger.warning(f"[EnhancedIndexer] Failed to index {file_path}: {e}")
        
        self._save_index()
        
        self._stats.total_files = len(files_to_index)
        self._stats.total_chunks = total_chunks
        self._stats.total_tokens = total_tokens
        self._stats.indexing_time_seconds = time.time() - start_time
        self._stats.average_chunk_size = total_tokens / total_chunks if total_chunks > 0 else 0
        self._stats.language_distribution = language_dist
        self._stats.last_indexed = datetime.now().isoformat()
        
        self._update_index_size()
        
        logger.info(f"[EnhancedIndexer] Indexed {total_chunks} chunks in {self._stats.indexing_time_seconds:.2f}s")
        
        return self._stats
    
    async def _index_file(self, file_path: Path) -> Tuple[int, int]:
        """Index a single file."""
        content = file_path.read_text(encoding="utf-8")
        
        chunks = self._chunk_content(content, str(file_path))
        
        if self.embedding_model:
            embeddings = await self._get_embeddings([c["content"] for c in chunks])
        else:
            embeddings = [[0.0] * 384 for _ in chunks]
        
        for chunk, embedding in zip(chunks, embeddings):
            self._add_to_store(chunk, embedding)
        
        self._file_hashes[str(file_path)] = self._compute_hash(content)
        
        return len(chunks), sum(len(c["content"].split()) for c in chunks)
    
    def _chunk_content(
        self,
        content: str,
        file_path: str,
        max_chunk_size: int = 500
    ) -> List[Dict[str, Any]]:
        """Chunk content into smaller pieces."""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            line_size = len(line.split())
            
            if current_size + line_size > max_chunk_size and current_chunk:
                chunks.append({
                    "content": '\n'.join(current_chunk),
                    "file_path": file_path,
                    "start_line": i - len(current_chunk) + 1,
                    "end_line": i,
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_size
        
        if current_chunk:
            chunks.append({
                "content": '\n'.join(current_chunk),
                "file_path": file_path,
                "start_line": len(lines) - len(current_chunk) + 1,
                "end_line": len(lines),
            })
        
        return chunks
    
    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        if not self.embedding_model:
            return [[0.0] * 384 for _ in texts]
        
        if hasattr(self.embedding_model, 'encode'):
            if asyncio.iscoroutinefunction(self.embedding_model.encode):
                embeddings = await self.embedding_model.encode(texts)
            else:
                embeddings = self.embedding_model.encode(texts)
            
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            return list(embeddings)
        
        return [[0.0] * 384 for _ in texts]
    
    def _add_to_store(self, chunk: Dict[str, Any], embedding: List[float]):
        """Add chunk to vector store."""
        import uuid
        
        chunk_id = str(uuid.uuid4())
        
        if self._vector_store == "faiss":
            self._add_to_faiss(chunk_id, chunk, embedding)
        elif self._vector_store == "chroma":
            self._add_to_chroma(chunk_id, chunk, embedding)
        else:
            self._add_to_sqlite(chunk_id, chunk, embedding)
    
    def _add_to_faiss(self, chunk_id: str, chunk: Dict[str, Any], embedding: List[float]):
        """Add to FAISS index."""
        import numpy as np
        
        vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        self._faiss_index.add(vector)
        
        self._faiss_metadata.append({
            "id": chunk_id,
            "file_path": chunk["file_path"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "content_preview": chunk["content"][:200],
        })
    
    def _add_to_chroma(self, chunk_id: str, chunk: Dict[str, Any], embedding: List[float]):
        """Add to Chroma collection."""
        self._chroma_collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk["content"]],
            metadatas=[{
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
            }]
        )
    
    def _add_to_sqlite(self, chunk_id: str, chunk: Dict[str, Any], embedding: List[float]):
        """Add to SQLite store."""
        import pickle
        
        self._sqlite_conn.execute(
            """INSERT OR REPLACE INTO chunks 
               (id, file_path, content, embedding, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                chunk_id,
                chunk["file_path"],
                chunk["content"],
                pickle.dumps(embedding),
                json.dumps({
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                }),
                datetime.now().isoformat()
            )
        )
        self._sqlite_conn.commit()
    
    async def search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar code."""
        if not self.embedding_model:
            return []
        
        query_embedding = (await self._get_embeddings([query]))[0]
        
        if self._vector_store == "faiss":
            return self._search_faiss(query_embedding, k)
        elif self._vector_store == "chroma":
            return self._search_chroma(query_embedding, k, filter_dict)
        else:
            return self._search_sqlite(query_embedding, k)
    
    def _search_faiss(self, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Search FAISS index."""
        import numpy as np
        
        vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vector)
        
        distances, indices = self._faiss_index.search(vector, min(k, self._faiss_index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._faiss_metadata):
                meta = self._faiss_metadata[idx]
                results.append({
                    "score": float(dist),
                    "file_path": meta["file_path"],
                    "start_line": meta["start_line"],
                    "end_line": meta["end_line"],
                    "content_preview": meta["content_preview"],
                })
        
        return results
    
    def _search_chroma(
        self,
        query_embedding: List[float],
        k: int,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search Chroma collection."""
        where = filter_dict if filter_dict else None
        
        results = self._chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        for i, doc in enumerate(results["documents"][0]):
            formatted.append({
                "content": doc,
                "score": 1 - results["distances"][0][i],
                "file_path": results["metadatas"][0][i].get("file_path"),
                "start_line": results["metadatas"][0][i].get("start_line"),
                "end_line": results["metadatas"][0][i].get("end_line"),
            })
        
        return formatted
    
    def _search_sqlite(self, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Search SQLite store."""
        import pickle
        
        cursor = self._sqlite_conn.execute(
            "SELECT id, file_path, content, embedding, metadata FROM chunks"
        )
        
        results = []
        for row in cursor:
            chunk_id, file_path, content, embedding_blob, metadata = row
            embedding = pickle.loads(embedding_blob)
            
            score = self._cosine_similarity(query_embedding, embedding)
            
            results.append({
                "content": content,
                "score": score,
                "file_path": file_path,
                **json.loads(metadata)
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        import math
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def _has_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed."""
        current_hash = self._compute_hash(file_path.read_bytes())
        stored_hash = self._file_hashes.get(str(file_path))
        return current_hash != stored_hash
    
    def _compute_hash(self, content: bytes) -> str:
        """Compute content hash."""
        return hashlib.md5(content).hexdigest()
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect language from extension."""
        ext_map = {
            ".java": "java",
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
        }
        return ext_map.get(file_path.suffix, "unknown")
    
    def _save_index(self):
        """Save index to disk."""
        if self._vector_store == "faiss":
            import faiss
            faiss.write_index(self._faiss_index, str(self.index_path / "faiss.index"))
            with open(self.index_path / "faiss_metadata.json", 'w') as f:
                json.dump(self._faiss_metadata, f)
        
        with open(self.index_path / "file_hashes.json", 'w') as f:
            json.dump(self._file_hashes, f)
        
        with open(self.index_path / "stats.json", 'w') as f:
            json.dump(self._stats.to_dict(), f, indent=2)
    
    def _update_index_size(self):
        """Update index size statistic."""
        total_size = 0
        for path in self.index_path.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        
        self._stats.index_size_mb = total_size / (1024 * 1024)
    
    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        return self._stats
    
    def clear_index(self):
        """Clear the index."""
        import shutil
        
        if self._vector_store == "chroma":
            self._chroma_client.delete_collection("code_chunks")
        elif self._vector_store == "sqlite":
            self._sqlite_conn.execute("DELETE FROM chunks")
            self._sqlite_conn.commit()
        
        self._file_hashes = {}
        self._stats = IndexStats()
