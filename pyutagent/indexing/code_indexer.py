"""Code indexer module for building and maintaining code indices.

This module provides code indexing capabilities using vector embeddings
for semantic search and retrieval.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio

from .code_chunker import CodeChunker, CodeChunk, ChunkingConfig, ChunkType

logger = logging.getLogger(__name__)


@dataclass
class IndexState:
    """State of the code index."""
    project_path: str
    total_files: int = 0
    total_chunks: int = 0
    last_indexed: Optional[str] = None
    file_hashes: Dict[str, str] = field(default_factory=dict)
    index_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_path": self.project_path,
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "last_indexed": self.last_indexed,
            "file_hashes": self.file_hashes,
            "index_version": self.index_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexState":
        return cls(
            project_path=data.get("project_path", ""),
            total_files=data.get("total_files", 0),
            total_chunks=data.get("total_chunks", 0),
            last_indexed=data.get("last_indexed"),
            file_hashes=data.get("file_hashes", {}),
            index_version=data.get("index_version", "1.0"),
        )


@dataclass
class IndexConfig:
    """Configuration for code indexing."""
    chunking_config: Optional[ChunkingConfig] = None
    embedding_dimension: int = 384
    batch_size: int = 100
    max_index_size: int = 100000
    enable_incremental: bool = True
    persist_index: bool = True
    index_path: str = ".pyutagent/code_index"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunking_config": self.chunking_config.to_dict() if self.chunking_config else None,
            "embedding_dimension": self.embedding_dimension,
            "batch_size": self.batch_size,
            "max_index_size": self.max_index_size,
            "enable_incremental": self.enable_incremental,
            "persist_index": self.persist_index,
            "index_path": self.index_path,
        }


class CodeIndexer:
    """Indexer for building and maintaining code indices."""
    
    def __init__(
        self,
        project_path: str,
        vector_store=None,
        embedding_model=None,
        config: Optional[IndexConfig] = None
    ):
        """Initialize code indexer.
        
        Args:
            project_path: Path to the project
            vector_store: Vector store for embeddings
            embedding_model: Model for generating embeddings
            config: Indexing configuration
        """
        self.project_path = Path(project_path)
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config or IndexConfig()
        
        self._chunker = CodeChunker(self.config.chunking_config)
        self._state = IndexState(project_path=str(self.project_path))
        self._index_path = self.project_path / self.config.index_path
        
        self._load_state()
    
    def _load_state(self):
        """Load index state from disk."""
        if not self.config.persist_index:
            return
        
        state_file = self._index_path / "index_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._state = IndexState.from_dict(data)
                logger.info(f"[CodeIndexer] Loaded state: {self._state.total_chunks} chunks")
            except Exception as e:
                logger.warning(f"[CodeIndexer] Failed to load state: {e}")
    
    def _save_state(self):
        """Save index state to disk."""
        if not self.config.persist_index:
            return
        
        self._index_path.mkdir(parents=True, exist_ok=True)
        state_file = self._index_path / "index_state.json"
        
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(self._state.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"[CodeIndexer] Failed to save state: {e}")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file content for change detection."""
        try:
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def _has_file_changed(self, file_path: Path) -> bool:
        """Check if file has changed since last indexing."""
        if not self.config.enable_incremental:
            return True
        
        current_hash = self._compute_file_hash(file_path)
        stored_hash = self._state.file_hashes.get(str(file_path))
        
        return current_hash != stored_hash
    
    async def index_project(
        self,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Index all files in the project.
        
        Args:
            file_patterns: File patterns to include
            exclude_patterns: Patterns to exclude
            progress_callback: Optional callback for progress updates
            
        Returns:
            Indexing results
        """
        logger.info(f"[CodeIndexer] Starting project indexing: {self.project_path}")
        
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
                if self._has_file_changed(file_path):
                    files_to_index.append(file_path)
        
        total_files = len(files_to_index)
        total_chunks = 0
        errors = []
        
        logger.info(f"[CodeIndexer] Found {total_files} files to index")
        
        for i, file_path in enumerate(files_to_index):
            try:
                chunks = self._chunker.chunk_file(str(file_path))
                
                if self.vector_store and self.embedding_model:
                    await self._index_chunks(chunks)
                
                self._state.file_hashes[str(file_path)] = self._compute_file_hash(file_path)
                total_chunks += len(chunks)
                
                if progress_callback:
                    progress_callback({
                        "file": str(file_path),
                        "current": i + 1,
                        "total": total_files,
                        "chunks": len(chunks),
                    })
                
            except Exception as e:
                errors.append({"file": str(file_path), "error": str(e)})
                logger.warning(f"[CodeIndexer] Failed to index {file_path}: {e}")
        
        self._state.total_files = total_files
        self._state.total_chunks = total_chunks
        self._state.last_indexed = datetime.now().isoformat()
        
        self._save_state()
        
        result = {
            "success": True,
            "total_files": total_files,
            "total_chunks": total_chunks,
            "errors": errors,
        }
        
        logger.info(f"[CodeIndexer] Indexing complete: {total_chunks} chunks from {total_files} files")
        return result
    
    async def _index_chunks(self, chunks: List[CodeChunk]):
        """Index chunks with embeddings."""
        if not chunks:
            return
        
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]
        
        if hasattr(self.embedding_model, 'encode'):
            if asyncio.iscoroutinefunction(self.embedding_model.encode):
                embeddings = await self.embedding_model.encode(texts)
            else:
                embeddings = self.embedding_model.encode(texts)
        else:
            embeddings = [[0.0] * self.config.embedding_dimension for _ in texts]
        
        if self.vector_store:
            self.vector_store.add(texts, embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings, metadatas)
    
    async def index_file(self, file_path: str) -> Dict[str, Any]:
        """Index a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Indexing results
        """
        path = Path(file_path)
        
        if not path.exists():
            return {"success": False, "error": "File not found"}
        
        try:
            chunks = self._chunker.chunk_file(str(path))
            
            if self.vector_store:
                self.vector_store.delete(str(path))
            
            if self.vector_store and self.embedding_model:
                await self._index_chunks(chunks)
            
            self._state.file_hashes[str(path)] = self._compute_file_hash(path)
            self._save_state()
            
            return {
                "success": True,
                "file": str(path),
                "chunks": len(chunks),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def remove_file(self, file_path: str) -> Dict[str, Any]:
        """Remove a file from the index.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Removal results
        """
        path = Path(file_path)
        
        try:
            if self.vector_store:
                self.vector_store.delete(str(path))
            
            if str(path) in self._state.file_hashes:
                del self._state.file_hashes[str(path)]
            
            self._save_state()
            
            return {"success": True, "file": str(path)}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def search(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search the code index.
        
        Args:
            query: Search query
            k: Number of results
            filter_dict: Optional filters
            
        Returns:
            List of search results
        """
        if not self.vector_store or not self.embedding_model:
            return []
        
        if hasattr(self.embedding_model, 'encode'):
            if asyncio.iscoroutinefunction(self.embedding_model.encode):
                query_embedding = await self.embedding_model.encode([query])
            else:
                query_embedding = self.embedding_model.encode([query])
        else:
            return []
        
        results = self.vector_store.search(
            query_embedding.tolist()[0] if hasattr(query_embedding, 'tolist') else query_embedding[0],
            k=k
        )
        
        formatted_results = []
        for content, distance, metadata in results:
            result = {
                "content": content,
                "score": 1.0 - distance,
                "distance": distance,
                **metadata,
            }
            
            if filter_dict:
                if not all(result.get(k) == v for k, v in filter_dict.items()):
                    continue
            
            formatted_results.append(result)
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "project_path": str(self.project_path),
            "total_files": self._state.total_files,
            "total_chunks": self._state.total_chunks,
            "last_indexed": self._state.last_indexed,
            "index_size": len(self._state.file_hashes),
        }
    
    def clear_index(self):
        """Clear the entire index."""
        if self.vector_store:
            self.vector_store.clear()
        
        self._state = IndexState(project_path=str(self.project_path))
        self._save_state()
        
        logger.info("[CodeIndexer] Index cleared")
