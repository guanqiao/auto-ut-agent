"""Context assembler module for assembling relevant code context.

This module provides context assembly capabilities for retrieving
and organizing relevant code context for LLM prompts.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncio

from .code_chunker import CodeChunk, ChunkType

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Strategies for context assembly."""
    RELEVANCE = "relevance"
    PROXIMITY = "proximity"
    HIERARCHICAL = "hierarchical"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ContextWindow:
    """A window of context for LLM input."""
    content: str
    total_tokens: int
    sources: List[str]
    chunks: List[Dict[str, Any]]
    strategy: ContextStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "total_tokens": self.total_tokens,
            "sources": self.sources,
            "chunks": self.chunks,
            "strategy": self.strategy.value,
            "metadata": self.metadata,
        }


@dataclass
class AssemblerConfig:
    """Configuration for context assembly."""
    max_tokens: int = 10000
    max_chunks: int = 50
    overlap_tokens: int = 100
    include_imports: bool = True
    include_signatures: bool = True
    include_file_headers: bool = True
    strategy: ContextStrategy = ContextStrategy.RELEVANCE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "max_chunks": self.max_chunks,
            "overlap_tokens": self.overlap_tokens,
            "include_imports": self.include_imports,
            "include_signatures": self.include_signatures,
            "include_file_headers": self.include_file_headers,
            "strategy": self.strategy.value,
        }


class ContextAssembler:
    """Assembler for building relevant code context."""
    
    def __init__(
        self,
        code_indexer=None,
        config: Optional[AssemblerConfig] = None
    ):
        """Initialize context assembler.
        
        Args:
            code_indexer: Code indexer for retrieval
            config: Assembly configuration
        """
        self.code_indexer = code_indexer
        self.config = config or AssemblerConfig()
    
    async def assemble_context(
        self,
        query: str,
        target_file: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ContextWindow:
        """Assemble relevant context for a query.
        
        Args:
            query: The query or task description
            target_file: Optional target file path
            additional_context: Optional additional context
            
        Returns:
            ContextWindow with assembled context
        """
        if self.config.strategy == ContextStrategy.RELEVANCE:
            return await self._assemble_by_relevance(query, target_file)
        elif self.config.strategy == ContextStrategy.PROXIMITY:
            return await self._assemble_by_proximity(query, target_file)
        elif self.config.strategy == ContextStrategy.HIERARCHICAL:
            return await self._assemble_hierarchical(query, target_file)
        else:
            return await self._assemble_comprehensive(query, target_file)
    
    async def _assemble_by_relevance(
        self,
        query: str,
        target_file: Optional[str] = None
    ) -> ContextWindow:
        """Assemble context based on relevance to query."""
        chunks = []
        sources = set()
        
        if self.code_indexer:
            search_results = await self.code_indexer.search(
                query,
                k=self.config.max_chunks
            )
            
            for result in search_results:
                chunks.append({
                    "content": result.get("content", ""),
                    "file_path": result.get("file_path", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "name": result.get("name", ""),
                    "score": result.get("score", 0),
                })
                if result.get("file_path"):
                    sources.add(result["file_path"])
        
        if target_file:
            target_chunks = await self._get_file_chunks(target_file)
            for chunk in target_chunks:
                if chunk not in chunks:
                    chunks.insert(0, chunk)
                    sources.add(chunk.get("file_path", ""))
        
        chunks = self._prune_chunks(chunks)
        content = self._format_chunks(chunks)
        total_tokens = self._estimate_tokens(content)
        
        return ContextWindow(
            content=content,
            total_tokens=total_tokens,
            sources=list(sources),
            chunks=chunks,
            strategy=ContextStrategy.RELEVANCE,
        )
    
    async def _assemble_by_proximity(
        self,
        query: str,
        target_file: Optional[str] = None
    ) -> ContextWindow:
        """Assemble context based on code proximity."""
        chunks = []
        sources = set()
        
        if target_file:
            file_chunks = await self._get_file_chunks(target_file)
            chunks.extend(file_chunks)
            sources.add(target_file)
            
            related_files = await self._find_related_files(target_file)
            for related_file in related_files[:5]:
                related_chunks = await self._get_file_chunks(related_file)
                chunks.extend(related_chunks[:3])
                sources.add(related_file)
        
        if self.code_indexer and len(chunks) < self.config.max_chunks:
            search_results = await self.code_indexer.search(
                query,
                k=self.config.max_chunks - len(chunks)
            )
            for result in search_results:
                chunks.append({
                    "content": result.get("content", ""),
                    "file_path": result.get("file_path", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "name": result.get("name", ""),
                    "score": result.get("score", 0),
                })
        
        chunks = self._prune_chunks(chunks)
        content = self._format_chunks(chunks)
        total_tokens = self._estimate_tokens(content)
        
        return ContextWindow(
            content=content,
            total_tokens=total_tokens,
            sources=list(sources),
            chunks=chunks,
            strategy=ContextStrategy.PROXIMITY,
        )
    
    async def _assemble_hierarchical(
        self,
        query: str,
        target_file: Optional[str] = None
    ) -> ContextWindow:
        """Assemble context with hierarchical structure."""
        chunks = []
        sources = set()
        
        if self.code_indexer:
            class_results = await self.code_indexer.search(
                query,
                k=10,
                filter_dict={"chunk_type": "class"}
            )
            
            for result in class_results:
                chunks.append({
                    "content": result.get("content", ""),
                    "file_path": result.get("file_path", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "name": result.get("name", ""),
                    "score": result.get("score", 0),
                    "level": 1,
                })
                sources.add(result.get("file_path", ""))
            
            method_results = await self.code_indexer.search(
                query,
                k=20,
                filter_dict={"chunk_type": "method"}
            )
            
            for result in method_results:
                chunks.append({
                    "content": result.get("content", ""),
                    "file_path": result.get("file_path", ""),
                    "chunk_type": result.get("chunk_type", ""),
                    "name": result.get("name", ""),
                    "parent": result.get("parent", ""),
                    "score": result.get("score", 0),
                    "level": 2,
                })
        
        if target_file:
            target_chunks = await self._get_file_chunks(target_file)
            for chunk in target_chunks:
                chunk["level"] = 0
            chunks = target_chunks + chunks
            sources.add(target_file)
        
        chunks = self._prune_chunks(chunks)
        chunks = self._sort_hierarchical(chunks)
        content = self._format_chunks_hierarchical(chunks)
        total_tokens = self._estimate_tokens(content)
        
        return ContextWindow(
            content=content,
            total_tokens=total_tokens,
            sources=list(sources),
            chunks=chunks,
            strategy=ContextStrategy.HIERARCHICAL,
        )
    
    async def _assemble_comprehensive(
        self,
        query: str,
        target_file: Optional[str] = None
    ) -> ContextWindow:
        """Assemble comprehensive context."""
        chunks = []
        sources = set()
        
        if target_file:
            file_chunks = await self._get_file_chunks(target_file)
            chunks.extend(file_chunks)
            sources.add(target_file)
        
        if self.code_indexer:
            search_results = await self.code_indexer.search(
                query,
                k=self.config.max_chunks
            )
            
            for result in search_results:
                if result.get("file_path") not in sources:
                    chunks.append({
                        "content": result.get("content", ""),
                        "file_path": result.get("file_path", ""),
                        "chunk_type": result.get("chunk_type", ""),
                        "name": result.get("name", ""),
                        "score": result.get("score", 0),
                    })
                    sources.add(result.get("file_path", ""))
        
        chunks = self._prune_chunks(chunks)
        content = self._format_chunks(chunks)
        total_tokens = self._estimate_tokens(content)
        
        return ContextWindow(
            content=content,
            total_tokens=total_tokens,
            sources=list(sources),
            chunks=chunks,
            strategy=ContextStrategy.COMPREHENSIVE,
        )
    
    async def _get_file_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all chunks from a file."""
        if self.code_indexer:
            search_results = await self.code_indexer.search(
                f"file:{file_path}",
                k=100
            )
            
            return [
                {
                    "content": r.get("content", ""),
                    "file_path": r.get("file_path", ""),
                    "chunk_type": r.get("chunk_type", ""),
                    "name": r.get("name", ""),
                    "start_line": r.get("start_line", 0),
                    "end_line": r.get("end_line", 0),
                }
                for r in search_results
                if r.get("file_path") == file_path
            ]
        
        path = Path(file_path)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            return [{
                "content": content,
                "file_path": file_path,
                "chunk_type": "file",
                "name": path.stem,
            }]
        
        return []
    
    async def _find_related_files(self, file_path: str) -> List[str]:
        """Find files related to the given file."""
        related = []
        
        if not self.code_indexer:
            return related
        
        path = Path(file_path)
        file_name = path.stem
        
        search_results = await self.code_indexer.search(
            file_name,
            k=20
        )
        
        for result in search_results:
            result_path = result.get("file_path", "")
            if result_path and result_path != file_path:
                related.append(result_path)
        
        return list(dict.fromkeys(related))[:10]
    
    def _prune_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prune chunks to fit within token limit."""
        pruned = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk.get("content", ""))
            
            if total_tokens + chunk_tokens <= self.config.max_tokens:
                pruned.append(chunk)
                total_tokens += chunk_tokens
            
            if len(pruned) >= self.config.max_chunks:
                break
        
        return pruned
    
    def _sort_hierarchical(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort chunks hierarchically."""
        return sorted(chunks, key=lambda x: x.get("level", 0))
    
    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context string."""
        parts = []
        
        for chunk in chunks:
            file_path = chunk.get("file_path", "unknown")
            name = chunk.get("name", "")
            chunk_type = chunk.get("chunk_type", "")
            content = chunk.get("content", "")
            
            header = f"// File: {file_path}"
            if name:
                header += f" | {chunk_type}: {name}"
            
            parts.append(f"{header}\n{content}")
        
        return "\n\n".join(parts)
    
    def _format_chunks_hierarchical(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks with hierarchical structure."""
        parts = []
        current_level = -1
        
        for chunk in chunks:
            level = chunk.get("level", 0)
            file_path = chunk.get("file_path", "unknown")
            name = chunk.get("name", "")
            chunk_type = chunk.get("chunk_type", "")
            content = chunk.get("content", "")
            
            indent = "  " * level
            
            if level != current_level:
                parts.append(f"\n{'=' * 60}")
                current_level = level
            
            header = f"{indent}// [{chunk_type}] {name} ({file_path})"
            parts.append(f"{header}\n{indent}{content}")
        
        return "\n".join(parts)
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        words = len(content.split())
        return int(words * 1.3)
    
    async def get_context_for_task(
        self,
        task_description: str,
        target_files: Optional[List[str]] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Get context string for a specific task.
        
        Args:
            task_description: Description of the task
            target_files: Optional list of target files
            max_tokens: Optional max tokens override
            
        Returns:
            Context string
        """
        if max_tokens:
            original_max = self.config.max_tokens
            self.config.max_tokens = max_tokens
        
        try:
            target_file = target_files[0] if target_files else None
            context_window = await self.assemble_context(task_description, target_file)
            return context_window.content
        finally:
            if max_tokens:
                self.config.max_tokens = original_max
