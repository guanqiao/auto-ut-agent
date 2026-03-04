"""Code indexing module for semantic code search and retrieval.

This module provides:
- CodeChunker: Intelligent code splitting into chunks
- CodeIndexer: Building and maintaining code indices
- ContextAssembler: Assembling relevant context for LLM prompts
"""

from .code_chunker import (
    ChunkType,
    ChunkStrategy,
    CodeChunk,
    ChunkingConfig,
    CodeChunker,
)
from .code_indexer import (
    IndexState,
    IndexConfig,
    CodeIndexer,
)
from .context_assembler import (
    ContextStrategy,
    ContextWindow,
    AssemblerConfig,
    ContextAssembler,
)

__all__ = [
    "ChunkType",
    "ChunkStrategy",
    "CodeChunk",
    "ChunkingConfig",
    "CodeChunker",
    "IndexState",
    "IndexConfig",
    "CodeIndexer",
    "ContextStrategy",
    "ContextWindow",
    "AssemblerConfig",
    "ContextAssembler",
]
