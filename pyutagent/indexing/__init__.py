"""Code indexing module for semantic code search and retrieval.

This module provides:
- CodeChunker: Intelligent code splitting into chunks
- CodeIndexer: Building and maintaining code indices
- ContextAssembler: Assembling relevant context for LLM prompts
- CodebaseIndexer: Comprehensive codebase indexing with dependency graphs
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
from .codebase_indexer import (
    SymbolType,
    RelationType,
    CodeSymbol,
    SymbolRelation,
    FileIndex,
    CodebaseIndexState,
    IndexerConfig,
    DependencyGraph,
    JavaSymbolExtractor,
    CodebaseIndexer,
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
    # Codebase indexer exports
    "SymbolType",
    "RelationType",
    "CodeSymbol",
    "SymbolRelation",
    "FileIndex",
    "CodebaseIndexState",
    "IndexerConfig",
    "DependencyGraph",
    "JavaSymbolExtractor",
    "CodebaseIndexer",
]
