"""Semantic search service for natural language code search.

This module provides semantic search capabilities using:
- Codebase indexing for code structure understanding
- Vector similarity for semantic matching
- LLM-based query understanding
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal, QThread

from pyutagent.indexing.codebase_indexer import CodebaseIndexer, CodeSymbol, SymbolType
from pyutagent.memory.vector_store import SQLiteVecStore

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single semantic search result."""
    file_path: str
    content: str
    score: float
    start_line: int
    end_line: int
    symbol_name: str = ""
    symbol_type: str = ""
    context: str = ""  # Surrounding context for preview
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def relevance_percentage(self) -> int:
        """Get relevance as percentage (0-100)."""
        return int(self.score * 100)

    @property
    def file_name(self) -> str:
        """Get just the file name."""
        return Path(self.file_path).name

    @property
    def location(self) -> str:
        """Get human-readable location."""
        if self.start_line == self.end_line:
            return f"{self.file_name}:{self.start_line}"
        return f"{self.file_name}:{self.start_line}-{self.end_line}"


@dataclass
class SearchQuery:
    """Parsed search query with intent understanding."""
    original_query: str
    keywords: List[str] = field(default_factory=list)
    intent: str = ""  # e.g., "find_function", "find_class", "explain"
    language_filter: Optional[str] = None
    symbol_type_filter: Optional[str] = None


class SearchWorker(QThread):
    """Background worker for semantic search."""

    results_ready = pyqtSignal(list)  # List[SearchResult]
    progress_update = pyqtSignal(int, int)  # current, total
    search_error = pyqtSignal(str)

    def __init__(
        self,
        query: str,
        project_path: str,
        indexer: Optional[CodebaseIndexer] = None,
        embedding_model: Optional[Any] = None,
        max_results: int = 20
    ):
        super().__init__()
        self.query = query
        self.project_path = project_path
        self.indexer = indexer
        self.embedding_model = embedding_model
        self.max_results = max_results
        self._is_cancelled = False

    def cancel(self):
        """Cancel the search operation."""
        self._is_cancelled = True

    def run(self):
        """Execute the search."""
        try:
            service = SemanticSearchService(
                project_path=self.project_path,
                indexer=self.indexer,
                embedding_model=self.embedding_model
            )

            results = service.search(
                query=self.query,
                max_results=self.max_results,
                progress_callback=self._on_progress
            )

            if not self._is_cancelled:
                self.results_ready.emit(results)

        except Exception as e:
            logger.exception("Search worker failed")
            if not self._is_cancelled:
                self.search_error.emit(str(e))

    def _on_progress(self, current: int, total: int):
        """Handle progress update."""
        if not self._is_cancelled:
            self.progress_update.emit(current, total)


class SemanticSearchService(QObject):
    """Service for semantic code search.

    Features:
    - Natural language query understanding
    - Vector-based semantic similarity
    - Symbol-based search fallback
    - Result ranking and filtering
    - Context extraction for previews
    """

    search_started = pyqtSignal()
    search_completed = pyqtSignal(list)  # List[SearchResult]
    search_error = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)  # current, total

    def __init__(
        self,
        project_path: str,
        indexer: Optional[CodebaseIndexer] = None,
        embedding_model: Optional[Any] = None,
        llm_client: Optional[Any] = None
    ):
        """Initialize the semantic search service.

        Args:
            project_path: Path to the project root
            indexer: Optional existing codebase indexer
            embedding_model: Optional embedding model for vector search
            llm_client: Optional LLM client for query enhancement
        """
        super().__init__()
        self.project_path = Path(project_path)
        self._indexer = indexer
        self._embedding_model = embedding_model
        self._llm_client = llm_client
        self._current_worker: Optional[SearchWorker] = None

    def _get_indexer(self) -> Optional[CodebaseIndexer]:
        """Get or create codebase indexer."""
        if self._indexer is None:
            try:
                self._indexer = CodebaseIndexer(
                    project_path=str(self.project_path),
                    embedding_model=self._embedding_model
                )
            except Exception as e:
                logger.warning(f"Failed to create indexer: {e}")
        return self._indexer

    def _parse_query(self, query: str) -> SearchQuery:
        """Parse and understand the search query.

        Args:
            query: Raw user query

        Returns:
            Parsed search query with intent
        """
        parsed = SearchQuery(original_query=query)

        # Extract keywords (remove common stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'and', 'but', 'or', 'yet', 'so',
            'if', 'because', 'although', 'though', 'while', 'where',
            'when', 'that', 'which', 'who', 'whom', 'whose', 'what',
            'this', 'these', 'those', 'i', 'me', 'my', 'myself', 'we',
            'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
            'find', 'search', 'look', 'get', 'show', 'display'
        }

        words = query.lower().split()
        parsed.keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Detect intent
        query_lower = query.lower()

        if any(kw in query_lower for kw in ['function', 'method', 'def', 'func']):
            parsed.intent = 'find_function'
            parsed.symbol_type_filter = 'method'
        elif any(kw in query_lower for kw in ['class', 'type', 'struct', 'interface']):
            parsed.intent = 'find_class'
            parsed.symbol_type_filter = 'class'
        elif any(kw in query_lower for kw in ['variable', 'field', 'property', 'attr']):
            parsed.intent = 'find_field'
            parsed.symbol_type_filter = 'field'
        elif any(kw in query_lower for kw in ['explain', 'how does', 'what does', 'describe']):
            parsed.intent = 'explain'
        elif any(kw in query_lower for kw in ['usage', 'use', 'call', 'invoke']):
            parsed.intent = 'find_usage'
        else:
            parsed.intent = 'general_search'

        # Detect language filter
        languages = ['java', 'python', 'javascript', 'typescript', 'go', 'rust', 'c++', 'cpp', 'c#', 'csharp']
        for lang in languages:
            if lang in query_lower:
                parsed.language_filter = lang.replace('csharp', 'cs').replace('cpp', 'c++')
                break

        return parsed

    def _extract_context(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        context_lines: int = 5
    ) -> str:
        """Extract surrounding context for a code snippet.

        Args:
            file_path: Path to the file
            start_line: Start line number (1-based)
            end_line: End line number (1-based)
            context_lines: Number of context lines to include

        Returns:
            Context string with line numbers
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return ""

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Calculate context range
            context_start = max(0, start_line - context_lines - 1)
            context_end = min(len(lines), end_line + context_lines)

            # Build context with line numbers
            context_parts = []
            for i in range(context_start, context_end):
                line_num = i + 1
                prefix = ">>> " if start_line <= line_num <= end_line else "    "
                context_parts.append(f"{prefix}{line_num:4d}: {lines[i].rstrip()}")

            return '\n'.join(context_parts)

        except Exception as e:
            logger.warning(f"Failed to extract context from {file_path}: {e}")
            return ""

    def _calculate_relevance_score(
        self,
        result: SearchResult,
        parsed_query: SearchQuery
    ) -> float:
        """Calculate relevance score for a search result.

        Args:
            result: Search result to score
            parsed_query: Parsed search query

        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = result.score

        # Boost exact keyword matches
        content_lower = result.content.lower()
        for keyword in parsed_query.keywords:
            if keyword in content_lower:
                score += 0.05
            if keyword in result.symbol_name.lower():
                score += 0.1

        # Boost symbol type matches
        if parsed_query.symbol_type_filter and result.symbol_type:
            if parsed_query.symbol_type_filter in result.symbol_type.lower():
                score += 0.1

        # Boost language matches
        if parsed_query.language_filter:
            file_ext = Path(result.file_path).suffix.lower()
            lang_ext_map = {
                'java': '.java',
                'python': '.py',
                'javascript': '.js',
                'typescript': '.ts',
                'go': '.go',
                'rust': '.rs',
                'c++': ['.cpp', '.hpp', '.cc', '.hh'],
                'cs': '.cs'
            }
            expected_ext = lang_ext_map.get(parsed_query.language_filter)
            if expected_ext:
                if isinstance(expected_ext, list):
                    if file_ext in expected_ext:
                        score += 0.05
                elif file_ext == expected_ext:
                    score += 0.05

        # Cap at 1.0
        return min(score, 1.0)

    def search(
        self,
        query: str,
        max_results: int = 20,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[SearchResult]:
        """Perform semantic search.

        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            progress_callback: Optional callback for progress updates

        Returns:
            List of search results sorted by relevance
        """
        if not query.strip():
            return []

        self.search_started.emit()

        # Parse query
        parsed_query = self._parse_query(query)
        logger.info(f"Parsed query: intent={parsed_query.intent}, "
                   f"keywords={parsed_query.keywords}, "
                   f"filters={parsed_query.symbol_type_filter}, {parsed_query.language_filter}")

        results: List[SearchResult] = []

        # Get indexer
        indexer = self._get_indexer()

        # Method 1: Vector-based semantic search (if available)
        if indexer and self._embedding_model:
            try:
                semantic_results = indexer.search_semantic(query, limit=max_results * 2)

                for i, sr in enumerate(semantic_results):
                    if progress_callback:
                        progress_callback(i + 1, len(semantic_results))

                    result = SearchResult(
                        file_path=sr.get('file_path', ''),
                        content=sr.get('content', ''),
                        score=sr.get('score', 0.0),
                        start_line=sr.get('start_line', 1),
                        end_line=sr.get('end_line', 1),
                        symbol_name=sr.get('name', ''),
                        symbol_type=sr.get('chunk_type', ''),
                        context=self._extract_context(
                            sr.get('file_path', ''),
                            sr.get('start_line', 1),
                            sr.get('end_line', 1)
                        ),
                        metadata=sr
                    )
                    result.score = self._calculate_relevance_score(result, parsed_query)
                    results.append(result)

            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")

        # Method 2: Symbol-based search as fallback/enhancement
        if indexer and len(results) < max_results:
            try:
                # Search symbols by name
                symbol_results = indexer.search_symbols(
                    query.replace(' ', '*'),
                    limit=max_results
                )

                existing_paths = {(r.file_path, r.start_line) for r in results}

                for symbol in symbol_results:
                    key = (symbol.file_path, symbol.start_line)
                    if key not in existing_paths:
                        # Get symbol content
                        content = self._get_symbol_content(symbol)

                        result = SearchResult(
                            file_path=symbol.file_path,
                            content=content,
                            score=0.6,  # Base score for symbol match
                            start_line=symbol.start_line,
                            end_line=symbol.end_line,
                            symbol_name=symbol.name,
                            symbol_type=symbol.symbol_type.value,
                            context=self._extract_context(
                                symbol.file_path,
                                symbol.start_line,
                                symbol.end_line
                            ),
                            metadata=symbol.to_dict()
                        )
                        result.score = self._calculate_relevance_score(result, parsed_query)
                        results.append(result)

            except Exception as e:
                logger.warning(f"Symbol search failed: {e}")

        # Method 3: Keyword-based text search as final fallback
        if len(results) < max_results // 2:
            try:
                keyword_results = self._keyword_search(query, max_results)
                existing_paths = {(r.file_path, r.start_line) for r in results}

                for result in keyword_results:
                    key = (result.file_path, result.start_line)
                    if key not in existing_paths:
                        results.append(result)

            except Exception as e:
                logger.warning(f"Keyword search failed: {e}")

        # Sort by relevance score
        results.sort(key=lambda r: r.score, reverse=True)

        # Take top results
        final_results = results[:max_results]

        self.search_completed.emit(final_results)
        return final_results

    def _get_symbol_content(self, symbol: CodeSymbol) -> str:
        """Get the content of a symbol from its file.

        Args:
            symbol: Code symbol

        Returns:
            Symbol content as string
        """
        try:
            path = Path(symbol.file_path)
            if not path.exists():
                return symbol.signature

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Extract symbol lines
            start_idx = max(0, symbol.start_line - 1)
            end_idx = min(len(lines), symbol.end_line)
            content = ''.join(lines[start_idx:end_idx])

            return content

        except Exception as e:
            logger.warning(f"Failed to get symbol content: {e}")
            return symbol.signature

    def _keyword_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Perform keyword-based text search.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of search results
        """
        results = []
        keywords = [w for w in query.lower().split() if len(w) > 2]

        if not keywords:
            return results

        # Search through project files
        include_patterns = ['**/*.java', '**/*.py', '**/*.ts', '**/*.js', '**/*.go', '**/*.rs']
        exclude_patterns = [
            '**/test/**', '**/tests/**', '**/__pycache__/**',
            '**/node_modules/**', '**/target/**', '**/build/**',
            '**/dist/**', '**/.git/**', '**/.idea/**', '**/.vscode/**'
        ]

        files_searched = 0
        for pattern in include_patterns:
            for file_path in self.project_path.glob(pattern):
                # Check exclude patterns
                if any(file_path.match(ex) for ex in exclude_patterns):
                    continue

                files_searched += 1
                if files_searched > 1000:  # Limit search scope
                    break

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.splitlines()

                    content_lower = content.lower()
                    match_count = sum(1 for kw in keywords if kw in content_lower)

                    if match_count > 0:
                        # Find first match location
                        for i, line in enumerate(lines):
                            line_lower = line.lower()
                            if any(kw in line_lower for kw in keywords):
                                score = min(0.5 + (match_count * 0.1), 0.9)
                                result = SearchResult(
                                    file_path=str(file_path),
                                    content=line.strip(),
                                    score=score,
                                    start_line=i + 1,
                                    end_line=i + 1,
                                    context=self._extract_context(
                                        str(file_path), i + 1, i + 1
                                    )
                                )
                                results.append(result)
                                break

                except Exception as e:
                    logger.debug(f"Failed to search file {file_path}: {e}")

                if len(results) >= max_results:
                    break

            if len(results) >= max_results:
                break

        return results[:max_results]

    def search_async(
        self,
        query: str,
        max_results: int = 20
    ) -> SearchWorker:
        """Start an asynchronous search.

        Args:
            query: Natural language search query
            max_results: Maximum number of results

        Returns:
            SearchWorker instance (already started)
        """
        # Cancel any existing search
        if self._current_worker and self._current_worker.isRunning():
            self._current_worker.cancel()
            self._current_worker.wait(1000)

        # Create and start new worker
        self._current_worker = SearchWorker(
            query=query,
            project_path=str(self.project_path),
            indexer=self._indexer,
            embedding_model=self._embedding_model,
            max_results=max_results
        )

        # Connect signals
        self._current_worker.results_ready.connect(self.search_completed.emit)
        self._current_worker.search_error.connect(self.search_error.emit)
        self._current_worker.progress_update.connect(self.progress_updated.emit)

        self._current_worker.start()
        self.search_started.emit()

        return self._current_worker

    def cancel_search(self):
        """Cancel any ongoing search."""
        if self._current_worker and self._current_worker.isRunning():
            self._current_worker.cancel()
            self._current_worker.wait(1000)

    def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query.

        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions

        Returns:
            List of suggestion strings
        """
        suggestions = []

        # Common code search patterns
        patterns = [
            f"find function {partial_query}",
            f"find class {partial_query}",
            f"how to use {partial_query}",
            f"explain {partial_query}",
            f"{partial_query} implementation",
            f"{partial_query} example",
        ]

        # Add symbol-based suggestions if indexer available
        indexer = self._get_indexer()
        if indexer:
            try:
                symbols = indexer.search_symbols(f"{partial_query}*", limit=limit)
                for symbol in symbols:
                    suggestions.append(f"@{symbol.name} - {symbol.symbol_type.value}")
            except Exception as e:
                logger.debug(f"Failed to get symbol suggestions: {e}")

        # Add pattern suggestions
        suggestions.extend(patterns)

        return suggestions[:limit]

    def index_project(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Index the project for semantic search.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Indexing results summary
        """
        indexer = self._get_indexer()
        if not indexer:
            return {"success": False, "error": "Failed to create indexer"}

        try:
            result = indexer.index_project(progress_callback)
            return result
        except Exception as e:
            logger.exception("Failed to index project")
            return {"success": False, "error": str(e)}

    def is_indexed(self) -> bool:
        """Check if the project has been indexed.

        Returns:
            True if project is indexed
        """
        indexer = self._get_indexer()
        if not indexer:
            return False

        stats = indexer.get_stats()
        return stats.get('total_files', 0) > 0

    def get_index_stats(self) -> Dict[str, Any]:
        """Get indexing statistics.

        Returns:
            Dictionary with index statistics
        """
        indexer = self._get_indexer()
        if not indexer:
            return {"indexed": False}

        return indexer.get_stats()
