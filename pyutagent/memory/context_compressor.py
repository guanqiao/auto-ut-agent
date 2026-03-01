"""Intelligent context compression for large projects.

This module provides context management capabilities:
- Relevance scoring for code snippets
- Token-aware context building
- Intelligent compression strategies
- Dependency-aware context selection
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeSnippet:
    """A snippet of code with metadata."""
    file_path: str
    content: str
    start_line: int
    end_line: int
    language: str = "java"
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    token_count: int = 0
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = self._estimate_tokens()
    
    def _estimate_tokens(self) -> int:
        return len(self.content) // 4


@dataclass
class ContextConfig:
    """Configuration for context building."""
    max_tokens: int = 8000
    min_relevance_score: float = 0.3
    max_snippets: int = 20
    include_imports: bool = True
    include_signatures: bool = True
    include_dependencies: bool = True
    compression_ratio: float = 0.7
    priority_files: List[str] = field(default_factory=list)


@dataclass
class CompressedContext:
    """Result of context compression."""
    content: str
    total_tokens: int
    snippets_included: int
    snippets_excluded: int
    compression_ratio: float
    relevance_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RelevanceScorer:
    """Scores relevance of code snippets to a query."""
    
    def __init__(self):
        self._keyword_weights: Dict[str, float] = {}
        self._symbol_weights: Dict[str, float] = {}
    
    def compute_relevance(
        self,
        snippet: CodeSnippet,
        query: str,
        target_file: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute relevance score for a snippet.
        
        Args:
            snippet: The code snippet to score
            query: The query/prompt for context
            target_file: The target file being processed
            context: Additional context information
            
        Returns:
            Relevance score (0.0-1.0)
        """
        score = 0.0
        
        query_terms = set(self._extract_terms(query))
        snippet_terms = set(self._extract_terms(snippet.content))
        
        if query_terms and snippet_terms:
            overlap = len(query_terms & snippet_terms)
            coverage = overlap / len(query_terms) if query_terms else 0
            score += coverage * 0.3
        
        if target_file:
            if snippet.file_path == target_file:
                score += 0.4
            elif self._is_related_file(snippet.file_path, target_file):
                score += 0.2
        
        if context:
            referenced_symbols = context.get("referenced_symbols", [])
            if referenced_symbols:
                symbol_overlap = len(set(snippet.symbols) & set(referenced_symbols))
                symbol_coverage = symbol_overlap / len(referenced_symbols) if referenced_symbols else 0
                score += symbol_coverage * 0.2
        
        if snippet.dependencies:
            score += min(0.1, len(snippet.dependencies) * 0.02)
        
        return min(1.0, score)
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text."""
        text = text.lower()
        
        text = re.sub(r'[^\w\s]', ' ', text)
        
        words = text.split()
        
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'only', 'own', 'same', 'than', 'too', 'very', 'just'}
        
        terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        return terms
    
    def _is_related_file(self, file1: str, file2: str) -> bool:
        """Check if two files are related."""
        path1 = Path(file1)
        path2 = Path(file2)
        
        if path1.parent == path2.parent:
            return True
        
        name1 = path1.stem.lower()
        name2 = path2.stem.lower()
        
        if name1 in name2 or name2 in name1:
            return True
        
        if name1.endswith('test') and name2 + 'test' == name1:
            return True
        if name2.endswith('test') and name1 + 'test' == name2:
            return True
        
        return False


class CodeCompressor:
    """Compresses code while preserving essential information."""
    
    def __init__(self, compression_ratio: float = 0.7):
        self.compression_ratio = compression_ratio
    
    def compress(
        self,
        content: str,
        target_tokens: int,
        preserve_signatures: bool = True,
        preserve_imports: bool = True
    ) -> str:
        """Compress code to target token count.
        
        Args:
            content: Original code content
            target_tokens: Target token count
            preserve_signatures: Keep method signatures
            preserve_imports: Keep import statements
            
        Returns:
            Compressed code
        """
        lines = content.split('\n')
        
        imports = []
        signatures = []
        body_lines = []
        comments = []
        
        in_block_comment = False
        current_signature = []
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('import ') or stripped.startswith('package '):
                if preserve_imports:
                    imports.append(line)
                continue
            
            if '/*' in stripped:
                in_block_comment = True
            if '*/' in stripped:
                in_block_comment = False
                comments.append(line)
                continue
            if in_block_comment or stripped.startswith('//'):
                comments.append(line)
                continue
            
            if self._is_signature_line(stripped):
                current_signature.append(line)
            elif current_signature and (stripped.startswith('{') or stripped.startswith('}')):
                if preserve_signatures:
                    signatures.extend(current_signature)
                    signatures.append(line)
                current_signature = []
            else:
                if current_signature:
                    body_lines.extend(current_signature)
                    current_signature = []
                body_lines.append(line)
        
        result_parts = []
        
        if imports:
            result_parts.extend(imports)
            result_parts.append('')
        
        if preserve_signatures and signatures:
            result_parts.extend(signatures)
        
        current_tokens = self._estimate_tokens('\n'.join(result_parts))
        remaining_tokens = target_tokens - current_tokens
        
        if remaining_tokens > 0:
            compressed_body = self._compress_body(body_lines, remaining_tokens)
            if compressed_body:
                result_parts.append('')
                result_parts.extend(compressed_body)
        
        return '\n'.join(result_parts)
    
    def _is_signature_line(self, line: str) -> bool:
        """Check if line is part of a method/class signature."""
        signature_patterns = [
            r'^(public|private|protected)\s+',
            r'^(static\s+)?(class|interface|enum)\s+',
            r'^@\w+',
            r'^\w+\s+\w+\s*\([^)]*\)\s*(throws\s+[\w,\s]+)?\s*$',
        ]
        
        for pattern in signature_patterns:
            if re.match(pattern, line):
                return True
        
        return False
    
    def _compress_body(
        self,
        lines: List[str],
        target_tokens: int
    ) -> List[str]:
        """Compress body lines to target tokens."""
        if not lines:
            return []
        
        total_lines = len(lines)
        tokens_per_line = self._estimate_tokens('\n'.join(lines)) / total_lines
        
        target_lines = int(target_tokens / tokens_per_line) if tokens_per_line > 0 else total_lines
        target_lines = min(target_lines, total_lines)
        
        if target_lines >= total_lines:
            return lines
        
        step = total_lines / target_lines
        selected_indices = [int(i * step) for i in range(target_lines)]
        
        compressed = []
        last_idx = -1
        
        for idx in selected_indices:
            if idx != last_idx + 1 and compressed:
                compressed.append('    // ...')
            compressed.append(lines[idx])
            last_idx = idx
        
        return compressed
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4


class ContextCompressor:
    """Main context compressor combining all strategies.
    
    Features:
    - Relevance-based snippet selection
    - Token-aware compression
    - Dependency-aware context building
    - Smart summarization
    """
    
    def __init__(
        self,
        config: Optional[ContextConfig] = None
    ):
        self.config = config or ContextConfig()
        self.scorer = RelevanceScorer()
        self.compressor = CodeCompressor(self.config.compression_ratio)
        
        self._snippets: List[CodeSnippet] = []
        self._file_cache: Dict[str, str] = {}
    
    def add_snippet(
        self,
        snippet: CodeSnippet
    ):
        """Add a code snippet to the context pool."""
        self._snippets.append(snippet)
    
    def add_file(
        self,
        file_path: str,
        content: str,
        language: str = "java"
    ):
        """Add a file to the context pool."""
        self._file_cache[file_path] = content
        
        lines = content.split('\n')
        
        imports = self._extract_imports(content)
        dependencies = self._extract_dependencies(content)
        symbols = self._extract_symbols(content)
        
        chunk_size = 100
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            snippet = CodeSnippet(
                file_path=file_path,
                content='\n'.join(chunk_lines),
                start_line=i + 1,
                end_line=min(i + chunk_size, len(lines)),
                language=language,
                imports=imports if i == 0 else [],
                dependencies=dependencies,
                symbols=symbols
            )
            self._snippets.append(snippet)
    
    def _extract_imports(self, content: str) -> List[str]:
        """Extract import statements from content."""
        imports = []
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('import '):
                imports.append(stripped)
        return imports
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract class dependencies from content."""
        dependencies = set()
        
        class_pattern = r'\b([A-Z][a-zA-Z0-9]*)\b'
        for match in re.finditer(class_pattern, content):
            dependencies.add(match.group(1))
        
        return list(dependencies)
    
    def _extract_symbols(self, content: str) -> List[str]:
        """Extract defined symbols from content."""
        symbols = []
        
        class_pattern = r'(?:class|interface|enum)\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            symbols.append(match.group(1))
        
        method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\('
        for match in re.finditer(method_pattern, content):
            symbols.append(match.group(1))
        
        return symbols
    
    def build_context(
        self,
        query: str,
        target_file: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> CompressedContext:
        """Build compressed context for a query.
        
        Args:
            query: The query/prompt for context
            target_file: The target file being processed
            additional_context: Additional context information
            
        Returns:
            CompressedContext with selected and compressed snippets
        """
        scored_snippets: List[Tuple[CodeSnippet, float]] = []
        
        for snippet in self._snippets:
            score = self.scorer.compute_relevance(
                snippet,
                query,
                target_file,
                additional_context
            )
            
            if snippet.file_path in self.config.priority_files:
                score = min(1.0, score + 0.3)
            
            if score >= self.config.min_relevance_score:
                scored_snippets.append((snippet, score))
        
        scored_snippets.sort(key=lambda x: x[1], reverse=True)
        
        selected_snippets: List[CodeSnippet] = []
        total_tokens = 0
        relevance_scores: Dict[str, float] = {}
        
        for snippet, score in scored_snippets:
            if len(selected_snippets) >= self.config.max_snippets:
                break
            
            if total_tokens + snippet.token_count > self.config.max_tokens:
                remaining_tokens = self.config.max_tokens - total_tokens
                if remaining_tokens > 100:
                    compressed_content = self.compressor.compress(
                        snippet.content,
                        remaining_tokens,
                        preserve_signatures=self.config.include_signatures,
                        preserve_imports=self.config.include_imports
                    )
                    if compressed_content:
                        snippet = CodeSnippet(
                            file_path=snippet.file_path,
                            content=compressed_content,
                            start_line=snippet.start_line,
                            end_line=snippet.end_line,
                            language=snippet.language,
                            relevance_score=score,
                            token_count=self._estimate_tokens(compressed_content)
                        )
                        selected_snippets.append(snippet)
                        relevance_scores[snippet.file_path] = score
                        total_tokens += snippet.token_count
                break
            
            snippet.relevance_score = score
            selected_snippets.append(snippet)
            relevance_scores[snippet.file_path] = score
            total_tokens += snippet.token_count
        
        context_parts = []
        current_file = None
        
        for snippet in selected_snippets:
            if snippet.file_path != current_file:
                if current_file is not None:
                    context_parts.append('')
                context_parts.append(f"// File: {snippet.file_path}")
                current_file = snippet.file_path
            
            if snippet.imports and self.config.include_imports:
                for imp in snippet.imports[:5]:
                    if imp not in '\n'.join(context_parts):
                        context_parts.append(imp)
                context_parts.append('')
            
            context_parts.append(snippet.content)
        
        final_content = '\n'.join(context_parts)
        
        return CompressedContext(
            content=final_content,
            total_tokens=total_tokens,
            snippets_included=len(selected_snippets),
            snippets_excluded=len(scored_snippets) - len(selected_snippets),
            compression_ratio=total_tokens / sum(s.token_count for s, _ in scored_snippets[:len(selected_snippets)]) if selected_snippets else 0,
            relevance_scores=relevance_scores,
            metadata={
                "query": query[:100],
                "target_file": target_file,
                "total_snippets_available": len(self._snippets)
            }
        )
    
    def build_minimal_context(
        self,
        target_file: str,
        query: str
    ) -> CompressedContext:
        """Build minimal context for a target file.
        
        Args:
            target_file: The target file
            query: The query/prompt
            
        Returns:
            Minimal CompressedContext
        """
        if target_file in self._file_cache:
            content = self._file_cache[target_file]
        else:
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"[ContextCompressor] Failed to read {target_file}: {e}")
                content = ""
        
        compressed = self.compressor.compress(
            content,
            self.config.max_tokens,
            preserve_signatures=True,
            preserve_imports=True
        )
        
        return CompressedContext(
            content=compressed,
            total_tokens=self._estimate_tokens(compressed),
            snippets_included=1,
            snippets_excluded=0,
            compression_ratio=self.config.compression_ratio,
            relevance_scores={target_file: 1.0},
            metadata={"minimal": True}
        )
    
    def clear(self):
        """Clear all cached snippets."""
        self._snippets.clear()
        self._file_cache.clear()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the context pool."""
        return {
            "total_snippets": len(self._snippets),
            "cached_files": len(self._file_cache),
            "total_tokens": sum(s.token_count for s in self._snippets),
            "files": list(set(s.file_path for s in self._snippets))
        }


class DependencyAwareContextBuilder:
    """Builds context considering file dependencies.
    
    Features:
    - Dependency graph analysis
    - Transitive dependency resolution
    - Import resolution
    """
    
    def __init__(
        self,
        context_compressor: ContextCompressor
    ):
        self.compressor = context_compressor
        self._dependency_graph: Dict[str, Set[str]] = {}
    
    def build_dependency_context(
        self,
        target_file: str,
        query: str,
        max_depth: int = 2
    ) -> CompressedContext:
        """Build context including dependencies.
        
        Args:
            target_file: The target file
            query: The query/prompt
            max_depth: Maximum dependency depth
            
        Returns:
            CompressedContext with dependencies
        """
        dependencies = self._resolve_dependencies(target_file, max_depth)
        
        for dep_file in dependencies:
            if dep_file not in self.compressor._file_cache:
                try:
                    with open(dep_file, 'r', encoding='utf-8') as f:
                        self.compressor.add_file(dep_file, f.read())
                except Exception as e:
                    logger.warning(f"[DependencyContext] Failed to read {dep_file}: {e}")
        
        return self.compressor.build_context(
            query=query,
            target_file=target_file,
            additional_context={"dependencies": list(dependencies)}
        )
    
    def _resolve_dependencies(
        self,
        file_path: str,
        max_depth: int
    ) -> Set[str]:
        """Resolve dependencies up to max_depth."""
        dependencies = set()
        to_visit = [(file_path, 0)]
        visited = set()
        
        while to_visit:
            current_file, depth = to_visit.pop(0)
            
            if current in visited or depth > max_depth:
                continue
            
            visited.add(current_file)
            
            if current_file in self._dependency_graph:
                for dep in self._dependency_graph[current_file]:
                    dependencies.add(dep)
                    if depth < max_depth:
                        to_visit.append((dep, depth + 1))
        
        return dependencies
    
    def register_dependency(
        self,
        file_path: str,
        dependency: str
    ):
        """Register a file dependency."""
        if file_path not in self._dependency_graph:
            self._dependency_graph[file_path] = set()
        self._dependency_graph[file_path].add(dependency)


def create_context_compressor(
    max_tokens: int = 8000,
    min_relevance: float = 0.3
) -> ContextCompressor:
    """Create a ContextCompressor instance.
    
    Args:
        max_tokens: Maximum tokens for context
        min_relevance: Minimum relevance score
        
    Returns:
        Configured ContextCompressor
    """
    config = ContextConfig(
        max_tokens=max_tokens,
        min_relevance_score=min_relevance
    )
    return ContextCompressor(config)
