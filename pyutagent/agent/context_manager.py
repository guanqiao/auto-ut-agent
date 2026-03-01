"""Context window management for handling large code files.

This module provides intelligent context compression and management
for LLM interactions with large Java files that exceed token limits.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum, auto

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Strategies for context compression."""
    NONE = auto()           # No compression
    METHOD_ONLY = auto()    # Keep only target method signatures
    SMART = auto()          # Smart selection based on relevance
    SUMMARY = auto()        # Replace with AI-generated summary
    HYBRID = auto()         # Combine multiple strategies


@dataclass
class CodeSnippet:
    """Represents a code snippet with metadata."""
    content: str
    start_line: int
    end_line: int
    snippet_type: str  # 'method', 'field', 'import', 'class_header', etc.
    name: Optional[str] = None
    relevance_score: float = 0.0
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class HierarchicalSummary:
    """Hierarchical summary of a Java class."""
    class_name: str
    package: str
    class_description: str
    method_summaries: Dict[str, str] = field(default_factory=dict)
    field_summaries: Dict[str, str] = field(default_factory=dict)
    key_dependencies: List[str] = field(default_factory=list)


@dataclass
class ContextResult:
    """Result of context processing."""
    processed_code: str
    original_tokens: int
    processed_tokens: int
    compression_ratio: float
    strategy_used: CompressionStrategy
    snippets_included: List[CodeSnippet]
    summary: Optional[HierarchicalSummary] = None


class TokenEstimator:
    """Estimates token count for text."""
    
    # Average tokens per character for different languages
    # English/Java code is typically 0.25-0.3 tokens per char
    TOKENS_PER_CHAR = 0.25
    
    @classmethod
    def estimate(cls, text: str) -> int:
        """Estimate token count for text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return int(len(text) * cls.TOKENS_PER_CHAR)
    
    @classmethod
    def estimate_snippets(cls, snippets: List[CodeSnippet]) -> int:
        """Estimate total tokens for snippets.
        
        Args:
            snippets: List of code snippets
            
        Returns:
            Total estimated tokens
        """
        return sum(cls.estimate(s.content) for s in snippets)


class ContextManager:
    """Manages context window for large code files.
    
    Provides intelligent compression strategies to fit large Java files
    within LLM context limits while preserving essential information.
    """
    
    def __init__(
        self,
        max_tokens: int = 8000,
        target_tokens: int = 6000,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID
    ):
        """Initialize context manager.
        
        Args:
            max_tokens: Maximum allowed tokens (hard limit)
            target_tokens: Target token count (soft limit)
            strategy: Default compression strategy
        """
        self.max_tokens = max_tokens
        self.target_tokens = target_tokens
        self.default_strategy = strategy
        self.token_estimator = TokenEstimator()
        
        logger.info(f"[ContextManager] Initialized - Max: {max_tokens}, Target: {target_tokens}")
    
    def compress_context(
        self,
        code: str,
        target_methods: Optional[List[str]] = None,
        strategy: Optional[CompressionStrategy] = None
    ) -> ContextResult:
        """Compress code context to fit within token limits.
        
        Args:
            code: Original source code
            target_methods: List of method names to prioritize
            strategy: Compression strategy (uses default if None)
            
        Returns:
            ContextResult with compressed code and metadata
        """
        strategy = strategy or self.default_strategy
        original_tokens = self.token_estimator.estimate(code)
        
        logger.info(f"[ContextManager] Compressing context - Strategy: {strategy.name}, "
                   f"Original: {original_tokens} tokens")
        
        if original_tokens <= self.target_tokens:
            logger.debug("[ContextManager] No compression needed")
            return ContextResult(
                processed_code=code,
                original_tokens=original_tokens,
                processed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy_used=CompressionStrategy.NONE,
                snippets_included=[]
            )
        
        # Parse code into snippets
        snippets = self._parse_code_snippets(code)
        
        # Apply compression strategy
        if strategy == CompressionStrategy.METHOD_ONLY:
            result = self._apply_method_only_compression(snippets, target_methods)
        elif strategy == CompressionStrategy.SMART:
            result = self._apply_smart_compression(snippets, target_methods)
        elif strategy == CompressionStrategy.SUMMARY:
            result = self._apply_summary_compression(snippets)
        elif strategy == CompressionStrategy.HYBRID:
            result = self._apply_hybrid_compression(snippets, target_methods)
        else:
            result = ContextResult(
                processed_code=code,
                original_tokens=original_tokens,
                processed_tokens=original_tokens,
                compression_ratio=1.0,
                strategy_used=CompressionStrategy.NONE,
                snippets_included=snippets
            )
        
        result.original_tokens = original_tokens
        result.strategy_used = strategy
        
        logger.info(f"[ContextManager] Compression complete - Ratio: {result.compression_ratio:.2%}, "
                   f"Final: {result.processed_tokens} tokens")
        
        return result
    
    def extract_key_snippets(
        self,
        code: str,
        method_names: List[str],
        include_dependencies: bool = True,
        include_context: int = 3
    ) -> List[CodeSnippet]:
        """Extract key code snippets related to target methods.
        
        Args:
            code: Source code
            method_names: Names of methods to extract
            include_dependencies: Whether to include dependent code
            include_context: Lines of context around snippets
            
        Returns:
            List of relevant code snippets
        """
        all_snippets = self._parse_code_snippets(code)
        result = []
        
        # Find target method snippets
        target_snippets = [
            s for s in all_snippets 
            if s.snippet_type == 'method' and s.name in method_names
        ]
        result.extend(target_snippets)
        
        # Include class header
        class_headers = [s for s in all_snippets if s.snippet_type == 'class_header']
        result.extend(class_headers)
        
        # Include imports
        imports = [s for s in all_snippets if s.snippet_type == 'import']
        result.extend(imports)
        
        # Include dependencies if requested
        if include_dependencies:
            dependent_names = set()
            for snippet in target_snippets:
                dependent_names.update(snippet.dependencies)
            
            dependency_snippets = [
                s for s in all_snippets 
                if s.name in dependent_names and s not in result
            ]
            result.extend(dependency_snippets)
        
        # Sort by line number to maintain order
        result.sort(key=lambda s: s.start_line)
        
        logger.debug(f"[ContextManager] Extracted {len(result)} key snippets for {method_names}")
        return result
    
    def build_hierarchical_summary(
        self,
        class_info: Dict[str, Any],
        max_method_description_length: int = 100
    ) -> HierarchicalSummary:
        """Build a hierarchical summary of a Java class.
        
        Args:
            class_info: Parsed class information
            max_method_description_length: Max length for method descriptions
            
        Returns:
            Hierarchical summary
        """
        class_name = class_info.get('name', 'Unknown')
        package = class_info.get('package', '')
        methods = class_info.get('methods', [])
        fields = class_info.get('fields', [])
        
        # Build method summaries
        method_summaries = {}
        for method in methods:
            name = method.get('name', 'unknown')
            return_type = method.get('return_type', 'void')
            params = method.get('parameters', [])
            param_str = ', '.join([
                f"{p[0] if isinstance(p, tuple) else p.get('type', 'Object')} "
                f"{p[1] if isinstance(p, tuple) else p.get('name', 'param')}"
                for p in params
            ])
            
            description = f"{return_type} {name}({param_str})"
            modifiers = method.get('modifiers', [])
            if 'public' in modifiers:
                description = f"public {description}"
            elif 'private' in modifiers:
                description = f"private {description}"
            
            annotations = method.get('annotations', [])
            if annotations:
                description = f"[{', '.join(annotations)}] {description}"
            
            method_summaries[name] = description[:max_method_description_length]
        
        # Build field summaries
        field_summaries = {}
        for field in fields:
            if isinstance(field, tuple):
                field_type, field_name = field
            else:
                field_type = field.get('type', 'Object')
                field_name = field.get('name', 'unknown')
            field_summaries[field_name] = f"{field_type} {field_name}"
        
        # Extract key dependencies from imports
        imports = class_info.get('imports', [])
        key_dependencies = imports[:10]  # Limit to first 10 imports
        
        summary = HierarchicalSummary(
            class_name=class_name,
            package=package,
            class_description=f"Class {class_name}",
            method_summaries=method_summaries,
            field_summaries=field_summaries,
            key_dependencies=key_dependencies
        )
        
        logger.debug(f"[ContextManager] Built hierarchical summary for {class_name} - "
                    f"{len(method_summaries)} methods, {len(field_summaries)} fields")
        
        return summary
    
    def format_summary_for_prompt(
        self,
        summary: HierarchicalSummary,
        include_methods: bool = True,
        include_fields: bool = True
    ) -> str:
        """Format hierarchical summary for LLM prompt.
        
        Args:
            summary: Hierarchical summary
            include_methods: Whether to include method summaries
            include_fields: Whether to include field summaries
            
        Returns:
            Formatted summary string
        """
        lines = [
            f"package {summary.package};" if summary.package else "",
            "",
            f"// Class: {summary.class_name}",
            ""
        ]
        
        if summary.key_dependencies:
            lines.extend(["// Key imports:"] + summary.key_dependencies[:5])
            lines.append("")
        
        if include_fields and summary.field_summaries:
            lines.append("// Fields:")
            for field_desc in list(summary.field_summaries.values())[:10]:
                lines.append(f"//   {field_desc}")
            lines.append("")
        
        if include_methods and summary.method_summaries:
            lines.append("// Methods:")
            for method_desc in list(summary.method_summaries.values())[:20]:
                lines.append(f"//   {method_desc}")
            if len(summary.method_summaries) > 20:
                lines.append(f"//   ... and {len(summary.method_summaries) - 20} more methods")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _parse_code_snippets(self, code: str) -> List[CodeSnippet]:
        """Parse code into structured snippets.
        
        Args:
            code: Source code
            
        Returns:
            List of code snippets
        """
        snippets = []
        lines = code.split('\n')
        
        # Extract package declaration
        for i, line in enumerate(lines):
            if line.strip().startswith('package '):
                snippets.append(CodeSnippet(
                    content=line.strip(),
                    start_line=i,
                    end_line=i,
                    snippet_type='package',
                    name='package'
                ))
                break
        
        # Extract imports
        import_start = -1
        import_end = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('import '):
                if import_start == -1:
                    import_start = i
                import_end = i
        
        if import_start >= 0:
            import_content = '\n'.join(lines[import_start:import_end+1])
            snippets.append(CodeSnippet(
                content=import_content,
                start_line=import_start,
                end_line=import_end,
                snippet_type='import',
                name='imports'
            ))
        
        # Extract class header
        class_pattern = re.compile(r'^(public\s+|private\s+|protected\s+)?(abstract\s+|final\s+)?class\s+(\w+)')
        for i, line in enumerate(lines):
            match = class_pattern.match(line.strip())
            if match:
                # Find class header end (opening brace)
                header_end = i
                brace_count = 0
                for j in range(i, min(i + 10, len(lines))):
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    if '{' in lines[j]:
                        header_end = j
                        break
                
                snippets.append(CodeSnippet(
                    content='\n'.join(lines[i:header_end+1]),
                    start_line=i,
                    end_line=header_end,
                    snippet_type='class_header',
                    name=match.group(3)
                ))
                break
        
        # Extract methods
        method_pattern = re.compile(
            r'^(\s*)(public|private|protected)?\s*(static\s+)?(final\s+)?'
            r'([\w<>,\s]+)\s+(\w+)\s*\([^)]*\)\s*(throws\s+[\w,\s]+)?\s*\{',
            re.MULTILINE
        )
        
        for match in method_pattern.finditer(code):
            method_name = match.group(6)
            start_pos = match.start()
            
            # Find method body end
            brace_count = 1
            pos = match.end() - 1
            while brace_count > 0 and pos < len(code):
                if code[pos] == '{':
                    brace_count += 1
                elif code[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            method_content = code[start_pos:pos]
            start_line = code[:start_pos].count('\n')
            end_line = code[:pos].count('\n')
            
            # Extract dependencies (simple heuristic)
            dependencies = set()
            for other_match in method_pattern.finditer(code):
                other_name = other_match.group(6)
                if other_name != method_name and other_name in method_content:
                    dependencies.add(other_name)
            
            snippets.append(CodeSnippet(
                content=method_content,
                start_line=start_line,
                end_line=end_line,
                snippet_type='method',
                name=method_name,
                dependencies=dependencies
            ))
        
        # Sort by line number
        snippets.sort(key=lambda s: s.start_line)
        
        return snippets
    
    def _apply_method_only_compression(
        self,
        snippets: List[CodeSnippet],
        target_methods: Optional[List[str]]
    ) -> ContextResult:
        """Apply method-only compression strategy.
        
        Args:
            snippets: Code snippets
            target_methods: Target method names
            
        Returns:
            Compression result
        """
        # Keep only essential snippets
        essential_types = {'package', 'import', 'class_header'}
        
        if target_methods:
            # Keep target methods and their signatures
            filtered = [
                s for s in snippets 
                if s.snippet_type in essential_types or 
                   (s.snippet_type == 'method' and s.name in target_methods)
            ]
        else:
            # Keep only signatures for non-target methods
            filtered = [s for s in snippets if s.snippet_type in essential_types]
            
            # Add method signatures
            for s in snippets:
                if s.snippet_type == 'method':
                    # Extract signature only (first line)
                    signature = s.content.split('{')[0].strip() + ';'
                    filtered.append(CodeSnippet(
                        content=signature,
                        start_line=s.start_line,
                        end_line=s.start_line,
                        snippet_type='method_signature',
                        name=s.name
                    ))
        
        # Sort and combine
        filtered.sort(key=lambda s: s.start_line)
        processed_code = '\n\n'.join(s.content for s in filtered)
        processed_tokens = self.token_estimator.estimate(processed_code)
        
        return ContextResult(
            processed_code=processed_code,
            original_tokens=0,  # Will be set by caller
            processed_tokens=processed_tokens,
            compression_ratio=0,  # Will be calculated by caller
            strategy_used=CompressionStrategy.METHOD_ONLY,
            snippets_included=filtered
        )
    
    def _apply_smart_compression(
        self,
        snippets: List[CodeSnippet],
        target_methods: Optional[List[str]]
    ) -> ContextResult:
        """Apply smart compression based on relevance.
        
        Args:
            snippets: Code snippets
            target_methods: Target method names
            
        Returns:
            Compression result
        """
        # Calculate relevance scores
        for snippet in snippets:
            score = 0.0
            
            # Base scores by type
            if snippet.snippet_type == 'package':
                score = 1.0
            elif snippet.snippet_type == 'import':
                score = 0.9
            elif snippet.snippet_type == 'class_header':
                score = 1.0
            elif snippet.snippet_type == 'method':
                if target_methods and snippet.name in target_methods:
                    score = 1.0
                else:
                    score = 0.5
            
            snippet.relevance_score = score
        
        # Sort by relevance and select top snippets
        snippets.sort(key=lambda s: s.relevance_score, reverse=True)
        
        selected = []
        total_tokens = 0
        
        for snippet in snippets:
            snippet_tokens = self.token_estimator.estimate(snippet.content)
            if total_tokens + snippet_tokens <= self.target_tokens:
                selected.append(snippet)
                total_tokens += snippet_tokens
        
        # Sort selected by line number
        selected.sort(key=lambda s: s.start_line)
        
        processed_code = '\n\n'.join(s.content for s in selected)
        
        return ContextResult(
            processed_code=processed_code,
            original_tokens=0,
            processed_tokens=total_tokens,
            compression_ratio=0,
            strategy_used=CompressionStrategy.SMART,
            snippets_included=selected
        )
    
    def _apply_summary_compression(
        self, snippets: List[CodeSnippet]
    ) -> ContextResult:
        """Apply summary-based compression.
        
        Args:
            snippets: Code snippets
            
        Returns:
            Compression result
        """
        # For now, fall back to smart compression
        # Full implementation would use LLM to generate summaries
        return self._apply_smart_compression(snippets, None)
    
    def _apply_hybrid_compression(
        self,
        snippets: List[CodeSnippet],
        target_methods: Optional[List[str]]
    ) -> ContextResult:
        """Apply hybrid compression combining multiple strategies.
        
        Args:
            snippets: Code snippets
            target_methods: Target method names
            
        Returns:
            Compression result
        """
        # Start with smart compression
        result = self._apply_smart_compression(snippets, target_methods)
        
        # If still over limit, apply more aggressive compression
        if result.processed_tokens > self.max_tokens:
            logger.warning(f"[ContextManager] Smart compression insufficient, "
                          f"applying method-only compression")
            result = self._apply_method_only_compression(snippets, target_methods)
        
        return result


# Convenience functions for direct use

def compress_code_context(
    code: str,
    target_methods: Optional[List[str]] = None,
    max_tokens: int = 8000
) -> ContextResult:
    """Compress code context with default settings.
    
    Args:
        code: Source code to compress
        target_methods: Methods to prioritize
        max_tokens: Maximum token limit
        
    Returns:
        Compression result
    """
    manager = ContextManager(max_tokens=max_tokens)
    return manager.compress_context(code, target_methods)


def extract_method_context(
    code: str,
    method_names: List[str]
) -> str:
    """Extract context for specific methods.
    
    Args:
        code: Source code
        method_names: Names of methods to extract
        
    Returns:
        Extracted context as string
    """
    manager = ContextManager()
    snippets = manager.extract_key_snippets(code, method_names)
    return '\n\n'.join(s.content for s in snippets)
