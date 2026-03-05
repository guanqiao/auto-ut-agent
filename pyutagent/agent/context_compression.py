"""Enhanced Context Compression Module.

This module provides advanced context compression capabilities:
- LLM-driven summarization
- Priority-based content selection
- Incremental compression
- Semantic-aware compression

This is part of Phase 3 Week 21-22: Context Compression Enhancement.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyutagent.agent.unified_context_manager import (
        CompressionStrategy as BaseCompressionStrategy,
        CodeSnippet,
        HierarchicalSummary,
    )

logger = logging.getLogger(__name__)


class ContentPriority(Enum):
    """Priority levels for content."""
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    OPTIONAL = 10


class CompressionLevel(Enum):
    """Levels of compression aggressiveness."""
    NONE = 0
    LIGHT = 1
    MODERATE = 2
    AGGRESSIVE = 3
    EXTREME = 4


class ContentType(Enum):
    """Types of content for compression."""
    CODE = "code"
    COMMENT = "comment"
    IMPORT = "import"
    SIGNATURE = "signature"
    DOCSTRING = "docstring"
    TEST = "test"
    CONFIG = "config"
    LOG = "log"
    ERROR = "error"
    CONTEXT = "context"


@dataclass
class ContentBlock:
    """A block of content with metadata."""
    content: str
    content_type: ContentType
    priority: ContentPriority = ContentPriority.MEDIUM
    start_line: int = 0
    end_line: int = 0
    name: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    semantic_importance: float = 0.5
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def token_count(self) -> int:
        """Estimate token count."""
        return int(len(self.content) * 0.25)
    
    def can_compress(self) -> bool:
        """Check if this block can be compressed."""
        return self.content_type not in [
            ContentType.SIGNATURE,
            ContentType.ERROR,
        ]


@dataclass
class CompressionContext:
    """Context for compression operation."""
    target_tokens: int
    current_tokens: int
    compression_level: CompressionLevel = CompressionLevel.MODERATE
    preserve_signatures: bool = True
    preserve_imports: bool = True
    preserve_errors: bool = True
    max_summary_length: int = 200
    include_dependencies: bool = True
    focus_methods: List[str] = field(default_factory=list)
    focus_classes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def compression_ratio_target(self) -> float:
        """Calculate target compression ratio."""
        if self.current_tokens == 0:
            return 1.0
        return self.target_tokens / self.current_tokens


@dataclass
class CompressionResult:
    """Result of compression operation."""
    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    blocks_processed: int
    blocks_removed: int
    blocks_summarized: int
    strategy_used: str
    compression_level: CompressionLevel
    processing_time_ms: int = 0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentAnalyzer:
    """Analyzes content for compression decisions."""
    
    IMPORT_PATTERN = re.compile(r'^\s*(import|from)\s+[\w.]+')
    CLASS_PATTERN = re.compile(r'^\s*(public|private|protected)?\s*(abstract|final)?\s*class\s+(\w+)')
    METHOD_PATTERN = re.compile(r'^\s*(public|private|protected)?\s*(static)?\s*[\w<>\[\],\s]+\s+(\w+)\s*\(')
    COMMENT_PATTERN = re.compile(r'^\s*(//|/\*|\*|\*/)')
    DOCSTRING_PATTERN = re.compile(r'^\s*("""|\'\'\'|/\*\*)')
    ANNOTATION_PATTERN = re.compile(r'^\s*@\w+')
    
    @classmethod
    def analyze_content(cls, content: str) -> List[ContentBlock]:
        """Analyze content and create blocks.
        
        Args:
            content: Source content to analyze
            
        Returns:
            List of content blocks
        """
        blocks = []
        lines = content.split('\n')
        
        current_block_lines = []
        current_block_type = None
        current_block_start = 0
        brace_depth = 0
        in_multiline_comment = False
        in_docstring = False
        
        for i, line in enumerate(lines):
            line_type = cls._classify_line(line, in_multiline_comment, in_docstring)
            
            if '/*' in line and '*/' not in line:
                in_multiline_comment = True
            if '*/' in line:
                in_multiline_comment = False
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
            
            if current_block_type is None:
                current_block_type = line_type
                current_block_start = i
            
            current_block_lines.append(line)
            brace_depth += line.count('{') - line.count('}')
            
            is_block_end = (
                (line_type != current_block_type and not in_multiline_comment) or
                (current_block_type in [ContentType.CODE, ContentType.TEST] and brace_depth == 0 and '{' in ''.join(current_block_lines)) or
                i == len(lines) - 1
            )
            
            if is_block_end:
                block_content = '\n'.join(current_block_lines)
                block = cls._create_block(
                    block_content,
                    current_block_type,
                    current_block_start,
                    i,
                )
                blocks.append(block)
                
                current_block_lines = []
                current_block_type = None
                current_block_start = i + 1
        
        return blocks
    
    @classmethod
    def _classify_line(
        cls,
        line: str,
        in_comment: bool,
        in_docstring: bool,
    ) -> ContentType:
        """Classify a single line."""
        stripped = line.strip()
        
        if not stripped:
            return ContentType.CONTEXT
        
        if in_comment or cls.COMMENT_PATTERN.match(stripped):
            if cls.DOCSTRING_PATTERN.match(stripped) or in_docstring:
                return ContentType.DOCSTRING
            return ContentType.COMMENT
        
        if cls.IMPORT_PATTERN.match(stripped):
            return ContentType.IMPORT
        
        if cls.ANNOTATION_PATTERN.match(stripped):
            return ContentType.CODE
        
        if cls.CLASS_PATTERN.match(stripped):
            return ContentType.CODE
        
        if cls.METHOD_PATTERN.match(stripped):
            if 'test' in stripped.lower():
                return ContentType.TEST
            return ContentType.CODE
        
        return ContentType.CODE
    
    @classmethod
    def _create_block(
        cls,
        content: str,
        content_type: ContentType,
        start_line: int,
        end_line: int,
    ) -> ContentBlock:
        """Create a content block."""
        priority = cls._determine_priority(content, content_type)
        name = cls._extract_name(content, content_type)
        dependencies = cls._extract_dependencies(content)
        semantic_importance = cls._calculate_semantic_importance(content, content_type)
        
        return ContentBlock(
            content=content,
            content_type=content_type,
            priority=priority,
            start_line=start_line,
            end_line=end_line,
            name=name,
            dependencies=dependencies,
            semantic_importance=semantic_importance,
        )
    
    @classmethod
    def _determine_priority(
        cls,
        content: str,
        content_type: ContentType,
    ) -> ContentPriority:
        """Determine content priority."""
        if content_type == ContentType.IMPORT:
            return ContentPriority.HIGH
        
        if content_type == ContentType.ERROR:
            return ContentPriority.CRITICAL
        
        if content_type == ContentType.SIGNATURE:
            return ContentPriority.HIGH
        
        if content_type == ContentType.CODE:
            if 'public' in content or 'class ' in content:
                return ContentPriority.HIGH
            return ContentPriority.MEDIUM
        
        if content_type == ContentType.TEST:
            return ContentPriority.MEDIUM
        
        if content_type == ContentType.DOCSTRING:
            return ContentPriority.LOW
        
        if content_type == ContentType.COMMENT:
            return ContentPriority.OPTIONAL
        
        return ContentPriority.MEDIUM
    
    @classmethod
    def _extract_name(cls, content: str, content_type: ContentType) -> Optional[str]:
        """Extract name from content."""
        if content_type == ContentType.CODE:
            class_match = cls.CLASS_PATTERN.search(content)
            if class_match:
                return class_match.group(3)
            
            method_match = cls.METHOD_PATTERN.search(content)
            if method_match:
                return method_match.group(3)
        
        return None
    
    @classmethod
    def _extract_dependencies(cls, content: str) -> Set[str]:
        """Extract dependencies from content."""
        dependencies = set()
        
        method_calls = re.findall(r'\b(\w+)\s*\(', content)
        dependencies.update(method_calls)
        
        type_refs = re.findall(r'\b([A-Z][a-zA-Z0-9]*)\b', content)
        dependencies.update(type_refs)
        
        return dependencies
    
    @classmethod
    def _calculate_semantic_importance(
        cls,
        content: str,
        content_type: ContentType,
    ) -> float:
        """Calculate semantic importance score."""
        score = 0.5
        
        if 'public' in content:
            score += 0.2
        if 'class ' in content:
            score += 0.2
        if '@Override' in content or '@Test' in content:
            score += 0.1
        if 'TODO' in content or 'FIXME' in content:
            score += 0.1
        if 'deprecated' in content.lower():
            score -= 0.2
        
        return min(1.0, max(0.0, score))


class CompressionStrategy(ABC):
    """Abstract base class for compression strategies."""
    
    @abstractmethod
    def compress(
        self,
        blocks: List[ContentBlock],
        context: CompressionContext,
    ) -> Tuple[List[ContentBlock], List[str]]:
        """Compress content blocks.
        
        Args:
            blocks: Content blocks to compress
            context: Compression context
            
        Returns:
            Tuple of (compressed blocks, warnings)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass


class PriorityBasedStrategy(CompressionStrategy):
    """Priority-based compression strategy."""
    
    @property
    def name(self) -> str:
        return "priority_based"
    
    def compress(
        self,
        blocks: List[ContentBlock],
        context: CompressionContext,
    ) -> Tuple[List[ContentBlock], List[str]]:
        """Compress based on priority."""
        warnings = []
        
        sorted_blocks = sorted(
            blocks,
            key=lambda b: (b.priority.value, b.semantic_importance),
            reverse=True,
        )
        
        selected = []
        total_tokens = 0
        
        for block in sorted_blocks:
            block_tokens = block.token_count
            
            if total_tokens + block_tokens <= context.target_tokens:
                selected.append(block)
                total_tokens += block_tokens
            elif block.priority == ContentPriority.CRITICAL:
                selected.append(block)
                total_tokens += block_tokens
                warnings.append(
                    f"Critical block '{block.name}' exceeds target"
                )
        
        selected.sort(key=lambda b: b.start_line)
        
        return selected, warnings


class SemanticStrategy(CompressionStrategy):
    """Semantic-aware compression strategy."""
    
    @property
    def name(self) -> str:
        return "semantic"
    
    def compress(
        self,
        blocks: List[ContentBlock],
        context: CompressionContext,
    ) -> Tuple[List[ContentBlock], List[str]]:
        """Compress based on semantic importance."""
        warnings = []
        
        focus_set = set(context.focus_methods + context.focus_classes)
        
        def calculate_score(block: ContentBlock) -> float:
            score = block.semantic_importance
            
            if block.name and block.name in focus_set:
                score += 0.5
            
            if context.include_dependencies:
                for dep in block.dependencies:
                    if dep in focus_set:
                        score += 0.2
            
            score += block.priority.value / 100 * 0.3
            
            return score
        
        sorted_blocks = sorted(blocks, key=calculate_score, reverse=True)
        
        selected = []
        total_tokens = 0
        
        for block in sorted_blocks:
            block_tokens = block.token_count
            
            if total_tokens + block_tokens <= context.target_tokens:
                selected.append(block)
                total_tokens += block_tokens
        
        selected.sort(key=lambda b: b.start_line)
        
        return selected, warnings


class SummarizationStrategy(CompressionStrategy):
    """Summarization-based compression strategy."""
    
    def __init__(self, llm_summarizer: Optional[Callable] = None):
        self.llm_summarizer = llm_summarizer
    
    @property
    def name(self) -> str:
        return "summarization"
    
    def compress(
        self,
        blocks: List[ContentBlock],
        context: CompressionContext,
    ) -> Tuple[List[ContentBlock], List[str]]:
        """Compress by summarizing low-priority content."""
        warnings = []
        result_blocks = []
        
        for block in blocks:
            if block.priority.value >= ContentPriority.HIGH.value:
                result_blocks.append(block)
            elif block.content_type in [ContentType.COMMENT, ContentType.DOCSTRING]:
                summary = self._summarize(block, context)
                if summary:
                    result_blocks.append(summary)
            elif block.can_compress():
                compressed = self._compress_block(block, context)
                result_blocks.append(compressed)
            else:
                result_blocks.append(block)
        
        return result_blocks, warnings
    
    def _summarize(
        self,
        block: ContentBlock,
        context: CompressionContext,
    ) -> Optional[ContentBlock]:
        """Create a summary block."""
        if self.llm_summarizer:
            try:
                summary_text = self.llm_summarizer(
                    block.content,
                    max_length=context.max_summary_length,
                )
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")
                summary_text = self._extract_key_points(block.content)
        else:
            summary_text = self._extract_key_points(block.content)
        
        if not summary_text:
            return None
        
        return ContentBlock(
            content=f"// Summary: {summary_text}",
            content_type=ContentType.COMMENT,
            priority=block.priority,
            start_line=block.start_line,
            end_line=block.end_line,
            name=block.name,
            compression_ratio=len(summary_text) / max(1, len(block.content)),
        )
    
    def _compress_block(
        self,
        block: ContentBlock,
        context: CompressionContext,
    ) -> ContentBlock:
        """Compress a single block."""
        if block.content_type == ContentType.CODE:
            return self._compress_code_block(block)
        
        return block
    
    def _compress_code_block(self, block: ContentBlock) -> ContentBlock:
        """Compress a code block."""
        lines = block.content.split('\n')
        compressed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if (stripped and 
                not stripped.startswith('//') and
                not stripped.startswith('/*') and
                not stripped.startswith('*')):
                compressed_lines.append(line)
        
        return ContentBlock(
            content='\n'.join(compressed_lines),
            content_type=block.content_type,
            priority=block.priority,
            start_line=block.start_line,
            end_line=block.end_line,
            name=block.name,
            dependencies=block.dependencies,
            compression_ratio=len(compressed_lines) / max(1, len(lines)),
        )
    
    def _extract_key_points(self, content: str) -> str:
        """Extract key points from content."""
        lines = content.split('\n')
        key_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 10:
                if any(kw in stripped.lower() for kw in ['important', 'note', 'warning', 'todo', 'fixme']):
                    key_lines.append(stripped)
        
        if key_lines:
            return ' | '.join(key_lines[:3])
        
        return content[:100] + '...' if len(content) > 100 else content


class HybridStrategy(CompressionStrategy):
    """Hybrid compression strategy combining multiple approaches."""
    
    def __init__(self, llm_summarizer: Optional[Callable] = None):
        self.priority_strategy = PriorityBasedStrategy()
        self.semantic_strategy = SemanticStrategy()
        self.summary_strategy = SummarizationStrategy(llm_summarizer)
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    def compress(
        self,
        blocks: List[ContentBlock],
        context: CompressionContext,
    ) -> Tuple[List[ContentBlock], List[str]]:
        """Apply hybrid compression."""
        all_warnings = []
        
        total_tokens = sum(b.token_count for b in blocks)
        
        if total_tokens <= context.target_tokens:
            return blocks, all_warnings
        
        ratio = context.target_tokens / max(1, total_tokens)
        
        if ratio > 0.7:
            blocks, warnings = self.priority_strategy.compress(blocks, context)
            all_warnings.extend(warnings)
        elif ratio > 0.4:
            blocks, warnings = self.semantic_strategy.compress(blocks, context)
            all_warnings.extend(warnings)
        else:
            blocks, warnings = self.summary_strategy.compress(blocks, context)
            all_warnings.extend(warnings)
            
            current_tokens = sum(b.token_count for b in blocks)
            if current_tokens > context.target_tokens:
                new_context = CompressionContext(
                    target_tokens=context.target_tokens,
                    current_tokens=current_tokens,
                    compression_level=context.compression_level,
                )
                blocks, warnings = self.priority_strategy.compress(blocks, new_context)
                all_warnings.extend(warnings)
        
        return blocks, all_warnings


class ContextCompressor:
    """Main context compression engine."""
    
    STRATEGIES = {
        "priority": PriorityBasedStrategy,
        "semantic": SemanticStrategy,
        "summarization": SummarizationStrategy,
        "hybrid": HybridStrategy,
    }
    
    def __init__(
        self,
        default_strategy: str = "hybrid",
        llm_summarizer: Optional[Callable] = None,
    ):
        self.default_strategy = default_strategy
        self.llm_summarizer = llm_summarizer
        self.analyzer = ContentAnalyzer()
    
    def compress(
        self,
        content: str,
        target_tokens: int,
        strategy: Optional[str] = None,
        focus_methods: Optional[List[str]] = None,
        focus_classes: Optional[List[str]] = None,
        compression_level: CompressionLevel = CompressionLevel.MODERATE,
    ) -> CompressionResult:
        """Compress content to target token count.
        
        Args:
            content: Content to compress
            target_tokens: Target token count
            strategy: Compression strategy to use
            focus_methods: Methods to focus on
            focus_classes: Classes to focus on
            compression_level: Aggressiveness level
            
        Returns:
            CompressionResult
        """
        start_time = datetime.now()
        
        strategy_name = strategy or self.default_strategy
        strategy_instance = self._create_strategy(strategy_name)
        
        blocks = self.analyzer.analyze_content(content)
        original_tokens = sum(b.token_count for b in blocks)
        
        context = CompressionContext(
            target_tokens=target_tokens,
            current_tokens=original_tokens,
            compression_level=compression_level,
            focus_methods=focus_methods or [],
            focus_classes=focus_classes or [],
        )
        
        compressed_blocks, warnings = strategy_instance.compress(blocks, context)
        
        compressed_content = '\n'.join(b.content for b in compressed_blocks)
        compressed_tokens = sum(b.token_count for b in compressed_blocks)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return CompressionResult(
            compressed_content=compressed_content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / max(1, original_tokens),
            blocks_processed=len(blocks),
            blocks_removed=len(blocks) - len(compressed_blocks),
            blocks_summarized=sum(1 for b in compressed_blocks if b.compression_ratio < 1.0),
            strategy_used=strategy_name,
            compression_level=compression_level,
            processing_time_ms=int(processing_time),
            warnings=warnings,
        )
    
    def _create_strategy(self, name: str) -> CompressionStrategy:
        """Create a strategy instance."""
        strategy_class = self.STRATEGIES.get(name, HybridStrategy)
        
        if name in ["summarization", "hybrid"]:
            return strategy_class(self.llm_summarizer)
        
        return strategy_class()
    
    def estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        return int(len(content) * 0.25)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        return list(self.STRATEGIES.keys())


def create_compressor(
    default_strategy: str = "hybrid",
    llm_summarizer: Optional[Callable] = None,
) -> ContextCompressor:
    """Create a context compressor instance.
    
    Args:
        default_strategy: Default compression strategy
        llm_summarizer: Optional LLM summarization function
        
    Returns:
        ContextCompressor instance
    """
    return ContextCompressor(
        default_strategy=default_strategy,
        llm_summarizer=llm_summarizer,
    )
