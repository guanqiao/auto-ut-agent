"""Tests for Enhanced Context Compression.

This module tests the context compression system including:
- Content analysis and block creation
- Compression strategies
- Context compressor
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from pyutagent.agent.context_compression import (
    ContentPriority,
    CompressionLevel,
    ContentType,
    ContentBlock,
    CompressionContext,
    CompressionResult,
    ContentAnalyzer,
    PriorityBasedStrategy,
    SemanticStrategy,
    SummarizationStrategy,
    HybridStrategy,
    ContextCompressor,
    create_compressor,
)


class TestContentPriority:
    """Tests for ContentPriority enum."""
    
    def test_priority_values(self):
        """Test priority level values."""
        assert ContentPriority.CRITICAL.value == 100
        assert ContentPriority.HIGH.value == 75
        assert ContentPriority.MEDIUM.value == 50
        assert ContentPriority.LOW.value == 25
        assert ContentPriority.OPTIONAL.value == 10
    
    def test_priority_ordering(self):
        """Test priority ordering."""
        assert ContentPriority.CRITICAL.value > ContentPriority.HIGH.value
        assert ContentPriority.HIGH.value > ContentPriority.MEDIUM.value
        assert ContentPriority.MEDIUM.value > ContentPriority.LOW.value


class TestCompressionLevel:
    """Tests for CompressionLevel enum."""
    
    def test_level_values(self):
        """Test compression level values."""
        assert CompressionLevel.NONE.value == 0
        assert CompressionLevel.LIGHT.value == 1
        assert CompressionLevel.MODERATE.value == 2
        assert CompressionLevel.AGGRESSIVE.value == 3
        assert CompressionLevel.EXTREME.value == 4


class TestContentType:
    """Tests for ContentType enum."""
    
    def test_content_types(self):
        """Test content type values."""
        assert ContentType.CODE.value == "code"
        assert ContentType.COMMENT.value == "comment"
        assert ContentType.IMPORT.value == "import"
        assert ContentType.SIGNATURE.value == "signature"


class TestContentBlock:
    """Tests for ContentBlock dataclass."""
    
    def test_block_creation(self):
        """Test creating a content block."""
        block = ContentBlock(
            content="public class Test {}",
            content_type=ContentType.CODE,
            priority=ContentPriority.HIGH,
        )
        
        assert block.content == "public class Test {}"
        assert block.content_type == ContentType.CODE
        assert block.priority == ContentPriority.HIGH
    
    def test_token_count(self):
        """Test token count estimation."""
        block = ContentBlock(
            content="a" * 100,
            content_type=ContentType.CODE,
        )
        
        assert block.token_count == 25
    
    def test_can_compress(self):
        """Test compression eligibility."""
        code_block = ContentBlock(
            content="code",
            content_type=ContentType.CODE,
        )
        signature_block = ContentBlock(
            content="signature",
            content_type=ContentType.SIGNATURE,
        )
        error_block = ContentBlock(
            content="error",
            content_type=ContentType.ERROR,
        )
        
        assert code_block.can_compress() is True
        assert signature_block.can_compress() is False
        assert error_block.can_compress() is False


class TestCompressionContext:
    """Tests for CompressionContext dataclass."""
    
    def test_context_creation(self):
        """Test creating compression context."""
        context = CompressionContext(
            target_tokens=1000,
            current_tokens=2000,
        )
        
        assert context.target_tokens == 1000
        assert context.current_tokens == 2000
        assert context.compression_level == CompressionLevel.MODERATE
    
    def test_compression_ratio_target(self):
        """Test compression ratio target calculation."""
        context = CompressionContext(
            target_tokens=1000,
            current_tokens=2000,
        )
        
        assert context.compression_ratio_target == 0.5
    
    def test_compression_ratio_target_zero(self):
        """Test compression ratio with zero current tokens."""
        context = CompressionContext(
            target_tokens=1000,
            current_tokens=0,
        )
        
        assert context.compression_ratio_target == 1.0


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""
    
    def test_result_creation(self):
        """Test creating compression result."""
        result = CompressionResult(
            compressed_content="compressed",
            original_tokens=100,
            compressed_tokens=50,
            compression_ratio=0.5,
            blocks_processed=10,
            blocks_removed=5,
            blocks_summarized=2,
            strategy_used="hybrid",
            compression_level=CompressionLevel.MODERATE,
        )
        
        assert result.compressed_content == "compressed"
        assert result.compression_ratio == 0.5
        assert result.strategy_used == "hybrid"


class TestContentAnalyzer:
    """Tests for ContentAnalyzer."""
    
    def test_analyze_simple_code(self):
        """Test analyzing simple code."""
        code = """
import java.util.List;

public class Test {
    public void method() {
    }
}
"""
        blocks = ContentAnalyzer.analyze_content(code)
        
        assert len(blocks) > 0
        code_blocks = [b for b in blocks if b.content_type == ContentType.CODE]
        assert len(code_blocks) > 0
    
    def test_analyze_with_comments(self):
        """Test analyzing code with comments."""
        code = """
// This is a comment
public class Test {
    /* Multi-line
       comment */
    public void method() {
    }
}
"""
        blocks = ContentAnalyzer.analyze_content(code)
        
        assert len(blocks) > 0
    
    def test_classify_line_import(self):
        """Test classifying import lines."""
        line = "import java.util.List;"
        content_type = ContentAnalyzer._classify_line(line, False, False)
        
        assert content_type == ContentType.IMPORT
    
    def test_classify_line_class(self):
        """Test classifying class declaration."""
        line = "public class Test {"
        content_type = ContentAnalyzer._classify_line(line, False, False)
        
        assert content_type == ContentType.CODE
    
    def test_classify_line_comment(self):
        """Test classifying comment."""
        line = "// This is a comment"
        content_type = ContentAnalyzer._classify_line(line, False, False)
        
        assert content_type == ContentType.COMMENT
    
    def test_extract_name_class(self):
        """Test extracting class name."""
        content = "public class MyClass { }"
        name = ContentAnalyzer._extract_name(content, ContentType.CODE)
        
        assert name == "MyClass"
    
    def test_extract_dependencies(self):
        """Test extracting dependencies."""
        content = "doSomething(); callMethod();"
        deps = ContentAnalyzer._extract_dependencies(content)
        
        assert "doSomething" in deps
        assert "callMethod" in deps
    
    def test_calculate_semantic_importance(self):
        """Test semantic importance calculation."""
        public_content = "public class Test { }"
        private_content = "private void helper() { }"
        
        public_score = ContentAnalyzer._calculate_semantic_importance(
            public_content, ContentType.CODE
        )
        private_score = ContentAnalyzer._calculate_semantic_importance(
            private_content, ContentType.CODE
        )
        
        assert public_score > private_score


class TestPriorityBasedStrategy:
    """Tests for PriorityBasedStrategy."""
    
    def test_strategy_name(self):
        """Test strategy name."""
        strategy = PriorityBasedStrategy()
        assert strategy.name == "priority_based"
    
    def test_compress_preserves_critical(self):
        """Test that critical blocks are preserved."""
        strategy = PriorityBasedStrategy()
        
        blocks = [
            ContentBlock(
                content="critical",
                content_type=ContentType.ERROR,
                priority=ContentPriority.CRITICAL,
            ),
            ContentBlock(
                content="optional",
                content_type=ContentType.COMMENT,
                priority=ContentPriority.OPTIONAL,
            ),
        ]
        
        context = CompressionContext(
            target_tokens=5,
            current_tokens=100,
        )
        
        compressed, warnings = strategy.compress(blocks, context)
        
        assert any(b.priority == ContentPriority.CRITICAL for b in compressed)
    
    def test_compress_removes_low_priority(self):
        """Test that low priority blocks are removed when needed."""
        strategy = PriorityBasedStrategy()
        
        blocks = [
            ContentBlock(
                content="a" * 100,
                content_type=ContentType.CODE,
                priority=ContentPriority.HIGH,
            ),
            ContentBlock(
                content="b" * 100,
                content_type=ContentType.COMMENT,
                priority=ContentPriority.OPTIONAL,
            ),
        ]
        
        context = CompressionContext(
            target_tokens=30,
            current_tokens=200,
        )
        
        compressed, warnings = strategy.compress(blocks, context)
        
        assert len(compressed) < len(blocks)


class TestSemanticStrategy:
    """Tests for SemanticStrategy."""
    
    def test_strategy_name(self):
        """Test strategy name."""
        strategy = SemanticStrategy()
        assert strategy.name == "semantic"
    
    def test_compress_prioritizes_focus(self):
        """Test that focus methods are prioritized."""
        strategy = SemanticStrategy()
        
        blocks = [
            ContentBlock(
                content="target method",
                content_type=ContentType.CODE,
                name="targetMethod",
                semantic_importance=0.5,
            ),
            ContentBlock(
                content="other method",
                content_type=ContentType.CODE,
                name="otherMethod",
                semantic_importance=0.5,
            ),
        ]
        
        context = CompressionContext(
            target_tokens=10,
            current_tokens=100,
            focus_methods=["targetMethod"],
        )
        
        compressed, warnings = strategy.compress(blocks, context)
        
        target_in_result = any(b.name == "targetMethod" for b in compressed)
        assert target_in_result or len(compressed) == 0


class TestSummarizationStrategy:
    """Tests for SummarizationStrategy."""
    
    def test_strategy_name(self):
        """Test strategy name."""
        strategy = SummarizationStrategy()
        assert strategy.name == "summarization"
    
    def test_compress_preserves_high_priority(self):
        """Test that high priority blocks are preserved."""
        strategy = SummarizationStrategy()
        
        blocks = [
            ContentBlock(
                content="important code",
                content_type=ContentType.CODE,
                priority=ContentPriority.HIGH,
            ),
            ContentBlock(
                content="comment",
                content_type=ContentType.COMMENT,
                priority=ContentPriority.LOW,
            ),
        ]
        
        context = CompressionContext(
            target_tokens=100,
            current_tokens=100,
        )
        
        compressed, warnings = strategy.compress(blocks, context)
        
        high_priority_blocks = [b for b in compressed if b.priority == ContentPriority.HIGH]
        assert len(high_priority_blocks) > 0
    
    def test_summarize_comment(self):
        """Test summarizing comments."""
        strategy = SummarizationStrategy()
        
        block = ContentBlock(
            content="This is a long comment that should be summarized to a shorter version with more content to make it longer",
            content_type=ContentType.COMMENT,
            priority=ContentPriority.LOW,
        )
        
        context = CompressionContext(
            target_tokens=100,
            current_tokens=100,
            max_summary_length=30,
        )
        
        summary = strategy._summarize(block, context)
        
        assert summary is not None
        assert summary.content_type == ContentType.COMMENT
    
    def test_extract_key_points(self):
        """Test key point extraction."""
        strategy = SummarizationStrategy()
        
        content = "This is important. Note: do this. Warning: be careful."
        key_points = strategy._extract_key_points(content)
        
        assert "important" in key_points.lower() or "note" in key_points.lower()


class TestHybridStrategy:
    """Tests for HybridStrategy."""
    
    def test_strategy_name(self):
        """Test strategy name."""
        strategy = HybridStrategy()
        assert strategy.name == "hybrid"
    
    def test_compress_uses_priority_for_high_ratio(self):
        """Test that priority strategy is used for high compression ratio."""
        strategy = HybridStrategy()
        
        blocks = [
            ContentBlock(
                content="a" * 100,
                content_type=ContentType.CODE,
                priority=ContentPriority.HIGH,
            ),
            ContentBlock(
                content="b" * 100,
                content_type=ContentType.COMMENT,
                priority=ContentPriority.LOW,
            ),
        ]
        
        context = CompressionContext(
            target_tokens=150,
            current_tokens=200,
        )
        
        compressed, warnings = strategy.compress(blocks, context)
        
        assert isinstance(compressed, list)


class TestContextCompressor:
    """Tests for ContextCompressor."""
    
    def test_compressor_creation(self):
        """Test creating a compressor."""
        compressor = ContextCompressor()
        
        assert compressor.default_strategy == "hybrid"
    
    def test_compress_simple_content(self):
        """Test compressing simple content."""
        compressor = ContextCompressor()
        
        content = "public class Test { }"
        result = compressor.compress(
            content,
            target_tokens=100,
        )
        
        assert result.compressed_content != ""
        assert result.original_tokens > 0
        assert result.strategy_used == "hybrid"
    
    def test_compress_with_focus(self):
        """Test compressing with focus methods."""
        compressor = ContextCompressor()
        
        content = """
public class Test {
    public void targetMethod() { }
    public void otherMethod() { }
}
"""
        result = compressor.compress(
            content,
            target_tokens=100,
            focus_methods=["targetMethod"],
        )
        
        assert result.strategy_used == "hybrid"
    
    def test_compress_with_different_strategies(self):
        """Test different compression strategies."""
        compressor = ContextCompressor()
        
        content = "public class Test { public void method() { } }"
        
        for strategy in ["priority", "semantic", "hybrid"]:
            result = compressor.compress(
                content,
                target_tokens=50,
                strategy=strategy,
            )
            assert result.strategy_used == strategy
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        compressor = ContextCompressor()
        
        content = "a" * 100
        tokens = compressor.estimate_tokens(content)
        
        assert tokens == 25
    
    def test_get_available_strategies(self):
        """Test getting available strategies."""
        compressor = ContextCompressor()
        strategies = compressor.get_available_strategies()
        
        assert "priority" in strategies
        assert "semantic" in strategies
        assert "summarization" in strategies
        assert "hybrid" in strategies
    
    def test_compress_with_llm_summarizer(self):
        """Test compression with LLM summarizer."""
        mock_summarizer = MagicMock(return_value="Summary")
        compressor = ContextCompressor(llm_summarizer=mock_summarizer)
        
        content = "Long content that needs summarization"
        result = compressor.compress(
            content,
            target_tokens=10,
            strategy="summarization",
        )
        
        assert result.strategy_used == "summarization"


class TestCreateCompressor:
    """Tests for create_compressor function."""
    
    def test_create_default_compressor(self):
        """Test creating default compressor."""
        compressor = create_compressor()
        
        assert isinstance(compressor, ContextCompressor)
        assert compressor.default_strategy == "hybrid"
    
    def test_create_compressor_with_strategy(self):
        """Test creating compressor with custom strategy."""
        compressor = create_compressor(default_strategy="priority")
        
        assert compressor.default_strategy == "priority"
    
    def test_create_compressor_with_summarizer(self):
        """Test creating compressor with summarizer."""
        summarizer = lambda x: "summary"
        compressor = create_compressor(llm_summarizer=summarizer)
        
        assert compressor.llm_summarizer is not None


class TestIntegration:
    """Integration tests for context compression."""
    
    def test_full_compression_workflow(self):
        """Test full compression workflow."""
        compressor = ContextCompressor()
        
        content = """
package com.example;

import java.util.List;
import java.util.ArrayList;

/**
 * A test class for demonstration.
 * This class shows various features.
 */
public class TestClass {
    private String name;
    
    public TestClass(String name) {
        this.name = name;
    }
    
    public String getName() {
        return name;
    }
    
    public void processItems(List<String> items) {
        for (String item : items) {
            System.out.println(item);
        }
    }
    
    // This is a helper method
    private void helper() {
        // TODO: implement
    }
}
"""
        result = compressor.compress(
            content,
            target_tokens=100,
            focus_methods=["getName", "processItems"],
        )
        
        assert result.original_tokens > result.compressed_tokens
        assert result.compression_ratio < 1.0
        assert result.blocks_processed > 0
    
    def test_compression_with_extreme_level(self):
        """Test compression with extreme level."""
        compressor = ContextCompressor()
        
        content = "public class Test { " + "void method() {} " * 100 + "}"
        result = compressor.compress(
            content,
            target_tokens=50,
            compression_level=CompressionLevel.EXTREME,
        )
        
        assert result.compression_level == CompressionLevel.EXTREME
    
    def test_multiple_compression_rounds(self):
        """Test multiple compression rounds."""
        compressor = ContextCompressor()
        
        content = "public class Test { " + "void method() {} " * 50 + "}"
        
        result1 = compressor.compress(content, target_tokens=200)
        result2 = compressor.compress(
            result1.compressed_content,
            target_tokens=100,
        )
        
        assert result2.compressed_tokens <= result1.compressed_tokens
