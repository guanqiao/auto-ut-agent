"""Unit tests for ContextManager."""

import pytest
from pyutagent.agent.context_manager import (
    ContextManager,
    CompressionStrategy,
    TokenEstimator,
    compress_code_context,
    extract_method_context
)


class TestTokenEstimator:
    """Tests for TokenEstimator."""
    
    def test_estimate_empty_string(self):
        """Test estimating empty string."""
        assert TokenEstimator.estimate("") == 0
    
    def test_estimate_simple_text(self):
        """Test estimating simple text."""
        text = "Hello World"
        # 11 chars * 0.25 = 2.75 -> 2
        assert TokenEstimator.estimate(text) == 2
    
    def test_estimate_java_code(self):
        """Test estimating Java code."""
        code = "public class Test { }"
        # 21 chars * 0.25 = 5.25 -> 5
        assert TokenEstimator.estimate(code) == 5


class TestContextManager:
    """Tests for ContextManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a ContextManager instance."""
        return ContextManager(max_tokens=1000, target_tokens=800)
    
    @pytest.fixture
    def sample_java_code(self):
        """Sample Java code for testing."""
        return '''
package com.example;

import java.util.List;
import java.util.ArrayList;

public class SampleClass {
    private String name;
    private int value;
    
    public SampleClass(String name, int value) {
        this.name = name;
        this.value = value;
    }
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public int getValue() {
        return value;
    }
    
    public void setValue(int value) {
        this.value = value;
    }
    
    public int calculate(int x, int y) {
        return x + y + value;
    }
}
'''
    
    def test_no_compression_needed(self, manager, sample_java_code):
        """Test that small code doesn't get compressed."""
        result = manager.compress_context(sample_java_code)
        
        assert result.strategy_used == CompressionStrategy.NONE
        assert result.compression_ratio == 1.0
        assert result.processed_code == sample_java_code
    
    def test_method_only_compression(self, sample_java_code):
        """Test method-only compression strategy."""
        # Use a manager with low token limit to force compression
        manager = ContextManager(max_tokens=100, target_tokens=80)
        
        result = manager.compress_context(
            sample_java_code,
            target_methods=['calculate'],
            strategy=CompressionStrategy.METHOD_ONLY
        )
        
        assert result.strategy_used == CompressionStrategy.METHOD_ONLY
        # Should keep calculate method
        assert 'calculate' in result.processed_code
        # Should keep class header
        assert 'class SampleClass' in result.processed_code
    
    def test_extract_key_snippets(self, manager, sample_java_code):
        """Test extracting key snippets."""
        snippets = manager.extract_key_snippets(
            sample_java_code,
            method_names=['calculate', 'getName']
        )
        
        assert len(snippets) > 0
        snippet_names = [s.name for s in snippets if s.name]
        assert 'calculate' in snippet_names
        assert 'getName' in snippet_names
    
    def test_build_hierarchical_summary(self, manager):
        """Test building hierarchical summary."""
        class_info = {
            'name': 'SampleClass',
            'package': 'com.example',
            'methods': [
                {'name': 'getName', 'return_type': 'String', 'parameters': [], 'modifiers': ['public']},
                {'name': 'setName', 'return_type': 'void', 'parameters': [('String', 'name')], 'modifiers': ['public']},
            ],
            'fields': [('String', 'name'), ('int', 'value')],
            'imports': ['java.util.List', 'java.util.ArrayList']
        }
        
        summary = manager.build_hierarchical_summary(class_info)
        
        assert summary.class_name == 'SampleClass'
        assert summary.package == 'com.example'
        assert 'getName' in summary.method_summaries
        assert 'name' in summary.field_summaries
        assert len(summary.key_dependencies) == 2
    
    def test_format_summary_for_prompt(self, manager):
        """Test formatting summary for prompt."""
        class_info = {
            'name': 'SampleClass',
            'package': 'com.example',
            'methods': [
                {'name': 'getName', 'return_type': 'String', 'parameters': [], 'modifiers': ['public']},
            ],
            'fields': [('String', 'name')],
            'imports': ['java.util.List']
        }
        
        summary = manager.build_hierarchical_summary(class_info)
        formatted = manager.format_summary_for_prompt(summary)
        
        assert 'com.example' in formatted
        assert 'SampleClass' in formatted
        assert 'getName' in formatted
        assert 'name' in formatted


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_compress_code_context(self):
        """Test compress_code_context function."""
        code = "public class Test { public void method1() {} public void method2() {} }"
        
        result = compress_code_context(code, target_methods=['method1'], max_tokens=500)
        
        assert result is not None
        assert result.processed_code is not None
    
    def test_extract_method_context(self):
        """Test extract_method_context function."""
        code = '''
public class Test {
    public void method1() { }
    public void method2() { }
}
'''
        
        result = extract_method_context(code, ['method1'])
        
        assert 'method1' in result


class TestContextManagerEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_code(self):
        """Test handling empty code."""
        manager = ContextManager()
        result = manager.compress_context("")
        
        assert result.processed_code == ""
        assert result.original_tokens == 0
    
    def test_code_without_methods(self):
        """Test handling code without methods."""
        manager = ContextManager()
        code = "public class EmptyClass { }"
        
        result = manager.compress_context(code)
        
        assert result.strategy_used == CompressionStrategy.NONE
    
    def test_very_large_code(self):
        """Test handling very large code."""
        manager = ContextManager(max_tokens=100, target_tokens=80)
        # Generate large code
        code = "public class LargeClass {\n"
        for i in range(100):
            code += f"    public void method{i}() {{ int x = {i}; }}\n"
        code += "}"
        
        result = manager.compress_context(code)
        
        # Should apply compression
        assert result.compression_ratio < 1.0 or result.strategy_used == CompressionStrategy.HYBRID
