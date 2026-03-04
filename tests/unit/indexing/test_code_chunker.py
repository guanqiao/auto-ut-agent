"""Tests for code chunker module."""

import pytest
from pathlib import Path

from pyutagent.indexing.code_chunker import (
    ChunkType,
    ChunkStrategy,
    CodeChunk,
    ChunkingConfig,
    CodeChunker,
)


class TestChunkType:
    """Tests for ChunkType enum."""

    def test_chunk_types_exist(self):
        assert ChunkType.FILE.value == "file"
        assert ChunkType.CLASS.value == "class"
        assert ChunkType.METHOD.value == "method"
        assert ChunkType.FUNCTION.value == "function"


class TestChunkStrategy:
    """Tests for ChunkStrategy enum."""

    def test_strategies_exist(self):
        assert ChunkStrategy.BY_FILE.value == "by_file"
        assert ChunkStrategy.BY_CLASS.value == "by_class"
        assert ChunkStrategy.BY_METHOD.value == "by_method"
        assert ChunkStrategy.HYBRID.value == "hybrid"


class TestCodeChunk:
    """Tests for CodeChunk."""

    def test_chunk_creation(self):
        chunk = CodeChunk(
            id="chunk_001",
            content="public class Test {}",
            chunk_type=ChunkType.CLASS,
            file_path="Test.java",
            start_line=1,
            end_line=10,
            name="Test",
        )
        
        assert chunk.id == "chunk_001"
        assert chunk.chunk_type == ChunkType.CLASS
        assert chunk.line_count == 10

    def test_to_dict(self):
        chunk = CodeChunk(
            id="chunk_001",
            content="test content",
            chunk_type=ChunkType.METHOD,
            file_path="Test.java",
            start_line=5,
            end_line=15,
            name="testMethod",
            parent="TestClass",
        )
        
        result = chunk.to_dict()
        
        assert result["id"] == "chunk_001"
        assert result["chunk_type"] == "method"
        assert result["name"] == "testMethod"
        assert result["parent"] == "TestClass"

    def test_token_estimate(self):
        chunk = CodeChunk(
            id="chunk_001",
            content="public void test() { return 1; }",
            chunk_type=ChunkType.METHOD,
            file_path="Test.java",
            start_line=1,
            end_line=1,
        )
        
        assert chunk.token_estimate > 0


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self):
        config = ChunkingConfig()
        
        assert config.strategy == ChunkStrategy.HYBRID
        assert config.max_chunk_lines == 100
        assert config.max_chunk_tokens == 2000

    def test_custom_config(self):
        config = ChunkingConfig(
            strategy=ChunkStrategy.BY_CLASS,
            max_chunk_lines=50,
            include_comments=True,
        )
        
        assert config.strategy == ChunkStrategy.BY_CLASS
        assert config.max_chunk_lines == 50
        assert config.include_comments is True


class TestCodeChunker:
    """Tests for CodeChunker."""

    @pytest.fixture
    def chunker(self):
        return CodeChunker()

    @pytest.fixture
    def sample_java_content(self):
        return '''
package com.example;

import java.util.List;

public class UserService {
    private String name;
    
    public UserService(String name) {
        this.name = name;
    }
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    public List<String> getAllNames() {
        return List.of(name);
    }
}

interface UserInterface {
    String getName();
}
'''

    def test_detect_language_java(self, chunker):
        assert chunker.detect_language("Test.java") == "java"
        assert chunker.detect_language("path/to/Test.java") == "java"

    def test_detect_language_python(self, chunker):
        assert chunker.detect_language("test.py") == "python"

    def test_detect_language_typescript(self, chunker):
        assert chunker.detect_language("test.ts") == "typescript"

    def test_detect_language_javascript(self, chunker):
        assert chunker.detect_language("test.js") == "javascript"

    def test_detect_language_unknown(self, chunker):
        assert chunker.detect_language("test.xyz") == "unknown"

    def test_chunk_by_file(self, chunker, sample_java_content):
        chunker.config.strategy = ChunkStrategy.BY_FILE
        chunks = chunker.chunk_file("Test.java", sample_java_content)
        
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.FILE

    def test_chunk_by_class(self, chunker, sample_java_content):
        chunker.config.strategy = ChunkStrategy.BY_CLASS
        chunks = chunker.chunk_file("Test.java", sample_java_content)
        
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.CLASS for c in chunks)

    def test_chunk_by_method(self, chunker, sample_java_content):
        chunker.config.strategy = ChunkStrategy.BY_METHOD
        chunks = chunker.chunk_file("Test.java", sample_java_content)
        
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        assert len(method_chunks) >= 1

    def test_chunk_by_lines(self, chunker, sample_java_content):
        chunker.config.strategy = ChunkStrategy.BY_LINES
        chunker.config.max_chunk_lines = 5
        chunks = chunker.chunk_file("Test.java", sample_java_content)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.line_count <= 5 + chunker.config.overlap_lines

    def test_chunk_hybrid(self, chunker, sample_java_content):
        chunker.config.strategy = ChunkStrategy.HYBRID
        chunks = chunker.chunk_file("Test.java", sample_java_content)
        
        assert len(chunks) >= 1

    def test_find_matching_brace(self, chunker):
        content = "public class Test { int x; { nested; } }"
        start = content.index('{')
        end = chunker._find_matching_brace(content, start)
        
        assert end == len(content) - 1

    def test_find_matching_brace_nested(self, chunker):
        content = "public void test() { if (true) { return; } }"
        start = content.index('{')
        end = chunker._find_matching_brace(content, start)
        
        assert content[end] == '}'

    def test_chunk_extracts_class_name(self, chunker, sample_java_content):
        chunker.config.strategy = ChunkStrategy.BY_CLASS
        chunks = chunker.chunk_file("Test.java", sample_java_content)
        
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert any(c.name == "UserService" for c in class_chunks)

    def test_chunk_extracts_method_name(self, chunker, sample_java_content):
        chunker.config.strategy = ChunkStrategy.BY_METHOD
        chunks = chunker.chunk_file("Test.java", sample_java_content)
        
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        method_names = [c.name for c in method_chunks]
        
        assert "getName" in method_names or any("get" in m for m in method_names)


class TestCodeChunkerPython:
    """Tests for Python code chunking."""

    @pytest.fixture
    def chunker(self):
        return CodeChunker()

    @pytest.fixture
    def sample_python_content(self):
        return '''
import os
from typing import List

class UserService:
    def __init__(self, name: str):
        self.name = name
    
    def get_name(self) -> str:
        return self.name
    
    def set_name(self, name: str) -> None:
        self.name = name

def helper_function():
    pass
'''

    def test_chunk_python_class(self, chunker, sample_python_content):
        chunker.config.strategy = ChunkStrategy.BY_CLASS
        chunks = chunker.chunk_file("test.py", sample_python_content)
        
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

    def test_chunk_python_detects_language(self, chunker):
        assert chunker.detect_language("test.py") == "python"
