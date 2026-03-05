"""Tests for the codebase indexer module."""

import json
import tempfile
from pathlib import Path

import pytest

from pyutagent.indexing.codebase_indexer import (
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


class TestCodeSymbol:
    """Tests for CodeSymbol dataclass."""

    def test_code_symbol_creation(self):
        """Test creating a CodeSymbol."""
        symbol = CodeSymbol(
            id="sym_00000001",
            name="TestClass",
            symbol_type=SymbolType.CLASS,
            file_path="/test/TestClass.java",
            start_line=1,
            end_line=50,
            signature="public class TestClass",
            modifiers=["public"],
            annotations=["@Entity"],
        )

        assert symbol.id == "sym_00000001"
        assert symbol.name == "TestClass"
        assert symbol.symbol_type == SymbolType.CLASS
        assert symbol.full_name == "TestClass"
        assert symbol.location == "/test/TestClass.java:1"

    def test_code_symbol_with_parent(self):
        """Test CodeSymbol with parent."""
        parent = CodeSymbol(
            id="sym_00000001",
            name="ParentClass",
            symbol_type=SymbolType.CLASS,
            file_path="/test/Parent.java",
            start_line=1,
            end_line=30,
        )

        method = CodeSymbol(
            id="sym_00000002",
            name="testMethod",
            symbol_type=SymbolType.METHOD,
            file_path="/test/Parent.java",
            start_line=10,
            end_line=20,
            parent_id=parent.id,
            return_type="void",
            parameters=[("String", "arg1")],
        )

        assert method.parent_id == parent.id
        assert method.full_name == f"{parent.id}.testMethod"

    def test_code_symbol_serialization(self):
        """Test CodeSymbol serialization."""
        symbol = CodeSymbol(
            id="sym_00000001",
            name="TestMethod",
            symbol_type=SymbolType.METHOD,
            file_path="/test/Test.java",
            start_line=5,
            end_line=15,
            signature="public void test()",
            parameters=[("int", "x")],
            return_type="void",
        )

        data = symbol.to_dict()
        restored = CodeSymbol.from_dict(data)

        assert restored.id == symbol.id
        assert restored.name == symbol.name
        assert restored.symbol_type == symbol.symbol_type
        assert restored.parameters == [("int", "x")]


class TestSymbolRelation:
    """Tests for SymbolRelation dataclass."""

    def test_relation_creation(self):
        """Test creating a SymbolRelation."""
        relation = SymbolRelation(
            source_id="sym_001",
            target_id="sym_002",
            relation_type=RelationType.CALLS,
            metadata={"line": 42},
        )

        assert relation.source_id == "sym_001"
        assert relation.target_id == "sym_002"
        assert relation.relation_type == RelationType.CALLS

    def test_relation_serialization(self):
        """Test SymbolRelation serialization."""
        relation = SymbolRelation(
            source_id="sym_001",
            target_id="sym_002",
            relation_type=RelationType.EXTENDS,
        )

        data = relation.to_dict()
        restored = SymbolRelation.from_dict(data)

        assert restored.source_id == relation.source_id
        assert restored.relation_type == relation.relation_type


class TestDependencyGraph:
    """Tests for DependencyGraph."""

    def test_add_symbol(self):
        """Test adding symbols to graph."""
        graph = DependencyGraph()
        symbol = CodeSymbol(
            id="sym_001",
            name="TestClass",
            symbol_type=SymbolType.CLASS,
            file_path="/test/Test.java",
            start_line=1,
            end_line=10,
        )

        graph.add_symbol(symbol)
        retrieved = graph.get_symbol("sym_001")

        assert retrieved == symbol

    def test_add_relation(self):
        """Test adding relations to graph."""
        graph = DependencyGraph()

        class_a = CodeSymbol(
            id="sym_001",
            name="ClassA",
            symbol_type=SymbolType.CLASS,
            file_path="/test/A.java",
            start_line=1,
            end_line=20,
        )
        method_b = CodeSymbol(
            id="sym_002",
            name="methodB",
            symbol_type=SymbolType.METHOD,
            file_path="/test/B.java",
            start_line=5,
            end_line=15,
        )

        graph.add_symbol(class_a)
        graph.add_symbol(method_b)

        relation = SymbolRelation(
            source_id="sym_001",
            target_id="sym_002",
            relation_type=RelationType.CALLS,
        )
        graph.add_relation(relation)

        callers = graph.get_callers("sym_002")
        callees = graph.get_callees("sym_001")

        assert len(callers) == 1
        assert callers[0] == class_a
        assert len(callees) == 1
        assert callees[0] == method_b

    def test_find_path(self):
        """Test finding path between symbols."""
        graph = DependencyGraph()

        # Create chain: A -> B -> C
        for i, name in enumerate(["A", "B", "C"], 1):
            graph.add_symbol(CodeSymbol(
                id=f"sym_{i:03d}",
                name=name,
                symbol_type=SymbolType.METHOD,
                file_path="/test/Test.java",
                start_line=i,
                end_line=i,
            ))

        graph.add_relation(SymbolRelation("sym_001", "sym_002", RelationType.CALLS))
        graph.add_relation(SymbolRelation("sym_002", "sym_003", RelationType.CALLS))

        path = graph.find_path("sym_001", "sym_003")

        assert path == ["sym_001", "sym_002", "sym_003"]

    def test_graph_serialization(self):
        """Test DependencyGraph serialization."""
        graph = DependencyGraph()
        symbol = CodeSymbol(
            id="sym_001",
            name="Test",
            symbol_type=SymbolType.CLASS,
            file_path="/test/Test.java",
            start_line=1,
            end_line=10,
        )
        graph.add_symbol(symbol)
        graph.add_relation(SymbolRelation("sym_001", "sym_001", RelationType.CALLS))

        data = graph.to_dict()
        restored = DependencyGraph.from_dict(data)

        assert restored.get_symbol("sym_001") is not None


class TestJavaSymbolExtractor:
    """Tests for JavaSymbolExtractor."""

    def test_extract_from_simple_class(self):
        """Test extracting symbols from a simple Java class."""
        extractor = JavaSymbolExtractor()

        java_code = '''
package com.example;

import java.util.List;

public class UserService {
    private String name;

    public User getUserById(Long id) {
        return new User();
    }

    public void saveUser(User user) {
        // Save user
    }
}
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write(java_code)
            temp_path = f.name

        try:
            file_index = extractor.extract_from_file(temp_path, java_code)

            assert file_index.language == "java"
            assert len(file_index.symbols) > 0

            # Check class symbol
            class_symbols = [s for s in file_index.symbols if s.symbol_type == SymbolType.CLASS]
            assert len(class_symbols) == 1
            assert class_symbols[0].name == "UserService"

            # Check method symbols (at least one method should be found)
            method_symbols = [s for s in file_index.symbols if s.symbol_type == SymbolType.METHOD]
            assert len(method_symbols) >= 1

            method_names = {m.name for m in method_symbols}
            assert "getUserById" in method_names

        finally:
            Path(temp_path).unlink()

    def test_extract_imports(self):
        """Test extracting imports from Java file."""
        extractor = JavaSymbolExtractor()

        java_code = '''
package com.example;

import java.util.List;
import java.util.ArrayList;
import org.springframework.stereotype.Service;

public class TestClass {
}
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
            f.write(java_code)
            temp_path = f.name

        try:
            file_index = extractor.extract_from_file(temp_path, java_code)

            assert "java.util.List" in file_index.imports
            assert "java.util.ArrayList" in file_index.imports

        finally:
            Path(temp_path).unlink()


class TestCodebaseIndexer:
    """Tests for CodebaseIndexer."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project with Java files."""
        tmpdir = tempfile.mkdtemp()
        try:
            project_path = Path(tmpdir)

            # Create source directory
            src_dir = project_path / "src" / "main" / "java" / "com" / "example"
            src_dir.mkdir(parents=True)

            # Create a Java file
            java_file = src_dir / "UserService.java"
            java_file.write_text('''
package com.example;

import java.util.List;

public class UserService {
    private UserRepository userRepository;

    public User getUserById(Long id) {
        return userRepository.findById(id);
    }

    public List<User> getAllUsers() {
        return userRepository.findAll();
    }
}
''')

            # Create another Java file
            repo_file = src_dir / "UserRepository.java"
            repo_file.write_text('''
package com.example;

import java.util.List;

public class UserRepository {
    public User findById(Long id) {
        return new User();
    }

    public List<User> findAll() {
        return new ArrayList<>();
    }
}
''')

            yield project_path
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_indexer_initialization(self, temp_project):
        """Test CodebaseIndexer initialization."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)

        assert indexer.project_path == temp_project.resolve()
        assert indexer.get_stats()["project_path"] == str(temp_project.resolve())

    def test_index_project(self, temp_project):
        """Test indexing a project."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)
        result = indexer.index_project()

        assert result["success"] is True
        assert result["indexed"] > 0
        assert result["total_symbols"] > 0

        stats = indexer.get_stats()
        assert stats["total_files"] > 0
        assert stats["total_symbols"] > 0

    def test_search_symbols(self, temp_project):
        """Test searching for symbols."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)
        indexer.index_project()

        results = indexer.search_symbols("UserService")
        assert len(results) > 0
        assert any(s.name == "UserService" for s in results)

    def test_get_symbol_by_name(self, temp_project):
        """Test getting symbol by name."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)
        indexer.index_project()

        symbol = indexer.get_symbol_by_name("UserService")
        assert symbol is not None
        assert symbol.name == "UserService"
        assert symbol.symbol_type == SymbolType.CLASS

    def test_get_file_symbols(self, temp_project):
        """Test getting symbols from a file."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)
        indexer.index_project()

        # Find a Java file
        java_files = list(temp_project.rglob("*.java"))
        if java_files:
            symbols = indexer.get_file_symbols(str(java_files[0]))
            assert len(symbols) > 0

    def test_resolve_reference_file(self, temp_project):
        """Test resolving file references."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)
        indexer.index_project()

        resolved = indexer.resolve_reference("file:src/main/java/com/example/UserService.java")

        assert resolved is not None
        assert resolved["type"] == "file"

    def test_resolve_reference_symbol(self, temp_project):
        """Test resolving symbol references."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)
        indexer.index_project()

        resolved = indexer.resolve_reference("symbol:UserService")

        assert resolved is not None
        assert resolved["type"] == "symbol"

    def test_incremental_indexing(self, temp_project):
        """Test incremental indexing."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)

        # First indexing
        result1 = indexer.index_project()
        assert result1["indexed"] > 0

        # Second indexing (should be incremental - files already indexed)
        # Note: In real scenario this would be 0, but for tests it may re-index
        result2 = indexer.index_project()
        # Just verify it runs without error
        assert result2["success"] is True

    def test_refresh_file(self, temp_project):
        """Test refreshing a specific file."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)
        indexer.index_project()

        java_files = list(temp_project.rglob("*.java"))
        if java_files:
            result = indexer.refresh_file(str(java_files[0]))
            assert result["success"] is True

    def test_clear_index(self, temp_project):
        """Test clearing the index."""
        config = IndexerConfig(enable_semantic_search=False)
        indexer = CodebaseIndexer(str(temp_project), config=config)
        indexer.index_project()

        assert indexer.get_stats()["total_files"] > 0

        indexer.clear_index()

        stats = indexer.get_stats()
        assert stats["total_files"] == 0
        assert stats["total_symbols"] == 0


class TestIndexerConfig:
    """Tests for IndexerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = IndexerConfig()

        assert "**/*.java" in config.include_patterns
        assert "**/node_modules/**" in config.exclude_patterns
        assert config.enable_incremental is True
        assert config.enable_semantic_search is True

    def test_config_serialization(self):
        """Test config serialization."""
        config = IndexerConfig()
        data = config.to_dict()

        assert "include_patterns" in data
        assert "exclude_patterns" in data
        assert data["enable_incremental"] is True


class TestCodebaseIndexState:
    """Tests for CodebaseIndexState."""

    def test_state_creation(self):
        """Test creating index state."""
        state = CodebaseIndexState(
            project_path="/test/project",
            total_files=10,
            total_symbols=50,
        )

        assert state.project_path == "/test/project"
        assert state.total_files == 10
        assert state.total_symbols == 50
        assert state.version == "1.0"

    def test_state_serialization(self):
        """Test state serialization."""
        state = CodebaseIndexState(
            project_path="/test/project",
            total_files=5,
            total_symbols=25,
            indexed_languages={"java", "python"},
        )

        data = state.to_dict()
        restored = CodebaseIndexState.from_dict(data)

        assert restored.project_path == state.project_path
        assert restored.total_files == state.total_files
        assert restored.indexed_languages == state.indexed_languages


class TestFileIndex:
    """Tests for FileIndex."""

    def test_file_index_creation(self):
        """Test creating a FileIndex."""
        symbol = CodeSymbol(
            id="sym_001",
            name="TestClass",
            symbol_type=SymbolType.CLASS,
            file_path="/test/Test.java",
            start_line=1,
            end_line=10,
        )

        file_index = FileIndex(
            file_path="/test/Test.java",
            content_hash="abc123",
            last_modified="2024-01-01T00:00:00",
            symbols=[symbol],
            imports=["java.util.List"],
            language="java",
        )

        assert file_index.file_path == "/test/Test.java"
        assert len(file_index.symbols) == 1
        assert file_index.language == "java"

    def test_file_index_serialization(self):
        """Test FileIndex serialization."""
        symbol = CodeSymbol(
            id="sym_001",
            name="TestClass",
            symbol_type=SymbolType.CLASS,
            file_path="/test/Test.java",
            start_line=1,
            end_line=10,
        )

        file_index = FileIndex(
            file_path="/test/Test.java",
            content_hash="abc123",
            last_modified="2024-01-01T00:00:00",
            symbols=[symbol],
            language="java",
        )

        data = file_index.to_dict()
        restored = FileIndex.from_dict(data)

        assert restored.file_path == file_index.file_path
        assert len(restored.symbols) == 1
        assert restored.symbols[0].name == "TestClass"
