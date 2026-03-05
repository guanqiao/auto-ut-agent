"""Tests for the symbol indexer service."""

import tempfile
import time
from pathlib import Path

import pytest

from pyutagent.ui.services.symbol_indexer import (
    SymbolIndexer,
    SymbolIndexEntry,
    SymbolIndexStats,
    Language,
    PythonSymbolParser,
    JavaScriptSymbolParser,
    GoSymbolParser,
    RustSymbolParser,
)
from pyutagent.indexing.codebase_indexer import SymbolType


class TestSymbolIndexEntry:
    """Tests for SymbolIndexEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a symbol index entry."""
        entry = SymbolIndexEntry(
            id="test#class#MyClass",
            name="MyClass",
            full_name="module.MyClass",
            symbol_type=SymbolType.CLASS,
            language=Language.PYTHON,
            file_path="/test/module.py",
            start_line=10,
            end_line=20,
            signature="class MyClass",
        )

        assert entry.id == "test#class#MyClass"
        assert entry.name == "MyClass"
        assert entry.symbol_type == SymbolType.CLASS
        assert entry.language == Language.PYTHON

    def test_entry_to_dict(self):
        """Test converting entry to dict."""
        entry = SymbolIndexEntry(
            id="test#method#foo",
            name="foo",
            full_name="module.foo",
            symbol_type=SymbolType.METHOD,
            language=Language.PYTHON,
            file_path="/test/module.py",
            start_line=5,
            end_line=10,
        )

        data = entry.to_dict()
        assert data["id"] == "test#method#foo"
        assert data["name"] == "foo"
        assert data["symbol_type"] == "method"
        assert data["language"] == "python"

    def test_entry_from_dict(self):
        """Test creating entry from dict."""
        data = {
            "id": "test#class#Test",
            "name": "Test",
            "full_name": "test.Test",
            "symbol_type": "class",
            "language": "python",
            "file_path": "/test.py",
            "start_line": 1,
            "end_line": 10,
            "signature": "class Test",
            "modifiers": [],
            "parameters": [],
            "return_type": None,
            "parent_name": None,
            "content_hash": "abc123",
            "last_accessed": time.time(),
            "access_count": 5,
        }

        entry = SymbolIndexEntry.from_dict(data)
        assert entry.name == "Test"
        assert entry.symbol_type == SymbolType.CLASS
        assert entry.access_count == 5

    def test_mark_accessed(self):
        """Test marking entry as accessed."""
        entry = SymbolIndexEntry(
            id="test",
            name="test",
            full_name="test",
            symbol_type=SymbolType.METHOD,
            language=Language.PYTHON,
            file_path="/test.py",
            start_line=1,
            end_line=1,
        )

        initial_count = entry.access_count
        initial_time = entry.last_accessed

        time.sleep(0.01)  # Small delay
        entry.mark_accessed()

        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_time

    def test_priority_score(self):
        """Test priority score calculation."""
        entry = SymbolIndexEntry(
            id="test",
            name="test",
            full_name="test",
            symbol_type=SymbolType.METHOD,
            language=Language.PYTHON,
            file_path="/test.py",
            start_line=1,
            end_line=1,
        )

        # Initially should have some priority
        initial_score = entry.priority_score
        assert 0 <= initial_score <= 1

        # After marking accessed multiple times, score should increase
        for _ in range(10):
            entry.mark_accessed()

        assert entry.priority_score > initial_score


class TestPythonSymbolParser:
    """Tests for Python symbol parser."""

    def test_parse_class(self):
        """Test parsing Python class."""
        parser = PythonSymbolParser()
        code = '''
class MyClass:
    """A test class."""
    pass
'''
        symbols = parser.parse("/test.py", code)

        classes = [s for s in symbols if s.symbol_type == SymbolType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "MyClass"
        assert "A test class" in classes[0].docstring

    def test_parse_method(self):
        """Test parsing Python method."""
        parser = PythonSymbolParser()
        code = '''
class MyClass:
    def my_method(self, x: int) -> str:
        """A test method."""
        return str(x)
'''
        symbols = parser.parse("/test.py", code)

        methods = [s for s in symbols if s.symbol_type == SymbolType.METHOD]
        assert len(methods) == 1
        assert methods[0].name == "my_method"
        assert methods[0].return_type == "str"
        assert len(methods[0].parameters) == 1

    def test_parse_function(self):
        """Test parsing module-level function."""
        parser = PythonSymbolParser()
        code = '''
def standalone_func(a, b):
    """A standalone function."""
    return a + b
'''
        symbols = parser.parse("/test.py", code)

        funcs = [s for s in symbols if s.symbol_type == SymbolType.METHOD]
        assert len(funcs) == 1
        assert funcs[0].name == "standalone_func"

    def test_parse_constructor(self):
        """Test parsing __init__ as constructor."""
        parser = PythonSymbolParser()
        code = '''
class MyClass:
    def __init__(self, name: str):
        self.name = name
'''
        symbols = parser.parse("/test.py", code)

        constructors = [s for s in symbols if s.symbol_type == SymbolType.CONSTRUCTOR]
        assert len(constructors) == 1
        assert constructors[0].name == "__init__"

    def test_parse_with_syntax_error(self):
        """Test parsing code with syntax error."""
        parser = PythonSymbolParser()
        code = 'def broken(  # incomplete'

        symbols = parser.parse("/test.py", code)
        assert len(symbols) == 0


class TestJavaScriptSymbolParser:
    """Tests for JavaScript/TypeScript symbol parser."""

    def test_parse_js_class(self):
        """Test parsing JavaScript class."""
        parser = JavaScriptSymbolParser()
        code = '''
class UserService {
    getUser(id) {
        return { id };
    }
}
'''
        symbols = parser.parse("/test.js", code, is_typescript=False)

        classes = [s for s in symbols if s.symbol_type == SymbolType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "UserService"

    def test_parse_ts_interface(self):
        """Test parsing TypeScript interface."""
        parser = JavaScriptSymbolParser()
        code = '''
interface User {
    id: number;
    name: string;
}
'''
        symbols = parser.parse("/test.ts", code, is_typescript=True)

        interfaces = [s for s in symbols if s.symbol_type == SymbolType.INTERFACE]
        assert len(interfaces) == 1
        assert interfaces[0].name == "User"

    def test_parse_function(self):
        """Test parsing JavaScript function."""
        parser = JavaScriptSymbolParser()
        code = '''
function calculateSum(a: number, b: number): number {
    return a + b;
}
'''
        symbols = parser.parse("/test.ts", code, is_typescript=True)

        funcs = [s for s in symbols if s.symbol_type == SymbolType.METHOD]
        assert len(funcs) == 1
        assert funcs[0].name == "calculateSum"


class TestGoSymbolParser:
    """Tests for Go symbol parser."""

    def test_parse_struct(self):
        """Test parsing Go struct."""
        parser = GoSymbolParser()
        code = '''
type User struct {
    ID   int
    Name string
}
'''
        symbols = parser.parse("/test.go", code)

        structs = [s for s in symbols if s.symbol_type == SymbolType.CLASS]
        assert len(structs) == 1
        assert structs[0].name == "User"

    def test_parse_interface(self):
        """Test parsing Go interface."""
        parser = GoSymbolParser()
        code = '''
type Reader interface {
    Read(p []byte) (n int, err error)
}
'''
        symbols = parser.parse("/test.go", code)

        interfaces = [s for s in symbols if s.symbol_type == SymbolType.INTERFACE]
        assert len(interfaces) == 1
        assert interfaces[0].name == "Reader"

    def test_parse_function(self):
        """Test parsing Go function."""
        parser = GoSymbolParser()
        code = '''
func GetUser(id int) (*User, error) {
    return nil, nil
}
'''
        symbols = parser.parse("/test.go", code)

        funcs = [s for s in symbols if s.symbol_type == SymbolType.METHOD]
        assert len(funcs) == 1
        assert funcs[0].name == "GetUser"


class TestRustSymbolParser:
    """Tests for Rust symbol parser."""

    def test_parse_struct(self):
        """Test parsing Rust struct."""
        parser = RustSymbolParser()
        code = '''
pub struct User {
    id: u64,
    name: String,
}
'''
        symbols = parser.parse("/test.rs", code)

        structs = [s for s in symbols if s.symbol_type == SymbolType.CLASS]
        assert len(structs) == 1
        assert structs[0].name == "User"

    def test_parse_enum(self):
        """Test parsing Rust enum."""
        parser = RustSymbolParser()
        code = '''
pub enum Status {
    Active,
    Inactive,
}
'''
        symbols = parser.parse("/test.rs", code)

        enums = [s for s in symbols if s.symbol_type == SymbolType.ENUM]
        assert len(enums) == 1
        assert enums[0].name == "Status"

    def test_parse_trait(self):
        """Test parsing Rust trait."""
        parser = RustSymbolParser()
        code = '''
pub trait Drawable {
    fn draw(&self);
}
'''
        symbols = parser.parse("/test.rs", code)

        traits = [s for s in symbols if s.symbol_type == SymbolType.INTERFACE]
        assert len(traits) == 1
        assert traits[0].name == "Drawable"

    def test_parse_function(self):
        """Test parsing Rust function."""
        parser = RustSymbolParser()
        code = '''
pub fn process_data(input: &str) -> Result<String, Error> {
    Ok(input.to_string())
}
'''
        symbols = parser.parse("/test.rs", code)

        funcs = [s for s in symbols if s.symbol_type == SymbolType.METHOD]
        assert len(funcs) == 1
        assert funcs[0].name == "process_data"


class TestSymbolIndexer:
    """Tests for SymbolIndexer."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project with various source files."""
        tmpdir = tempfile.mkdtemp()
        try:
            project_path = Path(tmpdir)

            # Create Python file
            py_file = project_path / "module.py"
            py_file.write_text('''
class UserService:
    def get_user(self, id: int) -> dict:
        return {"id": id}

def helper():
    pass
''')

            # Create JavaScript file
            js_file = project_path / "app.js"
            js_file.write_text('''
class Component {
    render() {
        return null;
    }
}

function init() {
    return new Component();
}
''')

            yield project_path
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_indexer_initialization(self, temp_project):
        """Test indexer initialization."""
        indexer = SymbolIndexer(str(temp_project))
        assert indexer.project_path == temp_project
        assert indexer.index_dir.exists()

    def test_index_file_python(self, temp_project):
        """Test indexing a Python file."""
        indexer = SymbolIndexer(str(temp_project))
        py_file = temp_project / "module.py"

        result = indexer.index_file(py_file)
        assert result is True

        # Check symbols were indexed
        symbols = indexer.get_file_symbols(str(py_file))
        assert len(symbols) >= 2  # At least class and method

    def test_index_file_javascript(self, temp_project):
        """Test indexing a JavaScript file."""
        indexer = SymbolIndexer(str(temp_project))
        js_file = temp_project / "app.js"

        result = indexer.index_file(js_file)
        assert result is True

        symbols = indexer.get_file_symbols(str(js_file))
        assert len(symbols) >= 1  # At least the class

    def test_search_symbols(self, temp_project):
        """Test searching symbols."""
        indexer = SymbolIndexer(str(temp_project))
        indexer.index_project()

        results = indexer.search("UserService")
        assert len(results) >= 1
        assert any(s.name == "UserService" for s in results)

    def test_search_with_fuzzy_match(self, temp_project):
        """Test fuzzy matching in search."""
        indexer = SymbolIndexer(str(temp_project))
        indexer.index_project()

        results = indexer.search("UsrSvc", fuzzy=True)  # Fuzzy match
        assert len(results) >= 1

    def test_get_symbol_by_name(self, temp_project):
        """Test getting symbol by name."""
        indexer = SymbolIndexer(str(temp_project))
        indexer.index_project()

        symbol = indexer.get_symbol_by_name("UserService")
        assert symbol is not None
        assert symbol.name == "UserService"

    def test_get_symbols_by_type(self, temp_project):
        """Test getting symbols by type."""
        indexer = SymbolIndexer(str(temp_project))
        indexer.index_project()

        classes = indexer.get_symbols_by_type(SymbolType.CLASS)
        assert len(classes) >= 1

    def test_recent_symbols(self, temp_project):
        """Test recent symbols tracking."""
        indexer = SymbolIndexer(str(temp_project))
        indexer.index_project()

        # Access some symbols
        symbols = indexer.search("get_user")
        if symbols:
            indexer.get_symbol(symbols[0].id)

        recent = indexer.get_recent_symbols()
        assert len(recent) >= 1

    def test_get_stats(self, temp_project):
        """Test getting index statistics."""
        indexer = SymbolIndexer(str(temp_project))
        indexer.index_project()

        stats = indexer.get_stats()
        assert stats.total_symbols >= 2
        assert stats.total_files >= 2
        assert "python" in stats.languages

    def test_clear_index(self, temp_project):
        """Test clearing the index."""
        indexer = SymbolIndexer(str(temp_project))
        indexer.index_project()

        assert indexer.get_stats().total_symbols > 0

        indexer.clear()

        assert indexer.get_stats().total_symbols == 0

    def test_incremental_update(self, temp_project):
        """Test incremental indexing (no change)."""
        indexer = SymbolIndexer(str(temp_project))
        py_file = temp_project / "module.py"

        # First index
        result1 = indexer.index_file(py_file)
        assert result1 is True

        # Second index (no change) - should return True but skip processing
        result2 = indexer.index_file(py_file)
        assert result2 is True

    def test_remove_file(self, temp_project):
        """Test removing file from index."""
        indexer = SymbolIndexer(str(temp_project))
        py_file = temp_project / "module.py"

        indexer.index_file(py_file)
        assert len(indexer.get_file_symbols(str(py_file))) > 0

        indexer.remove_file(str(py_file))
        assert len(indexer.get_file_symbols(str(py_file))) == 0


class TestSymbolIndexerPerformance:
    """Performance tests for SymbolIndexer."""

    @pytest.fixture
    def large_project(self):
        """Create a larger project for performance testing."""
        tmpdir = tempfile.mkdtemp()
        try:
            project_path = Path(tmpdir)

            # Create multiple Python files
            for i in range(10):
                py_file = project_path / f"module_{i}.py"
                py_file.write_text(f'''
class Service{i}:
    def method_1(self):
        pass
    
    def method_2(self):
        pass

def function_{i}():
    pass
''')

            yield project_path
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_search_performance(self, large_project):
        """Test search performance (should be < 200ms)."""
        import time

        indexer = SymbolIndexer(str(large_project))
        indexer.index_project()

        start = time.time()
        results = indexer.search("Service", limit=10)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 200, f"Search took {elapsed:.1f}ms, expected < 200ms"
        assert len(results) > 0

    def test_index_project_performance(self, large_project):
        """Test project indexing performance."""
        import time

        indexer = SymbolIndexer(str(large_project))

        start = time.time()
        stats = indexer.index_project()
        elapsed = (time.time() - start) * 1000

        # Indexing 10 files should be reasonably fast
        assert elapsed < 5000, f"Indexing took {elapsed:.1f}ms"
        assert stats.total_symbols >= 30  # 3 symbols per file * 10 files


class TestSymbolIndexerPersistence:
    """Tests for index persistence."""

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project."""
        tmpdir = tempfile.mkdtemp()
        try:
            project_path = Path(tmpdir)
            py_file = project_path / "test.py"
            py_file.write_text('''
class TestClass:
    def test_method(self):
        pass
''')
            yield project_path
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_save_and_load(self, temp_project):
        """Test saving and loading index."""
        # Create and populate indexer
        indexer1 = SymbolIndexer(str(temp_project))
        indexer1.index_project()
        stats1 = indexer1.get_stats()

        # Create new indexer (should load from disk)
        indexer2 = SymbolIndexer(str(temp_project))
        stats2 = indexer2.get_stats()

        assert stats2.total_symbols == stats1.total_symbols

    def test_refresh(self, temp_project):
        """Test refreshing the index."""
        indexer = SymbolIndexer(str(temp_project))
        indexer.index_project()

        initial_stats = indexer.get_stats()

        # Modify a file
        py_file = temp_project / "test.py"
        py_file.write_text('''
class TestClass:
    def test_method(self):
        pass
    
    def new_method(self):
        pass
''')

        # Refresh
        new_stats = indexer.refresh()
        assert new_stats.total_symbols >= initial_stats.total_symbols
