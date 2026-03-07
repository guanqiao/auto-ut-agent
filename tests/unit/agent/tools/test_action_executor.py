"""Tests for ActionExecutor import cleaning functionality."""

import pytest
from pyutagent.agent.tools.action_executor import ActionExecutor


class TestImportCleaning:
    """Tests for _clean_import_statement method."""
    
    @pytest.fixture
    def executor(self):
        """Create an ActionExecutor instance."""
        return ActionExecutor()
    
    def test_clean_simple_import(self, executor):
        """Test cleaning a simple import statement."""
        result = executor._clean_import_statement("java.sql.Connection")
        assert result == "import java.sql.Connection;"
    
    def test_clean_import_with_keyword(self, executor):
        """Test cleaning import with 'import' keyword."""
        result = executor._clean_import_statement("import java.sql.Connection;")
        assert result == "import java.sql.Connection;"
    
    def test_clean_import_with_quotes(self, executor):
        """Test cleaning import with quotes."""
        result = executor._clean_import_statement('"java.sql.Connection"')
        assert result == "import java.sql.Connection;"
    
    def test_clean_import_with_import_keyword_in_quotes(self, executor):
        """Test cleaning import with 'import' keyword in quotes."""
        result = executor._clean_import_statement('"import java.sql.Connection;"')
        assert result == "import java.sql.Connection;"
    
    def test_clean_import_with_double_import(self, executor):
        """Test cleaning malformed import with double 'import' keyword."""
        result = executor._clean_import_statement('import "import java.sql.Connection;"')
        assert result == "import java.sql.Connection;"
    
    def test_clean_import_with_array_syntax(self, executor):
        """Test cleaning import with array-like syntax."""
        result = executor._clean_import_statement('["java.sql.Connection"]')
        assert result == "import java.sql.Connection;"
    
    def test_clean_import_with_array_and_quotes(self, executor):
        """Test cleaning import with array syntax and quotes."""
        result = executor._clean_import_statement('["import java.sql.Connection;"]')
        assert result == "import java.sql.Connection;"
    
    def test_clean_static_import(self, executor):
        """Test cleaning static import."""
        result = executor._clean_import_statement("static org.junit.jupiter.api.Assertions.assertEquals")
        assert result == "import static org.junit.jupiter.api.Assertions.assertEquals;"
    
    def test_clean_static_import_with_keyword(self, executor):
        """Test cleaning static import with 'import static' keyword."""
        result = executor._clean_import_statement("import static org.junit.jupiter.api.Assertions.assertEquals;")
        assert result == "import static org.junit.jupiter.api.Assertions.assertEquals;"
    
    def test_clean_wildcard_import(self, executor):
        """Test cleaning wildcard import."""
        result = executor._clean_import_statement("java.util.*")
        assert result == "import java.util.*;"
    
    def test_clean_wildcard_import_with_semicolon(self, executor):
        """Test cleaning wildcard import with semicolon."""
        result = executor._clean_import_statement("java.util.*;")
        assert result == "import java.util.*;"
    
    def test_clean_import_with_extra_whitespace(self, executor):
        """Test cleaning import with extra whitespace."""
        result = executor._clean_import_statement("  java.sql.Connection  ")
        assert result == "import java.sql.Connection;"
    
    def test_clean_empty_import(self, executor):
        """Test cleaning empty import."""
        result = executor._clean_import_statement("")
        assert result is None
    
    def test_clean_none_import(self, executor):
        """Test cleaning None import."""
        result = executor._clean_import_statement(None)
        assert result is None
    
    def test_clean_invalid_import(self, executor):
        """Test cleaning invalid import format."""
        result = executor._clean_import_statement("invalid import format")
        assert result is None
    
    def test_clean_import_with_special_chars(self, executor):
        """Test cleaning import with special characters."""
        result = executor._clean_import_statement('import "import java.sql.Connection;"];')
        assert result == "import java.sql.Connection;"
    
    def test_clean_import_malformed_array(self, executor):
        """Test cleaning malformed array-like import."""
        result = executor._clean_import_statement('["import javax.sql.DataSource;"]')
        assert result == "import javax.sql.DataSource;"
    
    def test_clean_import_with_semicolon_in_quotes(self, executor):
        """Test cleaning import with semicolon inside quotes."""
        result = executor._clean_import_statement('"import java.sql.Connection;"')
        assert result == "import java.sql.Connection;"


class TestImportIntegration:
    """Integration tests for import fixing functionality."""
    
    @pytest.fixture
    def executor(self, tmp_path):
        """Create an ActionExecutor instance with a temporary project path."""
        return ActionExecutor(project_path=str(tmp_path))
    
    @pytest.mark.asyncio
    async def test_fix_imports_with_malformed_imports(self, executor, tmp_path):
        """Test fixing imports with various malformed formats."""
        test_file = tmp_path / "src" / "test" / "java" / "Test.java"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        initial_code = """package org.example;

public class Test {
    // test code
}
"""
        test_file.write_text(initial_code, encoding='utf-8')
        
        action = {
            'action': 'fix_imports',
            'imports': [
                'java.sql.Connection',
                '"javax.sql.DataSource"',
                'import "import java.util.List;"',
                '["org.junit.jupiter.api.Test"]'
            ]
        }
        
        context = {
            'test_file': str(test_file.relative_to(tmp_path))
        }
        
        result = await executor._fix_imports(action, context)
        
        assert result.success is True
        assert result.modified_file is not None
        
        modified_content = test_file.read_text(encoding='utf-8')
        
        assert 'import java.sql.Connection;' in modified_content
        assert 'import javax.sql.DataSource;' in modified_content
        assert 'import java.util.List;' in modified_content
        assert 'import org.junit.jupiter.api.Test;' in modified_content
        
        assert '["import' not in modified_content
        assert 'import "import' not in modified_content
