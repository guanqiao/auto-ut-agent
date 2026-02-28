"""Unit tests for code_editor module."""

import pytest
from pyutagent.tools.code_editor import (
    DiffParser, CodeEditor, TestCodeEditor,
    EditOperation, EditResult, dedent_text,
    create_edit_prompt
)


class TestDiffParser:
    """Tests for DiffParser class."""
    
    def test_parse_simple_diff(self):
        """Test parsing a simple diff."""
        diff_text = """<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE"""
        
        edits = DiffParser.parse(diff_text)
        
        assert len(edits) == 1
        assert edits[0].search_text.strip() == "old code"
        assert edits[0].replace_text.strip() == "new code"
    
    def test_parse_diff_with_file_path(self):
        """Test parsing diff with file path."""
        diff_text = """### Test.java
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE"""
        
        edits = DiffParser.parse(diff_text)
        
        assert len(edits) == 1
        assert edits[0].file_path == "Test.java"
        assert edits[0].search_text.strip() == "old code"
    
    def test_parse_multiple_diffs(self):
        """Test parsing multiple diffs."""
        diff_text = """### Test.java
<<<<<<< SEARCH
old1
=======
new1
>>>>>>> REPLACE

### Test.java
<<<<<<< SEARCH
old2
=======
new2
>>>>>>> REPLACE"""
        
        edits = DiffParser.parse(diff_text)
        
        assert len(edits) == 2
        assert edits[0].search_text.strip() == "old1"
        assert edits[1].search_text.strip() == "old2"
    
    def test_create_diff(self):
        """Test creating diff string."""
        diff = DiffParser.create_diff(
            search_text="old code",
            replace_text="new code",
            file_path="Test.java"
        )
        
        assert "### Test.java" in diff
        assert "<<<<<<< SEARCH" in diff
        assert "old code" in diff
        assert "=======" in diff
        assert "new code" in diff
        assert ">>>>>>> REPLACE" in diff


class TestDedentText:
    """Tests for dedent_text function."""
    
    def test_remove_common_indentation(self):
        """Test removing common indentation."""
        text = """
    line1
    line2
    line3
        """.strip()
        
        result = dedent_text(text)
        
        assert result == "line1\nline2\nline3"
    
    def test_no_indentation(self):
        """Test text with no indentation."""
        text = "line1\nline2\nline3"
        
        result = dedent_text(text)
        
        assert result == "line1\nline2\nline3"
    
    def test_empty_lines(self):
        """Test handling empty lines."""
        text = """
    line1

    line2
        """.strip()
        
        result = dedent_text(text)
        
        assert "line1" in result
        assert "line2" in result


class TestCodeEditor:
    """Tests for CodeEditor class."""
    
    @pytest.fixture
    def editor(self):
        """Create a CodeEditor instance."""
        return CodeEditor()
    
    def test_apply_single_edit_exact_match(self, editor):
        """Test applying a single edit with exact match."""
        content = """public class Test {
    public void method() {
        old code;
    }
}"""
        
        edit = EditOperation(
            search_text="        old code;",
            replace_text="        new code;"
        )
        
        result = editor.apply_edits(content, [edit])
        
        assert result.success
        assert "new code" in result.modified_content
        assert "old code" not in result.modified_content
    
    def test_apply_single_edit_multiline(self, editor):
        """Test applying a multiline edit."""
        content = """public class Test {
    public void method() {
        line1;
        line2;
    }
}"""
        
        edit = EditOperation(
            search_text="        line1;\n        line2;",
            replace_text="        newLine1;\n        newLine2;"
        )
        
        result = editor.apply_edits(content, [edit])
        
        assert result.success
        assert "newLine1" in result.modified_content
        assert "newLine2" in result.modified_content
    
    def test_apply_multiple_edits(self, editor):
        """Test applying multiple edits."""
        content = """public class Test {
    public void method1() {
        old1;
    }
    public void method2() {
        old2;
    }
}"""
        
        edits = [
            EditOperation(
                search_text="        old1;",
                replace_text="        new1;"
            ),
            EditOperation(
                search_text="        old2;",
                replace_text="        new2;"
            )
        ]
        
        result = editor.apply_edits(content, edits)
        
        assert result.success
        assert "new1" in result.modified_content
        assert "new2" in result.modified_content
    
    def test_apply_edit_not_found(self, editor):
        """Test applying an edit when search text is not found."""
        content = "public class Test {}"
        
        edit = EditOperation(
            search_text="nonexistent code",
            replace_text="new code"
        )
        
        result = editor.apply_edits(content, [edit])
        
        assert not result.success
        assert len(result.edits_failed) == 1
    
    def test_apply_edit_syntax_validation(self, editor):
        """Test syntax validation after edit."""
        content = """public class Test {
    public void method() {
        int x = 1;
    }
}"""
        
        # Valid edit
        edit = EditOperation(
            search_text="        int x = 1;",
            replace_text="        int x = 2;"
        )
        
        result = editor.apply_edits(content, [edit], validate=True)
        
        assert result.success
    
    def test_undo_edit(self, editor, tmp_path):
        """Test undoing an edit."""
        # Create a temporary file
        test_file = tmp_path / "Test.java"
        original_content = "public class Test { old; }"
        test_file.write_text(original_content)
        
        # Apply diff
        diff_text = """<<<<<<< SEARCH
public class Test { old; }
=======
public class Test { new; }
>>>>>>> REPLACE"""
        
        result = editor.apply_diff_to_file(str(test_file), diff_text, backup=True)
        
        assert result.success
        assert test_file.read_text() == "public class Test { new; }"
        
        # Undo
        undo_success = editor.undo_last_edit(str(test_file))
        
        assert undo_success
        assert test_file.read_text() == original_content


class TestTestCodeEditor:
    """Tests for TestCodeEditor class."""
    
    @pytest.fixture
    def editor(self):
        """Create a TestCodeEditor instance."""
        return TestCodeEditor()
    
    def test_validate_test_structure_valid(self, editor):
        """Test validating valid test structure."""
        code = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    void testMethod() {
        assertTrue(true);
    }
}"""
        
        is_valid = editor._validate_test_structure(code)
        
        assert is_valid
    
    def test_validate_test_structure_no_class(self, editor):
        """Test validating code without class."""
        code = "@Test\nvoid testMethod() {}"
        
        is_valid = editor._validate_test_structure(code)
        
        assert not is_valid
    
    def test_validate_test_structure_unbalanced_braces(self, editor):
        """Test validating code with unbalanced braces."""
        code = """public class Test {
    @Test
    void testMethod() {
        assertTrue(true);
    // Missing closing brace
}"""
        
        is_valid = editor._validate_test_structure(code)
        
        assert not is_valid
    
    def test_apply_test_fixes(self, editor):
        """Test applying test fixes."""
        test_code = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    void testMethod() {
        int x = 1;
    }
}"""
        
        diff_text = """<<<<<<< SEARCH
        int x = 1;
=======
        int x = 2;
        assertEquals(2, x);
>>>>>>> REPLACE"""
        
        result = editor.apply_test_fixes(test_code, {}, diff_text)
        
        assert result.success
        assert "assertEquals(2, x)" in result.modified_content
    
    def test_generate_import_edit(self, editor):
        """Test generating import edit."""
        test_code = """package com.example;

import org.junit.jupiter.api.Test;

public class MyTest {
}"""
        
        edit = editor.generate_import_edit(
            "import org.mockito.Mockito;",
            test_code
        )
        
        assert edit is not None
        assert "import org.mockito.Mockito;" in edit.replace_text
    
    def test_generate_import_edit_already_exists(self, editor):
        """Test generating import edit when import already exists."""
        test_code = """import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

public class MyTest {
}"""
        
        edit = editor.generate_import_edit(
            "import org.mockito.Mockito;",
            test_code
        )
        
        assert edit is None


class TestCreateEditPrompt:
    """Tests for create_edit_prompt function."""
    
    def test_create_prompt_contains_error(self):
        """Test that prompt contains error message."""
        code = "public class Test {}"
        error = "Cannot find symbol"
        instructions = "Add the missing import"
        
        prompt = create_edit_prompt(code, error, instructions)
        
        assert error in prompt
        assert instructions in prompt
        assert code in prompt
        assert "<<<<<<< SEARCH" in prompt
    
    def test_create_prompt_format(self):
        """Test prompt format."""
        code = "public class Test {}"
        error = "Error"
        instructions = "Fix it"
        
        prompt = create_edit_prompt(code, error, instructions)
        
        assert "## Error Message" in prompt
        assert "## Fix Instructions" in prompt
        assert "## Current Test Code" in prompt
        assert "## Task" in prompt


class TestEditOperation:
    """Tests for EditOperation dataclass."""
    
    def test_edit_operation_normalization(self):
        """Test that edit operation normalizes text."""
        edit = EditOperation(
            search_text="    line1\n    line2",
            replace_text="    new1\n    new2"
        )
        
        # Should have common indentation removed
        assert edit.search_text.startswith("line1") or "    line1" in edit.search_text
        assert edit.replace_text.startswith("new1") or "    new1" in edit.replace_text
    
    def test_edit_operation_defaults(self):
        """Test edit operation default values."""
        edit = EditOperation(
            search_text="old",
            replace_text="new"
        )
        
        from pyutagent.tools.code_editor import EditStatus
        assert edit.status == EditStatus.PENDING
        assert edit.file_path is None
        assert edit.error_message == ""


class TestIntegration:
    """Integration tests for the code editor."""
    
    def test_full_workflow_compilation_fix(self):
        """Test full workflow for fixing compilation error."""
        editor = TestCodeEditor()
        
        # Original code with missing import
        original_code = """public class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(3, calc.add(1, 2));
    }
}"""
        
        # Diff to add import
        diff_text = """<<<<<<< SEARCH
public class CalculatorTest {
=======
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class CalculatorTest {
>>>>>>> REPLACE"""
        
        result = editor.apply_test_fixes(original_code, {}, diff_text)
        
        assert result.success
        assert "import org.junit.jupiter.api.Test;" in result.modified_content
        assert "import static org.junit.jupiter.api.Assertions.assertEquals;" in result.modified_content
    
    def test_full_workflow_test_fix(self):
        """Test full workflow for fixing test failure."""
        editor = TestCodeEditor()
        
        # Original code with wrong assertion
        original_code = """import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(1, 2));
    }
}"""
        
        # Diff to fix assertion
        diff_text = """<<<<<<< SEARCH
        assertEquals(4, calc.add(1, 2));
=======
        assertEquals(3, calc.add(1, 2));
>>>>>>> REPLACE"""
        
        result = editor.apply_test_fixes(original_code, {}, diff_text)
        
        assert result.success
        assert "assertEquals(3, calc.add(1, 2))" in result.modified_content
        assert "assertEquals(4" not in result.modified_content
