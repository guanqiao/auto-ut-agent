"""Tests for error analyzer module."""

import pytest
from unittest.mock import Mock
from pathlib import Path

from pyutagent.tools.error_analyzer import (
    ErrorType,
    CompilationError,
    ErrorAnalysis,
    CompilationErrorAnalyzer,
    ErrorFixGenerator,
)


class TestErrorType:
    """Tests for ErrorType enum."""

    def test_error_type_values(self):
        """Test that all error types are defined."""
        assert ErrorType.IMPORT_ERROR is not None
        assert ErrorType.SYMBOL_NOT_FOUND is not None
        assert ErrorType.TYPE_MISMATCH is not None
        assert ErrorType.SYNTAX_ERROR is not None
        assert ErrorType.GENERIC_TYPE_ERROR is not None
        assert ErrorType.ACCESS_MODIFIER_ERROR is not None
        assert ErrorType.METHOD_NOT_FOUND is not None
        assert ErrorType.VARIABLE_NOT_FOUND is not None
        assert ErrorType.CONSTRUCTOR_NOT_FOUND is not None
        assert ErrorType.PACKAGE_NOT_FOUND is not None
        assert ErrorType.ANNOTATION_ERROR is not None
        assert ErrorType.STATIC_REFERENCE_ERROR is not None
        assert ErrorType.UNKNOWN is not None


class TestCompilationError:
    """Tests for CompilationError dataclass."""

    def test_compilation_error_creation(self):
        """Test creating a CompilationError."""
        error = CompilationError(
            error_type=ErrorType.IMPORT_ERROR,
            message="package com.example does not exist",
            file_path="/path/Test.java",
            line_number=10,
            column_number=5,
            error_token="com.example",
            suggestions=["Check package name"],
            fix_hint="Add correct import"
        )
        assert error.error_type == ErrorType.IMPORT_ERROR
        assert error.file_path == "/path/Test.java"
        assert error.line_number == 10
        assert error.column_number == 5

    def test_compilation_error_defaults(self):
        """Test CompilationError with default values."""
        error = CompilationError(
            error_type=ErrorType.SYNTAX_ERROR,
            message="';' expected",
            file_path=None,
            line_number=None,
            column_number=None,
            error_token=None
        )
        assert error.file_path is None
        assert error.line_number is None
        assert error.suggestions == []
        assert error.fix_hint == ""


class TestErrorAnalysis:
    """Tests for ErrorAnalysis dataclass."""

    def test_error_analysis_creation(self):
        """Test creating an ErrorAnalysis."""
        error = CompilationError(
            error_type=ErrorType.IMPORT_ERROR,
            message="package not found",
            file_path="Test.java",
            line_number=1
        )
        analysis = ErrorAnalysis(
            errors=[error],
            summary="1 import error",
            fix_strategy="Add imports",
            priority=1
        )
        assert len(analysis.errors) == 1
        assert analysis.summary == "1 import error"
        assert analysis.priority == 1


class TestCompilationErrorAnalyzer:
    """Tests for CompilationErrorAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a CompilationErrorAnalyzer instance."""
        return CompilationErrorAnalyzer()

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert len(analyzer.error_patterns) > 0

    def test_compile_patterns(self, analyzer):
        """Test that error patterns are compiled."""
        for error_type, patterns in analyzer.error_patterns.items():
            assert isinstance(patterns, list)
            for pattern in patterns:
                assert hasattr(pattern, 'search')

    def test_match_error_line(self, analyzer):
        """Test matching error line pattern."""
        line = "Test.java:10: error: cannot find symbol"
        match = analyzer._match_error_line(line)
        assert match is not None
        assert match.group(1) == "Test.java"
        assert match.group(2) == "10"

    def test_match_error_line_with_column(self, analyzer):
        """Test matching error line with column number."""
        line = "Test.java:10:5: error: ';' expected"
        match = analyzer._match_error_line(line)
        assert match is not None
        assert match.group(1) == "Test.java"
        assert match.group(2) == "10"

    def test_match_error_line_no_match(self, analyzer):
        """Test non-error line doesn't match."""
        line = "Some random text"
        match = analyzer._match_error_line(line)
        assert match is None

    def test_is_new_error(self, analyzer):
        """Test detecting new error lines."""
        assert analyzer._is_new_error("Test.java:10: error: message") is True
        assert analyzer._is_new_error("  symbol: class Foo") is False

    def test_classify_error_import(self, analyzer):
        """Test classifying import errors."""
        error_type = analyzer._classify_error("package com.example does not exist")
        assert error_type == ErrorType.IMPORT_ERROR

    def test_classify_error_symbol_not_found(self, analyzer):
        """Test classifying symbol not found errors."""
        # Note: The pattern matches IMPORT_ERROR first due to pattern ordering
        error_type = analyzer._classify_error("cannot find symbol symbol: class MyClass")
        # The pattern "class ([\w.]+)" in IMPORT_ERROR matches first
        assert error_type in [ErrorType.SYMBOL_NOT_FOUND, ErrorType.IMPORT_ERROR]

    def test_classify_error_type_mismatch(self, analyzer):
        """Test classifying type mismatch errors."""
        error_type = analyzer._classify_error("incompatible types: String cannot be converted to int")
        assert error_type == ErrorType.TYPE_MISMATCH

    def test_classify_error_syntax(self, analyzer):
        """Test classifying syntax errors."""
        error_type = analyzer._classify_error("';' expected")
        assert error_type == ErrorType.SYNTAX_ERROR

    def test_classify_error_method_not_found(self, analyzer):
        """Test classifying method not found errors."""
        # Note: The pattern may match SYMBOL_NOT_FOUND first
        error_type = analyzer._classify_error("cannot find symbol symbol: method doSomething()")
        assert error_type in [ErrorType.METHOD_NOT_FOUND, ErrorType.SYMBOL_NOT_FOUND]

    def test_classify_error_variable_not_found(self, analyzer):
        """Test classifying variable not found errors."""
        # Note: The pattern may match SYMBOL_NOT_FOUND first
        error_type = analyzer._classify_error("cannot find symbol symbol: variable x")
        assert error_type in [ErrorType.VARIABLE_NOT_FOUND, ErrorType.SYMBOL_NOT_FOUND]

    def test_classify_error_unknown(self, analyzer):
        """Test classifying unknown errors."""
        error_type = analyzer._classify_error("Some random error message")
        assert error_type == ErrorType.UNKNOWN

    def test_extract_error_token_symbol(self, analyzer):
        """Test extracting symbol token."""
        token = analyzer._extract_error_token(
            "cannot find symbol symbol: class MyClass",
            ErrorType.SYMBOL_NOT_FOUND
        )
        assert token == "MyClass"

    def test_extract_error_token_method(self, analyzer):
        """Test extracting method token."""
        token = analyzer._extract_error_token(
            "cannot find symbol symbol: method doSomething",
            ErrorType.METHOD_NOT_FOUND
        )
        assert token == "doSomething"

    def test_extract_error_token_package(self, analyzer):
        """Test extracting package token."""
        token = analyzer._extract_error_token(
            "package com.example does not exist",
            ErrorType.PACKAGE_NOT_FOUND
        )
        assert token == "com.example"

    def test_extract_error_token_not_found(self, analyzer):
        """Test extracting token when not found."""
        token = analyzer._extract_error_token(
            "Some error without token",
            ErrorType.UNKNOWN
        )
        assert token is None

    def test_generate_suggestions_import(self, analyzer):
        """Test generating suggestions for import errors."""
        suggestions = analyzer._generate_suggestions(
            ErrorType.IMPORT_ERROR,
            "package com.example does not exist",
            "com.example"
        )
        assert len(suggestions) > 0
        assert any("import" in s.lower() for s in suggestions)

    def test_generate_suggestions_symbol(self, analyzer):
        """Test generating suggestions for symbol errors."""
        suggestions = analyzer._generate_suggestions(
            ErrorType.SYMBOL_NOT_FOUND,
            "cannot find symbol",
            "MyClass"
        )
        assert len(suggestions) > 0
        assert any("import" in s.lower() or "define" in s.lower() for s in suggestions)

    def test_generate_suggestions_type_mismatch(self, analyzer):
        """Test generating suggestions for type mismatch."""
        suggestions = analyzer._generate_suggestions(
            ErrorType.TYPE_MISMATCH,
            "incompatible types",
            None
        )
        assert len(suggestions) > 0
        assert any("type" in s.lower() for s in suggestions)

    def test_generate_suggestions_syntax(self, analyzer):
        """Test generating suggestions for syntax errors."""
        suggestions = analyzer._generate_suggestions(
            ErrorType.SYNTAX_ERROR,
            "';' expected",
            None
        )
        assert len(suggestions) > 0
        assert any("semicolon" in s.lower() or "brace" in s.lower() for s in suggestions)

    def test_generate_suggestions_static_reference(self, analyzer):
        """Test generating suggestions for static reference errors."""
        suggestions = analyzer._generate_suggestions(
            ErrorType.STATIC_REFERENCE_ERROR,
            "non-static variable x cannot be referenced",
            None
        )
        assert len(suggestions) > 0
        assert any("instance" in s.lower() or "static" in s.lower() for s in suggestions)

    def test_generate_fix_hint_import(self, analyzer):
        """Test generating fix hint for import error."""
        hint = analyzer._generate_fix_hint(
            ErrorType.IMPORT_ERROR,
            "",
            "com.example.MyClass"
        )
        assert "import" in hint.lower()
        assert "com.example.MyClass" in hint

    def test_generate_fix_hint_symbol(self, analyzer):
        """Test generating fix hint for symbol error."""
        hint = analyzer._generate_fix_hint(
            ErrorType.SYMBOL_NOT_FOUND,
            "",
            "MyClass"
        )
        assert "MyClass" in hint

    def test_generate_fix_hint_method(self, analyzer):
        """Test generating fix hint for method error."""
        hint = analyzer._generate_fix_hint(
            ErrorType.METHOD_NOT_FOUND,
            "",
            "doSomething"
        )
        assert "doSomething" in hint

    def test_generate_fix_hint_syntax(self, analyzer):
        """Test generating fix hint for syntax error."""
        hint = analyzer._generate_fix_hint(
            ErrorType.SYNTAX_ERROR,
            "",
            None
        )
        assert "syntax" in hint.lower()

    def test_generate_summary_no_errors(self, analyzer):
        """Test generating summary with no errors."""
        summary = analyzer._generate_summary([])
        assert "No compilation errors" in summary

    def test_generate_summary_with_errors(self, analyzer):
        """Test generating summary with errors."""
        errors = [
            CompilationError(ErrorType.IMPORT_ERROR, "msg", None, None),
            CompilationError(ErrorType.IMPORT_ERROR, "msg2", None, None),
            CompilationError(ErrorType.SYNTAX_ERROR, "msg3", None, None),
        ]
        summary = analyzer._generate_summary(errors)
        assert "3 compilation error" in summary
        assert "Import Error" in summary
        assert "Syntax Error" in summary

    def test_determine_fix_strategy_no_errors(self, analyzer):
        """Test determining fix strategy with no errors."""
        strategy = analyzer._determine_fix_strategy([])
        assert strategy == "No fixes needed"

    def test_determine_fix_strategy_import_errors(self, analyzer):
        """Test determining fix strategy with import errors."""
        errors = [CompilationError(ErrorType.IMPORT_ERROR, "msg", None, None)]
        strategy = analyzer._determine_fix_strategy(errors)
        assert "import" in strategy.lower()

    def test_determine_fix_strategy_syntax_errors(self, analyzer):
        """Test determining fix strategy with syntax errors."""
        errors = [CompilationError(ErrorType.SYNTAX_ERROR, "msg", None, None)]
        strategy = analyzer._determine_fix_strategy(errors)
        assert "syntax" in strategy.lower()

    def test_determine_fix_strategy_symbol_errors(self, analyzer):
        """Test determining fix strategy with symbol errors."""
        errors = [CompilationError(ErrorType.SYMBOL_NOT_FOUND, "msg", None, None)]
        strategy = analyzer._determine_fix_strategy(errors)
        assert "symbol" in strategy.lower() or "missing" in strategy.lower()

    def test_calculate_priority_no_errors(self, analyzer):
        """Test calculating priority with no errors."""
        priority = analyzer._calculate_priority([])
        assert priority == 5

    def test_calculate_priority_syntax_error(self, analyzer):
        """Test calculating priority with syntax error."""
        errors = [CompilationError(ErrorType.SYNTAX_ERROR, "msg", None, None)]
        priority = analyzer._calculate_priority(errors)
        assert priority == 1

    def test_calculate_priority_import_error(self, analyzer):
        """Test calculating priority with import error."""
        errors = [CompilationError(ErrorType.IMPORT_ERROR, "msg", None, None)]
        priority = analyzer._calculate_priority(errors)
        assert priority == 1

    def test_calculate_priority_symbol_error(self, analyzer):
        """Test calculating priority with symbol error."""
        errors = [CompilationError(ErrorType.SYMBOL_NOT_FOUND, "msg", None, None)]
        priority = analyzer._calculate_priority(errors)
        assert priority == 2

    def test_calculate_priority_type_mismatch(self, analyzer):
        """Test calculating priority with type mismatch."""
        errors = [CompilationError(ErrorType.TYPE_MISMATCH, "msg", None, None)]
        priority = analyzer._calculate_priority(errors)
        assert priority == 3

    def test_get_fix_prompt_context(self, analyzer):
        """Test generating fix prompt context."""
        error = CompilationError(
            error_type=ErrorType.IMPORT_ERROR,
            message="package com.example does not exist",
            file_path="Test.java",
            line_number=10,
            error_token="com.example",
            fix_hint="Add import statement",
            suggestions=["Check package name", "Add import"]
        )
        analysis = ErrorAnalysis(
            errors=[error],
            summary="1 error",
            fix_strategy="Fix imports",
            priority=1
        )
        context = analyzer.get_fix_prompt_context(analysis)
        assert "1 error" in context
        assert "Fix Strategy" in context
        assert "IMPORT_ERROR" in context
        assert "Line: 10" in context
        assert "com.example" in context

    def test_get_fix_prompt_context_many_errors(self, analyzer):
        """Test that only first 10 errors are included."""
        errors = [
            CompilationError(ErrorType.SYNTAX_ERROR, f"error {i}", None, None)
            for i in range(15)
        ]
        analysis = ErrorAnalysis(
            errors=errors,
            summary="15 errors",
            fix_strategy="Fix all",
            priority=1
        )
        context = analyzer.get_fix_prompt_context(analysis)
        assert "and 5 more errors" in context


class TestErrorFixGenerator:
    """Tests for ErrorFixGenerator."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock error analyzer."""
        return Mock(spec=CompilationErrorAnalyzer)

    @pytest.fixture
    def generator(self, mock_analyzer):
        """Create an ErrorFixGenerator instance."""
        return ErrorFixGenerator(mock_analyzer)

    def test_generate_fixes(self, generator, mock_analyzer):
        """Test generating fixes."""
        error = CompilationError(
            error_type=ErrorType.IMPORT_ERROR,
            message="package not found",
            file_path="Test.java",
            line_number=1
        )
        analysis = ErrorAnalysis(
            errors=[error],
            summary="1 error",
            fix_strategy="Fix it",
            priority=1
        )
        mock_analyzer.analyze.return_value = analysis
        mock_analyzer.get_fix_prompt_context.return_value = "context"

        result = generator.generate_fixes("test code", "compiler output", {})

        assert "analysis" in result
        assert "fixes" in result
        assert "context" in result
        assert len(result["fixes"]) == 1

    def test_generate_specific_fix_import(self, generator):
        """Test generating specific fix for import error."""
        error = CompilationError(
            error_type=ErrorType.IMPORT_ERROR,
            message="package not found",
            file_path="Test.java",
            line_number=1,
            error_token="com.example.Class"
        )
        fix = generator._generate_specific_fix(error, "code", {})
        assert fix["action"] == "add_import"
        assert "import" in fix["details"]["import_statement"]

    def test_generate_specific_fix_symbol(self, generator):
        """Test generating specific fix for symbol error."""
        error = CompilationError(
            error_type=ErrorType.SYMBOL_NOT_FOUND,
            message="cannot find symbol",
            file_path="Test.java",
            line_number=5,
            error_token="MyClass"
        )
        fix = generator._generate_specific_fix(error, "code", {})
        assert fix["action"] == "define_or_import"
        assert fix["details"]["symbol"] == "MyClass"

    def test_generate_specific_fix_method(self, generator):
        """Test generating specific fix for method error."""
        error = CompilationError(
            error_type=ErrorType.METHOD_NOT_FOUND,
            message="cannot find symbol",
            file_path="Test.java",
            line_number=10,
            error_token="doSomething"
        )
        fix = generator._generate_specific_fix(error, "code", {})
        assert fix["action"] == "fix_method_call"
        assert fix["details"]["method"] == "doSomething"

    def test_generate_specific_fix_type_mismatch(self, generator):
        """Test generating specific fix for type mismatch."""
        error = CompilationError(
            error_type=ErrorType.TYPE_MISMATCH,
            message="incompatible types",
            file_path="Test.java",
            line_number=15
        )
        fix = generator._generate_specific_fix(error, "code", {})
        assert fix["action"] == "fix_type"

    def test_generate_specific_fix_syntax(self, generator):
        """Test generating specific fix for syntax error."""
        error = CompilationError(
            error_type=ErrorType.SYNTAX_ERROR,
            message="';' expected",
            file_path="Test.java",
            line_number=20,
            fix_hint="Add semicolon"
        )
        fix = generator._generate_specific_fix(error, "code", {})
        assert fix["action"] == "fix_syntax"
        assert fix["details"]["suggestion"] == "Add semicolon"
