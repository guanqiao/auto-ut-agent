"""Tests for terminal error detection functionality."""

import pytest
from pyutagent.ui.terminal import ErrorPattern, ErrorType, TerminalError


class TestErrorPattern:
    """Test error pattern detection."""
    
    def test_detect_python_syntax_error(self):
        """Test detection of Python syntax errors."""
        output = '''
  File "test.py", line 10
    if x > 0
            ^
SyntaxError: invalid syntax
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        assert any(e.error_type == ErrorType.PYTHON_SYNTAX for e in errors)
        
    def test_detect_python_runtime_error(self):
        """Test detection of Python runtime errors."""
        output = '''
Traceback (most recent call last):
  File "test.py", line 5, in <module>
    result = 1 / 0
ZeroDivisionError: division by zero
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        assert any(e.error_type == ErrorType.PYTHON_RUNTIME for e in errors)
        
    def test_detect_python_import_error(self):
        """Test detection of Python import errors."""
        output = '''
ModuleNotFoundError: No module named 'nonexistent_module'
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        assert any(e.error_type == ErrorType.PYTHON_IMPORT for e in errors)
        
    def test_detect_java_compile_error(self):
        """Test detection of Java compile errors."""
        output = '''
Main.java:10: error: cannot find symbol
        System.out.println(x);
                           ^
  symbol:   variable x
  location: class Main
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        assert any(e.error_type == ErrorType.JAVA_COMPILE for e in errors)
        
    def test_detect_java_runtime_error(self):
        """Test detection of Java runtime exceptions."""
        output = '''
Exception in thread "main" java.lang.NullPointerException
    at Main.main(Main.java:15)
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        assert any(e.error_type == ErrorType.JAVA_RUNTIME for e in errors)
        
    def test_detect_maven_error(self):
        """Test detection of Maven build errors."""
        output = '''
[ERROR] Failed to execute goal
[ERROR] Compilation failure
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        assert any(e.error_type == ErrorType.MAVEN_BUILD for e in errors)
        
    def test_detect_gradle_error(self):
        """Test detection of Gradle build errors."""
        output = '''
FAILURE: Build failed with an exception.
* What went wrong:
Execution failed for task ':compileJava'.
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        assert any(e.error_type == ErrorType.GRADLE_BUILD for e in errors)
        
    def test_detect_npm_error(self):
        """Test detection of NPM errors."""
        output = '''
npm ERR! code ENOENT
npm ERR! syscall open
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        assert any(e.error_type == ErrorType.NPM_ERROR for e in errors)
        
    def test_detect_generic_error(self):
        """Test detection of generic errors."""
        output = '''
Error: Something went wrong
Fatal error: connection failed
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        assert any(e.error_type == ErrorType.GENERIC_ERROR for e in errors)
        
    def test_extract_file_path_and_line(self):
        """Test extraction of file path and line number."""
        output = '''
  File "/path/to/file.py", line 42
    some_code()
'''
        errors = ErrorPattern.detect_errors(output)
        assert len(errors) > 0
        # Check that file path or line number was extracted
        assert any(e.file_path is not None or e.line_number is not None for e in errors)
        
    def test_no_false_positives(self):
        """Test that normal output doesn't trigger false positives."""
        output = '''
Success! All tests passed.
Build completed successfully.
No errors found.
'''
        errors = ErrorPattern.detect_errors(output)
        # Should not detect any errors in normal output
        assert len(errors) == 0
        
    def test_multiple_errors(self):
        """Test detection of multiple errors in output."""
        output = '''
[ERROR] First error
[ERROR] Second error
[ERROR] Third error
'''
        errors = ErrorPattern.detect_errors(output)
        # Should detect multiple errors
        assert len(errors) >= 1


class TestTerminalError:
    """Test TerminalError dataclass."""
    
    def test_error_creation(self):
        """Test creating a TerminalError."""
        error = TerminalError(
            error_type=ErrorType.PYTHON_SYNTAX,
            message="invalid syntax",
            line_number=10,
            file_path="test.py",
            raw_output="SyntaxError: invalid syntax"
        )
        assert error.error_type == ErrorType.PYTHON_SYNTAX
        assert error.message == "invalid syntax"
        assert error.line_number == 10
        assert error.file_path == "test.py"
        
    def test_error_without_optional_fields(self):
        """Test creating a TerminalError without optional fields."""
        error = TerminalError(
            error_type=ErrorType.GENERIC_ERROR,
            message="Something went wrong"
        )
        assert error.error_type == ErrorType.GENERIC_ERROR
        assert error.message == "Something went wrong"
        assert error.line_number is None
        assert error.file_path is None
        
    def test_error_type_values(self):
        """Test that all error types have correct values."""
        assert ErrorType.PYTHON_SYNTAX.value == "python_syntax"
        assert ErrorType.PYTHON_RUNTIME.value == "python_runtime"
        assert ErrorType.PYTHON_IMPORT.value == "python_import"
        assert ErrorType.JAVA_COMPILE.value == "java_compile"
        assert ErrorType.JAVA_RUNTIME.value == "java_runtime"
        assert ErrorType.MAVEN_BUILD.value == "maven_build"
        assert ErrorType.GRADLE_BUILD.value == "gradle_build"
        assert ErrorType.NPM_ERROR.value == "npm_error"
        assert ErrorType.GENERIC_ERROR.value == "generic_error"
