"""Unit tests for edit_validator module."""

import pytest
from pyutagent.tools.edit_validator import (
    SyntaxValidator, TestCodeValidator, EditImpactAnalyzer, EditValidator,
    ValidationError, ValidationResult, ValidationErrorType,
    validate_syntax_only, validate_test_code
)


class TestSyntaxValidator:
    """Tests for SyntaxValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a SyntaxValidator instance."""
        return SyntaxValidator()
    
    def test_valid_java_code(self, validator):
        """Test validating valid Java code."""
        code = """public class Test {
    public void method() {
        int x = 1;
    }
}"""
        
        result = validator.validate(code)
        
        assert result.is_valid
        assert result.error_count == 0
    
    def test_unbalanced_braces(self, validator):
        """Test detecting unbalanced braces."""
        code = """public class Test {
    public void method() {
        int x = 1;
    // Missing closing brace
}"""
        
        result = validator.validate(code)
        
        assert not result.is_valid
        assert any(e.error_type == ValidationErrorType.BALANCE_ERROR for e in result.errors)
    
    def test_unbalanced_parentheses(self, validator):
        """Test detecting unbalanced parentheses."""
        code = """public class Test {
    public void method(
        int x
    // Missing closing parenthesis
    {
    }
}"""
        
        result = validator.validate(code)
        
        assert not result.is_valid
        assert any(e.error_type == ValidationErrorType.BALANCE_ERROR for e in result.errors)
    
    def test_valid_test_class(self, validator):
        """Test validating a valid test class."""
        code = """import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MyTest {
    @Test
    void testMethod() {
        assertTrue(true);
    }
}"""
        
        result = validator.validate(code)
        
        assert result.is_valid


class TestTestCodeValidator:
    """Tests for TestCodeValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a TestCodeValidator instance."""
        return TestCodeValidator()
    
    def test_valid_test_structure(self, validator):
        """Test validating valid test structure."""
        code = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    void testMethod() {
        assertTrue(true);
    }
}"""
        
        result = validator.validate(code)
        
        assert result.is_valid
    
    def test_missing_test_annotation(self, validator):
        """Test detecting missing @Test annotation."""
        code = """public class MyTest {
    void testMethod() {
        assertTrue(true);
    }
}"""
        
        result = validator.validate(code)
        
        # Should have warnings but still be valid
        assert any("@Test" in w.message for w in result.warnings)
    
    def test_class_naming_convention(self, validator):
        """Test class naming convention warning."""
        code = """import org.junit.jupiter.api.Test;

public class MyClass {
    @Test
    void testMethod() {
    }
}"""
        
        result = validator.validate(code)
        
        assert any("Test" in w.message and "should end" in w.message for w in result.warnings)
    
    def test_public_test_method_warning(self, validator):
        """Test warning for public test methods."""
        code = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    public void testMethod() {
    }
}"""
        
        result = validator.validate(code)
        
        assert any("public" in w.message and "package-private" in w.message for w in result.warnings)
    
    def test_validate_imports(self, validator):
        """Test validating imports."""
        code = """import org.junit.jupiter.api.Test;

public class MyTest {
}"""
        
        missing_required, missing_recommended = validator.validate_imports(code)
        
        # Test import is present, BeforeEach might be missing
        assert 'org.junit.jupiter.api.Test' not in missing_required
    
    def test_validate_imports_with_wildcard(self, validator):
        """Test validating imports with wildcard."""
        code = """import org.junit.jupiter.api.*;

public class MyTest {
}"""
        
        missing_required, _ = validator.validate_imports(code)
        
        # Wildcard import should cover required imports
        assert 'org.junit.jupiter.api.Test' not in missing_required


class TestEditImpactAnalyzer:
    """Tests for EditImpactAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an EditImpactAnalyzer instance."""
        return EditImpactAnalyzer()
    
    def test_analyze_simple_change(self, analyzer):
        """Test analyzing a simple change."""
        original = """public class Test {
    public void method() {
        int x = 1;
    }
}"""
        
        modified = """public class Test {
    public void method() {
        int x = 2;
    }
}"""
        
        impact = analyzer.analyze_impact(original, modified)
        
        assert impact['lines_changed'] > 0
        assert impact['methods_added'] == 0
        assert impact['methods_removed'] == 0
        assert impact['risk_level'] in ['low', 'medium', 'high']
    
    def test_analyze_method_addition(self, analyzer):
        """Test analyzing method addition."""
        original = """public class Test {
    public void method1() {
    }
}"""
        
        modified = """public class Test {
    public void method1() {
    }
    
    public void method2() {
    }
}"""
        
        impact = analyzer.analyze_impact(original, modified)
        
        assert impact['methods_added'] == 1
        assert impact['methods_removed'] == 0
    
    def test_analyze_method_removal(self, analyzer):
        """Test analyzing method removal."""
        original = """public class Test {
    public void method1() {
    }
    
    public void method2() {
    }
}"""
        
        modified = """public class Test {
    public void method1() {
    }
}"""
        
        impact = analyzer.analyze_impact(original, modified)
        
        assert impact['methods_removed'] == 1
        assert impact['risk_level'] in ['medium', 'high']
    
    def test_analyze_import_changes(self, analyzer):
        """Test analyzing import changes."""
        original = """import org.junit.jupiter.api.Test;

public class Test {
}"""
        
        modified = """import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

public class Test {
}"""
        
        impact = analyzer.analyze_impact(original, modified)
        
        assert 'org.mockito.Mockito' in impact['imports_added']


class TestEditValidator:
    """Tests for EditValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create an EditValidator instance."""
        return EditValidator()
    
    def test_validate_valid_edit(self, validator):
        """Test validating a valid edit."""
        original = """public class Test {
    public void method() {
        int x = 1;
    }
}"""
        
        modified = """public class Test {
    public void method() {
        int x = 2;
    }
}"""
        
        result = validator.validate_edit(original, modified, is_test_code=False)
        
        assert result.is_valid
    
    def test_validate_invalid_syntax(self, validator):
        """Test validating edit with invalid syntax."""
        original = """public class Test {
    public void method() {
    }
}"""
        
        modified = """public class Test {
    public void method() {
        int x = // Invalid syntax
    }
}"""
        
        result = validator.validate_edit(original, modified, is_test_code=False)
        
        assert not result.is_valid
    
    def test_quick_validate(self, validator):
        """Test quick validation."""
        valid_code = "public class Test { }"
        invalid_code = "public class Test {"
        
        assert validator.quick_validate(valid_code)
        assert not validator.quick_validate(invalid_code)


class TestValidationError:
    """Tests for ValidationError dataclass."""
    
    def test_error_creation(self):
        """Test creating a validation error."""
        error = ValidationError(
            error_type=ValidationErrorType.SYNTAX_ERROR,
            message="Syntax error",
            line_number=10,
            column=5,
            severity="error",
            suggestion="Fix the syntax"
        )
        
        assert error.error_type == ValidationErrorType.SYNTAX_ERROR
        assert error.message == "Syntax error"
        assert error.line_number == 10
        assert error.column == 5
        assert error.severity == "error"
        assert error.suggestion == "Fix the syntax"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_valid_result(self):
        """Test a valid result."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid
        assert not result.has_errors
        assert result.error_count == 0
        assert result.warning_count == 0
    
    def test_invalid_result_with_errors(self):
        """Test an invalid result with errors."""
        result = ValidationResult(
            is_valid=False,
            errors=[
                ValidationError(
                    error_type=ValidationErrorType.SYNTAX_ERROR,
                    message="Error 1",
                    severity="error"
                ),
                ValidationError(
                    error_type=ValidationErrorType.SYNTAX_ERROR,
                    message="Error 2",
                    severity="error"
                )
            ],
            warnings=[
                ValidationError(
                    error_type=ValidationErrorType.TEST_STRUCTURE_ERROR,
                    message="Warning",
                    severity="warning"
                )
            ]
        )
        
        assert not result.is_valid
        assert result.has_errors
        assert result.error_count == 2
        assert result.warning_count == 1


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_validate_syntax_only_valid(self):
        """Test validate_syntax_only with valid code."""
        code = """public class Test {
    public void method() {
    }
}"""
        
        assert validate_syntax_only(code)
    
    def test_validate_syntax_only_invalid(self):
        """Test validate_syntax_only with invalid code."""
        code = "public class Test {"
        
        assert not validate_syntax_only(code)
    
    def test_validate_test_code_valid(self):
        """Test validate_test_code with valid test code."""
        code = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    void testMethod() {
    }
}"""
        
        result = validate_test_code(code)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid or not result.has_errors
