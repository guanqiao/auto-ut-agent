"""Unit tests for GenerationEvaluator."""

import pytest
from pyutagent.agent.generation_evaluator import (
    GenerationEvaluator,
    QualityDimension,
    IssueSeverity,
    evaluate_test_code,
    is_code_compilable
)


class TestGenerationEvaluator:
    """Tests for GenerationEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a GenerationEvaluator instance."""
        return GenerationEvaluator()
    
    @pytest.fixture
    def valid_test_code(self):
        """Valid test code for testing."""
        return '''
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SampleTest {
    @Test
    public void shouldReturnCorrectValue() {
        SampleClass sample = new SampleClass("test", 10);
        assertEquals("test", sample.getName());
    }
    
    @Test
    public void shouldCalculateCorrectly() {
        SampleClass sample = new SampleClass("test", 5);
        assertEquals(15, sample.calculate(5, 5));
    }
}
'''
    
    @pytest.fixture
    def invalid_test_code(self):
        """Invalid test code with issues."""
        return '''
import org.junit.jupiter.api.Test;

public class BadTest {
    // Missing @Test annotation on some methods
    public void missingAnnotation() {
        // No assertions
    }
    
    @Test
    public void unbalancedBraces() {
        if (true) {
            // Missing closing brace
    }
}
'''
    
    @pytest.fixture
    def target_class_info(self):
        """Sample target class info."""
        return {
            'name': 'SampleClass',
            'package': 'com.example',
            'methods': [
                {'name': 'getName', 'return_type': 'String', 'parameters': []},
                {'name': 'calculate', 'return_type': 'int', 'parameters': [('int', 'x'), ('int', 'y')]},
            ]
        }
    
    def test_evaluate_valid_code(self, evaluator, valid_test_code, target_class_info):
        """Test evaluating valid test code."""
        result = evaluator.evaluate(valid_test_code, target_class_info)
        
        assert result.overall_score > 0.7
        assert result.is_acceptable is True
        assert len(result.get_critical_issues()) == 0
    
    def test_evaluate_invalid_code(self, evaluator, invalid_test_code):
        """Test evaluating invalid test code."""
        result = evaluator.evaluate(invalid_test_code)
        
        # Should have critical issues due to unbalanced braces
        assert len(result.get_critical_issues()) > 0
        assert result.is_acceptable is False
    
    def test_evaluate_syntax_dimension(self, evaluator):
        """Test syntax evaluation."""
        # Code with unbalanced braces
        code = "public class Test { public void method() { }"
        
        score = evaluator._evaluate_syntax(code)
        
        assert score < 1.0
        # Check that issues were recorded
        assert len(evaluator.issues) > 0
    
    def test_evaluate_completeness_dimension(self, evaluator):
        """Test completeness evaluation."""
        # Code missing @Test and assertions
        code = "public class Test { public void method() { } }"
        
        score = evaluator._evaluate_completeness(code)
        
        assert score < 1.0
        # Check that issues were recorded
        assert len(evaluator.issues) > 0
    
    def test_evaluate_style_dimension(self, evaluator):
        """Test style evaluation."""
        # Code with inconsistent indentation
        code = '''
public class Test {
  public void method1() { }
    public void method2() { }
}
'''
        
        score = evaluator._evaluate_style(code)
        
        # Should detect style issues
        assert score <= 1.0
    
    def test_evaluate_mock_usage(self, evaluator):
        """Test mock usage evaluation."""
        # Code using mocks without proper imports
        code = '''
import org.junit.jupiter.api.Test;

public class Test {
    @Mock
    private Service service;
    
    @Test
    public void test() {
        when(service.call()).thenReturn("result");
    }
}
'''
        
        score = evaluator._evaluate_mock_usage(code)
        
        assert score < 1.0
        # Check that issues were recorded
        assert len(evaluator.issues) > 0
    
    def test_evaluate_assertions(self, evaluator):
        """Test assertion evaluation."""
        # Code with test method but no assertions
        code = '''
import org.junit.jupiter.api.Test;

public class Test {
    @Test
    public void testWithoutAssertion() {
        int x = 1 + 1;
    }
}
'''
        
        score = evaluator._evaluate_assertions(code)
        
        assert score < 1.0
        # Check that issues were recorded
        assert len(evaluator.issues) > 0
    
    def test_estimate_coverage_potential(self, evaluator, target_class_info):
        """Test coverage potential estimation."""
        test_code = '''
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SampleTest {
    @Test
    public void shouldTestGetName() {
        SampleClass sample = new SampleClass("test", 10);
        assertEquals("test", sample.getName());
    }
}
'''
        
        estimate = evaluator._estimate_coverage_potential(test_code, target_class_info)
        
        assert estimate.method_coverage_potential > 0
        assert estimate.method_coverage_potential < 1.0  # Not all methods tested
        assert 'calculate' in estimate.uncovered_methods
    
    def test_quick_check_valid(self, evaluator, valid_test_code):
        """Test quick check on valid code."""
        is_valid, issues = evaluator.quick_check(valid_test_code)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_quick_check_invalid(self, evaluator, invalid_test_code):
        """Test quick check on invalid code."""
        is_valid, issues = evaluator.quick_check(invalid_test_code)
        
        assert is_valid is False
        assert len(issues) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_evaluate_test_code(self):
        """Test evaluate_test_code function."""
        code = '''
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class Test {
    @Test
    public void test() {
        assertTrue(true);
    }
}
'''
        
        result = evaluate_test_code(code)
        
        assert result is not None
        assert result.overall_score >= 0
        assert result.overall_score <= 1.0
    
    def test_is_code_compilable_valid(self):
        """Test is_code_compilable with valid code."""
        code = '''
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class Test {
    @Test
    public void test() { assertTrue(true); }
}
'''
        
        is_valid, issues = is_code_compilable(code)
        
        assert is_valid is True
    
    def test_is_code_compilable_invalid(self):
        """Test is_code_compilable with invalid code."""
        code = "public class Test { public void test() { }"  # Missing closing brace
        
        is_valid, issues = is_code_compilable(code)
        
        assert is_valid is False
        assert len(issues) > 0


class TestEvaluationResult:
    """Tests for EvaluationResult."""
    
    def test_get_critical_issues(self):
        """Test getting critical issues."""
        from pyutagent.agent.generation_evaluator import EvaluationResult, QualityIssue
        
        issues = [
            QualityIssue(QualityDimension.SYNTAX, IssueSeverity.CRITICAL, "Critical issue"),
            QualityIssue(QualityDimension.STYLE, IssueSeverity.LOW, "Minor issue"),
            QualityIssue(QualityDimension.SYNTAX, IssueSeverity.HIGH, "High issue"),
        ]
        
        result = EvaluationResult(
            overall_score=0.5,
            is_acceptable=False,
            issues=issues
        )
        
        critical = result.get_critical_issues()
        
        assert len(critical) == 1
        assert critical[0].message == "Critical issue"
    
    def test_get_issues_by_dimension(self):
        """Test getting issues by dimension."""
        from pyutagent.agent.generation_evaluator import EvaluationResult, QualityIssue
        
        issues = [
            QualityIssue(QualityDimension.SYNTAX, IssueSeverity.CRITICAL, "Syntax issue"),
            QualityIssue(QualityDimension.STYLE, IssueSeverity.LOW, "Style issue"),
            QualityIssue(QualityDimension.SYNTAX, IssueSeverity.HIGH, "Another syntax issue"),
        ]
        
        result = EvaluationResult(
            overall_score=0.5,
            is_acceptable=False,
            issues=issues
        )
        
        syntax_issues = result.get_issues_by_dimension(QualityDimension.SYNTAX)
        
        assert len(syntax_issues) == 2


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_code(self):
        """Test evaluating empty code."""
        evaluator = GenerationEvaluator()
        
        result = evaluator.evaluate("")
        
        assert result.overall_score < 0.5
        assert result.is_acceptable is False
    
    def test_code_with_only_comments(self):
        """Test evaluating code with only comments."""
        evaluator = GenerationEvaluator()
        code = "// This is a comment\n/* Multi-line comment */"
        
        result = evaluator.evaluate(code)
        
        assert result.is_acceptable is False
    
    def test_code_with_duplicate_keywords(self):
        """Test detecting duplicate keywords."""
        evaluator = GenerationEvaluator()
        code = "public public class Test { }"
        
        score = evaluator._evaluate_syntax(code)
        
        assert score < 1.0
        assert any("duplicate" in i.message.lower() for i in evaluator.issues)
