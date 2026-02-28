"""Tests for failure analyzer module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import xml.etree.ElementTree as ET

from pyutagent.tools.failure_analyzer import (
    FailureType,
    TestFailure,
    FailureAnalysis,
    TestFailureAnalyzer,
    FailureFixGenerator,
)


class TestFailureType:
    """Tests for FailureType enum."""

    def test_failure_type_values(self):
        """Test that all failure types are defined."""
        assert FailureType.ASSERTION_FAILURE is not None
        assert FailureType.NULL_POINTER is not None
        assert FailureType.INDEX_OUT_OF_BOUNDS is not None
        assert FailureType.ILLEGAL_ARGUMENT is not None
        assert FailureType.ILLEGAL_STATE is not None
        assert FailureType.CLASS_CAST is not None
        assert FailureType.ARITHMETIC is not None
        assert FailureType.TIMEOUT is not None
        assert FailureType.UNEXPECTED_EXCEPTION is not None
        assert FailureType.MOCK_VERIFICATION is not None
        assert FailureType.MOCK_CONFIGURATION is not None
        assert FailureType.SETUP_FAILURE is not None
        assert FailureType.TEARDOWN_FAILURE is not None
        assert FailureType.COMPILATION_ERROR is not None
        assert FailureType.UNKNOWN is not None


class TestTestFailure:
    """Tests for TestFailure dataclass."""

    def test_test_failure_creation(self):
        """Test creating a TestFailure."""
        failure = TestFailure(
            failure_type=FailureType.ASSERTION_FAILURE,
            test_class="TestClass",
            test_method="testMethod",
            message="Expected 1 but was 2",
            stack_trace="at TestClass.testMethod(Test.java:10)",
            line_number=10,
            cause="Assertion failed",
            suggestions=["Check expected value"],
            fix_hint="Update assertion"
        )
        assert failure.failure_type == FailureType.ASSERTION_FAILURE
        assert failure.test_class == "TestClass"
        assert failure.test_method == "testMethod"
        assert failure.line_number == 10

    def test_test_failure_defaults(self):
        """Test TestFailure with default values."""
        failure = TestFailure(
            failure_type=FailureType.NULL_POINTER,
            test_class="TestClass",
            test_method="testMethod",
            message="NPE",
            stack_trace=""
        )
        assert failure.line_number is None
        assert failure.cause is None
        assert failure.suggestions == []
        assert failure.fix_hint == ""


class TestFailureAnalysis:
    """Tests for FailureAnalysis dataclass."""

    def test_failure_analysis_creation(self):
        """Test creating a FailureAnalysis."""
        failure = TestFailure(
            failure_type=FailureType.ASSERTION_FAILURE,
            test_class="TestClass",
            test_method="testMethod",
            message="Failed",
            stack_trace=""
        )
        analysis = FailureAnalysis(
            failures=[failure],
            summary="1 failure",
            fix_strategy="Fix assertions",
            priority=1,
            total_tests=5,
            passed_tests=4,
            failed_tests=1,
            error_tests=0,
            skipped_tests=0
        )
        assert len(analysis.failures) == 1
        assert analysis.summary == "1 failure"
        assert analysis.priority == 1
        assert analysis.total_tests == 5


class TestTestFailureAnalyzer:
    """Tests for TestFailureAnalyzer."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        """Create a TestFailureAnalyzer instance."""
        return TestFailureAnalyzer(str(tmp_path))

    def test_initialization(self, analyzer, tmp_path):
        """Test analyzer initialization."""
        assert analyzer.project_path == Path(tmp_path)
        assert analyzer.surefire_dir == tmp_path / "target" / "surefire-reports"
        assert len(analyzer.exception_patterns) > 0

    def test_compile_patterns(self, analyzer):
        """Test that exception patterns are compiled."""
        for failure_type, patterns in analyzer.exception_patterns.items():
            assert isinstance(patterns, list)
            for pattern in patterns:
                # Check that patterns are compiled regex
                assert hasattr(pattern, 'search')

    def test_classify_failure_assertion(self, analyzer):
        """Test classifying assertion failures."""
        failure_type = analyzer._classify_failure(
            "org.opentest4j.AssertionFailedError",
            "expected:<1> but was:<2>",
            ""
        )
        assert failure_type == FailureType.ASSERTION_FAILURE

    def test_classify_failure_null_pointer(self, analyzer):
        """Test classifying null pointer exceptions."""
        failure_type = analyzer._classify_failure(
            "java.lang.NullPointerException",
            "",
            ""
        )
        assert failure_type == FailureType.NULL_POINTER

    def test_classify_failure_mock_verification(self, analyzer):
        """Test classifying mock verification failures."""
        failure_type = analyzer._classify_failure(
            "",
            "Wanted but not invoked",
            ""
        )
        assert failure_type == FailureType.MOCK_VERIFICATION

    def test_classify_failure_setup(self, analyzer):
        """Test classifying setup failures."""
        failure_type = analyzer._classify_failure(
            "",
            "",
            "at TestClass.setUp(Test.java:10) @BeforeEach"
        )
        assert failure_type == FailureType.SETUP_FAILURE

    def test_classify_failure_teardown(self, analyzer):
        """Test classifying teardown failures."""
        failure_type = analyzer._classify_failure(
            "",
            "",
            "at TestClass.tearDown(Test.java:10) @AfterEach"
        )
        assert failure_type == FailureType.TEARDOWN_FAILURE

    def test_classify_failure_unknown(self, analyzer):
        """Test classifying unknown failures."""
        failure_type = analyzer._classify_failure(
            "SomeRandomException",
            "Unknown error",
            ""
        )
        assert failure_type == FailureType.UNEXPECTED_EXCEPTION

    def test_extract_line_number(self, analyzer):
        """Test extracting line number from stack trace."""
        stack_trace = "at TestClass.testMethod(Test.java:42)"
        line = analyzer._extract_line_number(stack_trace)
        assert line == 42

    def test_extract_line_number_not_found(self, analyzer):
        """Test extracting line number when not found."""
        stack_trace = "at TestClass.testMethod"
        line = analyzer._extract_line_number(stack_trace)
        assert line is None

    def test_extract_cause_from_stack_trace(self, analyzer):
        """Test extracting cause from stack trace."""
        stack_trace = "Some error\nCaused by: java.lang.NullPointerException: detail"
        cause = analyzer._extract_cause("", stack_trace)
        assert "NullPointerException" in cause

    def test_extract_cause_from_message(self, analyzer):
        """Test extracting cause from message when no stack trace cause."""
        message = "Error occurred\nMore details"
        cause = analyzer._extract_cause(message, "")
        assert cause == "Error occurred"

    def test_generate_suggestions_assertion(self, analyzer):
        """Test generating suggestions for assertion failures."""
        suggestions = analyzer._generate_suggestions(
            FailureType.ASSERTION_FAILURE,
            "expected:<1> but was:<2>",
            None
        )
        assert len(suggestions) > 0
        assert any("expected" in s.lower() for s in suggestions)

    def test_generate_suggestions_null_pointer(self, analyzer):
        """Test generating suggestions for null pointer."""
        suggestions = analyzer._generate_suggestions(
            FailureType.NULL_POINTER,
            "",
            None
        )
        assert len(suggestions) > 0
        assert any("null" in s.lower() for s in suggestions)

    def test_generate_suggestions_mock_verification(self, analyzer):
        """Test generating suggestions for mock verification."""
        suggestions = analyzer._generate_suggestions(
            FailureType.MOCK_VERIFICATION,
            "",
            None
        )
        assert len(suggestions) > 0
        assert any("mock" in s.lower() for s in suggestions)

    def test_generate_fix_hint_assertion(self, analyzer):
        """Test generating fix hint for assertion failure."""
        hint = analyzer._generate_fix_hint(
            FailureType.ASSERTION_FAILURE,
            "expected:<value1> but was:<value2>",
            None
        )
        assert "value1" in hint
        assert "value2" in hint

    def test_generate_fix_hint_null_pointer(self, analyzer):
        """Test generating fix hint for null pointer."""
        hint = analyzer._generate_fix_hint(
            FailureType.NULL_POINTER,
            "",
            None
        )
        assert "null" in hint.lower()

    def test_generate_summary_no_failures(self, analyzer):
        """Test generating summary with no failures."""
        summary = analyzer._generate_summary([], 10, 10, 0, 0)
        assert "All tests passed" in summary
        assert "10/10" in summary

    def test_generate_summary_with_failures(self, analyzer):
        """Test generating summary with failures."""
        failures = [
            TestFailure(FailureType.ASSERTION_FAILURE, "C", "m", "", ""),
            TestFailure(FailureType.ASSERTION_FAILURE, "C", "m2", "", ""),
            TestFailure(FailureType.NULL_POINTER, "C", "m3", "", ""),
        ]
        summary = analyzer._generate_summary(failures, 5, 2, 3, 0)
        assert "2 passed, 3 failed" in summary
        assert "Assertion Failure" in summary
        assert "Null Pointer" in summary

    def test_determine_fix_strategy_no_failures(self, analyzer):
        """Test determining fix strategy with no failures."""
        strategy = analyzer._determine_fix_strategy([])
        assert strategy == "No fixes needed"

    def test_determine_fix_strategy_setup_failure(self, analyzer):
        """Test determining fix strategy with setup failures."""
        failures = [TestFailure(FailureType.SETUP_FAILURE, "C", "m", "", "")]
        strategy = analyzer._determine_fix_strategy(failures)
        assert "setup" in strategy.lower()

    def test_determine_fix_strategy_mock_issues(self, analyzer):
        """Test determining fix strategy with mock issues."""
        failures = [TestFailure(FailureType.MOCK_CONFIGURATION, "C", "m", "", "")]
        strategy = analyzer._determine_fix_strategy(failures)
        assert "mock" in strategy.lower()

    def test_determine_fix_strategy_null_pointer(self, analyzer):
        """Test determining fix strategy with null pointer."""
        failures = [TestFailure(FailureType.NULL_POINTER, "C", "m", "", "")]
        strategy = analyzer._determine_fix_strategy(failures)
        assert "null" in strategy.lower()

    def test_calculate_priority_no_failures(self, analyzer):
        """Test calculating priority with no failures."""
        priority = analyzer._calculate_priority([])
        assert priority == 5

    def test_calculate_priority_setup_failure(self, analyzer):
        """Test calculating priority with setup failure."""
        failures = [TestFailure(FailureType.SETUP_FAILURE, "C", "m", "", "")]
        priority = analyzer._calculate_priority(failures)
        assert priority == 1

    def test_calculate_priority_null_pointer(self, analyzer):
        """Test calculating priority with null pointer."""
        failures = [TestFailure(FailureType.NULL_POINTER, "C", "m", "", "")]
        priority = analyzer._calculate_priority(failures)
        assert priority == 2

    def test_get_fix_prompt_context(self, analyzer):
        """Test generating fix prompt context."""
        failure = TestFailure(
            failure_type=FailureType.ASSERTION_FAILURE,
            test_class="TestClass",
            test_method="testMethod",
            message="Expected 1 but was 2",
            stack_trace="",
            line_number=10,
            cause="Assertion failed",
            fix_hint="Update expected value"
        )
        analysis = FailureAnalysis(
            failures=[failure],
            summary="1 failure",
            fix_strategy="Fix assertions",
            priority=1
        )
        context = analyzer.get_fix_prompt_context(analysis)
        assert "1 failure" in context
        assert "Fix Strategy" in context
        assert "TestClass.testMethod" in context
        assert "Line: 10" in context

    def test_get_fix_prompt_context_many_failures(self, analyzer):
        """Test that only first 10 failures are included."""
        failures = [
            TestFailure(FailureType.ASSERTION_FAILURE, "C", f"m{i}", "", "")
            for i in range(15)
        ]
        analysis = FailureAnalysis(
            failures=failures,
            summary="15 failures",
            fix_strategy="Fix all",
            priority=1
        )
        context = analyzer.get_fix_prompt_context(analysis)
        assert "and 5 more failures" in context


class TestFailureFixGenerator:
    """Tests for FailureFixGenerator."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock failure analyzer."""
        return Mock(spec=TestFailureAnalyzer)

    @pytest.fixture
    def generator(self, mock_analyzer):
        """Create a FailureFixGenerator instance."""
        return FailureFixGenerator(mock_analyzer)

    def test_generate_fixes(self, generator, mock_analyzer):
        """Test generating fixes."""
        failure = TestFailure(
            failure_type=FailureType.ASSERTION_FAILURE,
            test_class="TestClass",
            test_method="testMethod",
            message="Failed",
            stack_trace=""
        )
        analysis = FailureAnalysis(
            failures=[failure],
            summary="1 failure",
            fix_strategy="Fix it",
            priority=1
        )
        mock_analyzer.analyze.return_value = analysis
        mock_analyzer.get_fix_prompt_context.return_value = "context"

        result = generator.generate_fixes("test code", {})

        assert "analysis" in result
        assert "fixes" in result
        assert "context" in result
        assert len(result["fixes"]) == 1

    def test_generate_specific_fix_assertion(self, generator):
        """Test generating specific fix for assertion failure."""
        failure = TestFailure(
            failure_type=FailureType.ASSERTION_FAILURE,
            test_class="TestClass",
            test_method="testMethod",
            message="Failed",
            stack_trace=""
        )
        fix = generator._generate_specific_fix(failure, "code", {})
        assert fix["action"] == "fix_assertion"
        assert fix["test_method"] == "testMethod"

    def test_generate_specific_fix_null_pointer(self, generator):
        """Test generating specific fix for null pointer."""
        failure = TestFailure(
            failure_type=FailureType.NULL_POINTER,
            test_class="TestClass",
            test_method="testMethod",
            message="NPE",
            stack_trace=""
        )
        fix = generator._generate_specific_fix(failure, "code", {})
        assert fix["action"] == "add_initialization"

    def test_generate_specific_fix_mock_verification(self, generator):
        """Test generating specific fix for mock verification."""
        failure = TestFailure(
            failure_type=FailureType.MOCK_VERIFICATION,
            test_class="TestClass",
            test_method="testMethod",
            message="Verify failed",
            stack_trace=""
        )
        fix = generator._generate_specific_fix(failure, "code", {})
        assert fix["action"] == "fix_mock_verification"

    def test_generate_specific_fix_mock_config(self, generator):
        """Test generating specific fix for mock configuration."""
        failure = TestFailure(
            failure_type=FailureType.MOCK_CONFIGURATION,
            test_class="TestClass",
            test_method="testMethod",
            message="Unnecessary stubbing",
            stack_trace=""
        )
        fix = generator._generate_specific_fix(failure, "code", {})
        assert fix["action"] == "fix_mock_setup"

    def test_generate_specific_fix_setup(self, generator):
        """Test generating specific fix for setup failure."""
        failure = TestFailure(
            failure_type=FailureType.SETUP_FAILURE,
            test_class="TestClass",
            test_method="setUp",
            message="Setup failed",
            stack_trace=""
        )
        fix = generator._generate_specific_fix(failure, "code", {})
        assert fix["action"] == "fix_setup"

    def test_generate_specific_fix_timeout(self, generator):
        """Test generating specific fix for timeout."""
        failure = TestFailure(
            failure_type=FailureType.TIMEOUT,
            test_class="TestClass",
            test_method="testMethod",
            message="Timed out",
            stack_trace=""
        )
        fix = generator._generate_specific_fix(failure, "code", {})
        assert fix["action"] == "increase_timeout"
