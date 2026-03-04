"""Unit tests for SelfReflection module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pyutagent.agent.self_reflection import (
    SelfReflection,
    QualityDimension,
    IssueSeverity,
    QualityMetric,
    IdentifiedIssue,
    CoverageEstimate,
    CritiqueResult,
    ImprovementResult,
)


class TestSelfReflection:
    """Tests for SelfReflection class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        reflection = SelfReflection()
        
        assert reflection.llm_client is None
        assert reflection.quality_threshold == 0.7
        assert reflection.max_improvement_iterations == 3
        assert reflection._critique_history == []

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        mock_client = MagicMock()
        reflection = SelfReflection(
            llm_client=mock_client,
            quality_threshold=0.8,
            max_improvement_iterations=5
        )
        
        assert reflection.llm_client == mock_client
        assert reflection.quality_threshold == 0.8
        assert reflection.max_improvement_iterations == 5

    @pytest.mark.asyncio
    async def test_critique_generated_test_success(self):
        """Test successful critique of generated test code."""
        reflection = SelfReflection(quality_threshold=0.7)
        
        test_code = '''
@Test
@DisplayName("should calculate sum correctly")
void shouldCalculateSumCorrectly() {
    Calculator calc = new Calculator();
    int result = calc.add(2, 3);
    assertEquals(5, result);
}
'''
        source_code = '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
'''
        
        result = await reflection.critique_generated_test(test_code, source_code)
        
        assert isinstance(result, CritiqueResult)
        assert result.overall_quality_score >= 0.0
        assert result.overall_quality_score <= 1.0
        assert len(result.quality_metrics) > 0
        assert isinstance(result.coverage_estimate, CoverageEstimate)

    @pytest.mark.asyncio
    async def test_critique_generated_test_with_issues(self):
        """Test critique of test code with issues."""
        reflection = SelfReflection(quality_threshold=0.7)
        
        test_code = '''
@Test
void test1() {
    // No assertions
}
'''
        source_code = '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
'''
        
        result = await reflection.critique_generated_test(test_code, source_code)
        
        assert result.overall_quality_score < 0.7
        assert len(result.identified_issues) > 0
        assert any(i.issue_type == "MISSING_ASSERTIONS" for i in result.identified_issues)

    @pytest.mark.asyncio
    async def test_evaluate_quality_dimensions(self):
        """Test quality dimension evaluation."""
        reflection = SelfReflection()
        
        test_code = '''
@Test
@DisplayName("should handle valid input")
void shouldHandleValidInput() {
    Service service = new Service();
    String result = service.process("test");
    assertNotNull(result);
    assertEquals("processed", result);
}
'''
        source_code = '''
public class Service {
    public String process(String input) {
        return "processed";
    }
}
'''
        
        metrics = await reflection._evaluate_quality(test_code, source_code)
        
        assert len(metrics) == 6
        dimensions = [m.dimension for m in metrics]
        assert QualityDimension.READABILITY in dimensions
        assert QualityDimension.MAINTAINABILITY in dimensions
        assert QualityDimension.TESTABILITY in dimensions
        assert QualityDimension.COMPLETENESS in dimensions
        assert QualityDimension.CORRECTNESS in dimensions
        assert QualityDimension.BEST_PRACTICES in dimensions

    @pytest.mark.asyncio
    async def test_evaluate_readability(self):
        """Test readability evaluation."""
        reflection = SelfReflection()
        
        good_code = '''
@Test
@DisplayName("should return sum of two numbers")
void shouldReturnSum() {
    Calculator calc = new Calculator();
    int result = calc.add(2, 3);
    assertEquals(5, result);
}
'''
        metric = await reflection._evaluate_readability(good_code)
        
        assert metric.dimension == QualityDimension.READABILITY
        assert metric.score >= 0.5

    @pytest.mark.asyncio
    async def test_evaluate_testability(self):
        """Test testability evaluation."""
        reflection = SelfReflection()
        
        code_with_assertions = '''
@Test
void testWithAssertions() {
    assertEquals(1, 1);
    assertTrue(true);
    assertNotNull(new Object());
}
'''
        metric = await reflection._evaluate_testability(code_with_assertions)
        
        assert metric.dimension == QualityDimension.TESTABILITY
        assert metric.score >= 0.7

    @pytest.mark.asyncio
    async def test_estimate_coverage(self):
        """Test coverage estimation."""
        reflection = SelfReflection()
        
        test_code = '''
@Test
void testAdd() {
    Calculator calc = new Calculator();
    assertEquals(5, calc.add(2, 3));
}
@Test
void testSubtract() {
    Calculator calc = new Calculator();
    assertEquals(1, calc.subtract(3, 2));
}
'''
        source_code = '''
public class Calculator {
    public int add(int a, int b) { return a + b; }
    public int subtract(int a, int b) { return a - b; }
    public int multiply(int a, int b) { return a * b; }
}
'''
        class_info = {
            "methods": [
                {"name": "add"},
                {"name": "subtract"},
                {"name": "multiply"}
            ]
        }
        
        estimate = await reflection._estimate_coverage(test_code, source_code, class_info)
        
        assert isinstance(estimate, CoverageEstimate)
        assert estimate.estimated_line_coverage >= 0.0
        assert "add" in estimate.estimated_method_coverage
        assert "subtract" in estimate.estimated_method_coverage

    @pytest.mark.asyncio
    async def test_identify_issues_mock(self):
        """Test issue identification for mock-related issues."""
        reflection = SelfReflection()
        
        test_code = '''
@Mock
private UserRepository userRepo;

@Test
void testUser() {
    UserService service = new UserService();
    User user = service.getUser(1L);
}
'''
        source_code = '''
public class UserService {
    private UserRepository userRepo;
    public User getUser(Long id) { return userRepo.findById(id); }
}
'''
        
        issues = await reflection._identify_issues(test_code, source_code)
        
        mock_issues = [i for i in issues if i.issue_type == "UNSTUBBED_MOCK"]
        assert len(mock_issues) > 0

    @pytest.mark.asyncio
    async def test_identify_issues_missing_assertions(self):
        """Test issue identification for missing assertions."""
        reflection = SelfReflection()
        
        test_code = '''
@Test
void testWithoutAssertions() {
    Calculator calc = new Calculator();
    calc.add(2, 3);
}
'''
        source_code = '''
public class Calculator {
    public int add(int a, int b) { return a + b; }
}
'''
        
        issues = await reflection._identify_issues(test_code, source_code)
        
        assertion_issues = [i for i in issues if i.issue_type == "MISSING_ASSERTIONS"]
        assert len(assertion_issues) > 0

    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        reflection = SelfReflection()
        
        metrics = [
            QualityMetric(QualityDimension.READABILITY, 0.8),
            QualityMetric(QualityDimension.MAINTAINABILITY, 0.7),
            QualityMetric(QualityDimension.TESTABILITY, 0.9),
        ]
        issues = [
            IdentifiedIssue("issue1", IssueSeverity.LOW, "Low issue"),
            IdentifiedIssue("issue2", IssueSeverity.MEDIUM, "Medium issue"),
        ]
        
        score = reflection._calculate_overall_score(metrics, issues)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.8

    def test_calculate_overall_score_critical_issue(self):
        """Test overall score with critical issue."""
        reflection = SelfReflection()
        
        metrics = [
            QualityMetric(QualityDimension.READABILITY, 0.9),
            QualityMetric(QualityDimension.TESTABILITY, 0.9),
        ]
        issues = [
            IdentifiedIssue("critical", IssueSeverity.CRITICAL, "Critical issue"),
        ]
        
        score = reflection._calculate_overall_score(metrics, issues)
        
        assert score < 0.5

    def test_generate_improvement_suggestions(self):
        """Test improvement suggestion generation."""
        reflection = SelfReflection()
        
        issues = [
            IdentifiedIssue("issue1", IssueSeverity.CRITICAL, "Critical", suggestion="Fix critical"),
            IdentifiedIssue("issue2", IssueSeverity.HIGH, "High", suggestion="Fix high"),
            IdentifiedIssue("issue3", IssueSeverity.LOW, "Low", suggestion="Fix low"),
        ]
        metrics = [
            QualityMetric(QualityDimension.READABILITY, 0.4, "Low readability"),
        ]
        
        suggestions = reflection._generate_improvement_suggestions(issues, metrics)
        
        assert len(suggestions) > 0
        assert any("CRITICAL" in s for s in suggestions)

    def test_get_critique_stats(self):
        """Test critique statistics."""
        reflection = SelfReflection()
        
        reflection._critique_history = [
            CritiqueResult(
                overall_quality_score=0.8,
                quality_metrics=[],
                identified_issues=[],
                coverage_estimate=CoverageEstimate(0.5, 0.5, {}),
                improvement_suggestions=[],
                should_regenerate=False,
                confidence=0.9
            ),
            CritiqueResult(
                overall_quality_score=0.6,
                quality_metrics=[],
                identified_issues=[IdentifiedIssue("test", IssueSeverity.HIGH, "test")],
                coverage_estimate=CoverageEstimate(0.5, 0.5, {}),
                improvement_suggestions=[],
                should_regenerate=True,
                confidence=0.8
            ),
        ]
        
        stats = reflection.get_critique_stats()
        
        assert stats["total"] == 2
        assert stats["average_score"] == 0.7
        assert stats["regeneration_rate"] == 0.5


class TestQualityMetric:
    """Tests for QualityMetric dataclass."""

    def test_quality_metric_creation(self):
        """Test quality metric creation."""
        metric = QualityMetric(
            dimension=QualityDimension.READABILITY,
            score=0.85,
            details="Good readability"
        )
        
        assert metric.dimension == QualityDimension.READABILITY
        assert metric.score == 0.85
        assert metric.details == "Good readability"


class TestIdentifiedIssue:
    """Tests for IdentifiedIssue dataclass."""

    def test_identified_issue_creation(self):
        """Test identified issue creation."""
        issue = IdentifiedIssue(
            issue_type="MISSING_ASSERTION",
            severity=IssueSeverity.HIGH,
            description="No assertions found",
            line_number=10,
            suggestion="Add assertions",
            confidence=0.9
        )
        
        assert issue.issue_type == "MISSING_ASSERTION"
        assert issue.severity == IssueSeverity.HIGH
        assert issue.line_number == 10


class TestCoverageEstimate:
    """Tests for CoverageEstimate dataclass."""

    def test_coverage_estimate_creation(self):
        """Test coverage estimate creation."""
        estimate = CoverageEstimate(
            estimated_line_coverage=0.75,
            estimated_branch_coverage=0.5,
            estimated_method_coverage={"method1": 1.0, "method2": 0.5},
            uncovered_scenarios=["scenario1", "scenario2"]
        )
        
        assert estimate.estimated_line_coverage == 0.75
        assert estimate.estimated_branch_coverage == 0.5
        assert len(estimate.uncovered_scenarios) == 2


class TestCritiqueResult:
    """Tests for CritiqueResult dataclass."""

    def test_critique_result_creation(self):
        """Test critique result creation."""
        result = CritiqueResult(
            overall_quality_score=0.85,
            quality_metrics=[
                QualityMetric(QualityDimension.READABILITY, 0.9),
            ],
            identified_issues=[],
            coverage_estimate=CoverageEstimate(0.7, 0.6, {}),
            improvement_suggestions=["Add more tests"],
            should_regenerate=False,
            confidence=0.9
        )
        
        assert result.overall_quality_score == 0.85
        assert len(result.quality_metrics) == 1
        assert result.should_regenerate is False

    def test_get_issues_by_severity(self):
        """Test filtering issues by severity."""
        result = CritiqueResult(
            overall_quality_score=0.5,
            quality_metrics=[],
            identified_issues=[
                IdentifiedIssue("high1", IssueSeverity.HIGH, "High"),
                IdentifiedIssue("low1", IssueSeverity.LOW, "Low"),
                IdentifiedIssue("high2", IssueSeverity.HIGH, "High"),
            ],
            coverage_estimate=CoverageEstimate(0.5, 0.5, {}),
            improvement_suggestions=[],
            should_regenerate=True,
            confidence=0.8
        )
        
        high_issues = result.get_issues_by_severity(IssueSeverity.HIGH)
        
        assert len(high_issues) == 2

    def test_get_critical_issues(self):
        """Test getting critical issues."""
        result = CritiqueResult(
            overall_quality_score=0.3,
            quality_metrics=[],
            identified_issues=[
                IdentifiedIssue("critical1", IssueSeverity.CRITICAL, "Critical"),
                IdentifiedIssue("high1", IssueSeverity.HIGH, "High"),
            ],
            coverage_estimate=CoverageEstimate(0.5, 0.5, {}),
            improvement_suggestions=[],
            should_regenerate=True,
            confidence=0.9
        )
        
        critical = result.get_critical_issues()
        
        assert len(critical) == 1
        assert critical[0].severity == IssueSeverity.CRITICAL

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = CritiqueResult(
            overall_quality_score=0.85,
            quality_metrics=[
                QualityMetric(QualityDimension.READABILITY, 0.9, "Good"),
            ],
            identified_issues=[],
            coverage_estimate=CoverageEstimate(0.7, 0.6, {"m1": 1.0}),
            improvement_suggestions=["Suggestion"],
            should_regenerate=False,
            confidence=0.9
        )
        
        d = result.to_dict()
        
        assert d["overall_quality_score"] == 0.85
        assert "quality_metrics" in d
        assert "coverage_estimate" in d
        assert d["should_regenerate"] is False
