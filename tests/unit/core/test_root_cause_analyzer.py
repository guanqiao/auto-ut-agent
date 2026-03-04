"""单元测试 - RootCauseAnalyzer"""

import pytest
from pathlib import Path

from pyutagent.core.root_cause_analyzer import (
    RootCauseAnalyzer,
    ErrorCategory,
    ErrorSeverity,
    FixStrategyType,
    CompilationError,
    TestFailure,
    RootCause,
    FixStrategy,
    RootCauseAnalysis
)


class TestErrorCategory:
    """测试 ErrorCategory 枚举"""
    
    def test_error_category_values(self):
        """测试错误类别值"""
        assert ErrorCategory.SYNTAX_ERROR.name == "SYNTAX_ERROR"
        assert ErrorCategory.TYPE_ERROR.name == "TYPE_ERROR"
        assert ErrorCategory.IMPORT_ERROR.name == "IMPORT_ERROR"
        assert ErrorCategory.REFERENCE_ERROR.name == "REFERENCE_ERROR"


class TestErrorSeverity:
    """测试 ErrorSeverity 枚举"""
    
    def test_severity_values(self):
        """测试严重程度值"""
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.LOW.value == "low"


class TestFixStrategyType:
    """测试 FixStrategyType 枚举"""
    
    def test_strategy_type_values(self):
        """测试修复策略类型值"""
        assert FixStrategyType.SYNTAX_FIX.name == "SYNTAX_FIX"
        assert FixStrategyType.TYPE_CORRECTION.name == "TYPE_CORRECTION"
        assert FixStrategyType.IMPORT_ADDITION.name == "IMPORT_ADDITION"


class TestCompilationError:
    """测试 CompilationError 数据类"""
    
    def test_compilation_error_creation(self):
        """测试编译错误创建"""
        error = CompilationError(
            error_id="comp_001",
            error_type="compilation",
            message="';' expected",
            file_path="/path/to/File.java",
            line_number=10,
            column_number=5,
            category=ErrorCategory.SYNTAX_ERROR,
            severity=ErrorSeverity.CRITICAL
        )
        
        assert error.error_id == "comp_001"
        assert error.line_number == 10
        assert error.category == ErrorCategory.SYNTAX_ERROR
        assert error.severity == ErrorSeverity.CRITICAL


class TestTestFailure:
    """测试 TestFailure 数据类"""
    
    def test_test_failure_creation(self):
        """测试测试失败创建"""
        failure = TestFailure(
            failure_id="fail_001",
            test_class="UserServiceTest",
            test_method="testGetUser",
            error_type="AssertionError",
            message="Expected 5 but was 3",
            stack_trace="at UserServiceTest.testGetUser",
            category=ErrorCategory.ASSERTION_ERROR,
            severity=ErrorSeverity.HIGH,
            expected_value="5",
            actual_value="3"
        )
        
        assert failure.failure_id == "fail_001"
        assert failure.test_class == "UserServiceTest"
        assert failure.expected_value == "5"
        assert failure.actual_value == "3"


class TestRootCause:
    """测试 RootCause 数据类"""
    
    def test_root_cause_creation(self):
        """测试根本原因创建"""
        cause = RootCause(
            cause_id="cause_001",
            description="Syntax errors in code",
            confidence=0.85,
            category=ErrorCategory.SYNTAX_ERROR,
            location="File.java:10",
            contributing_factors=["Missing semicolons"],
            evidence=["';' expected"]
        )
        
        assert cause.cause_id == "cause_001"
        assert cause.confidence == 0.85
        assert len(cause.contributing_factors) == 1


class TestFixStrategy:
    """测试 FixStrategy 数据类"""
    
    def test_fix_strategy_creation(self):
        """测试修复策略创建"""
        strategy = FixStrategy(
            strategy_id="fix_001",
            strategy_type=FixStrategyType.SYNTAX_FIX,
            description="Fix syntax errors",
            priority=1,
            estimated_effort="low",
            code_changes=["Add missing semicolons"],
            success_probability=0.95,
            side_effects=[]
        )
        
        assert strategy.strategy_id == "fix_001"
        assert strategy.priority == 1
        assert strategy.success_probability == 0.95


class TestRootCauseAnalysis:
    """测试 RootCauseAnalysis 数据类"""
    
    def test_root_cause_analysis_creation(self):
        """测试根因分析结果创建"""
        analysis = RootCauseAnalysis(
            errors=[],
            root_causes=[],
            suggested_fixes=[],
            analysis_summary="No errors",
            confidence_score=1.0,
            requires_manual_review=False
        )
        
        assert analysis.confidence_score == 1.0
        assert analysis.requires_manual_review is False
        assert len(analysis.errors) == 0


class TestRootCauseAnalyzer:
    """测试 RootCauseAnalyzer 类"""
    
    def setup_method(self):
        """测试前准备"""
        self.analyzer = RootCauseAnalyzer()
    
    def teardown_method(self):
        """测试后清理"""
        self.analyzer.clear_history()
    
    def test_initialization(self):
        """测试初始化"""
        analyzer = RootCauseAnalyzer()
        
        assert analyzer.error_patterns is not None
        assert analyzer.fix_strategies is not None
        assert len(analyzer.analysis_history) == 0
    
    def test_analyze_no_compilation_errors(self):
        """测试没有编译错误的情况"""
        output = "BUILD SUCCESS"
        
        analysis = self.analyzer.analyze_compilation_errors(output)
        
        assert len(analysis.errors) == 0
        assert len(analysis.root_causes) == 0
        assert analysis.confidence_score == 1.0
        assert "No compilation errors" in analysis.analysis_summary
    
    def test_analyze_syntax_error(self):
        """测试语法错误分析"""
        output = """
[ERROR] /path/to/File.java:10:5: error: ';' expected
[ERROR] /path/to/File.java:15:3: error: '}' expected
        """
        
        analysis = self.analyzer.analyze_compilation_errors(output)
        
        assert len(analysis.errors) > 0
        assert any(e.category == ErrorCategory.SYNTAX_ERROR for e in analysis.errors)
        assert analysis.confidence_score > 0.5
    
    def test_analyze_type_error(self):
        """测试类型错误分析"""
        output = """
[ERROR] /path/to/File.java:20:10: error: incompatible types: String cannot be converted to int
[ERROR] /path/to/File.java:25:5: error: incompatible types: required List but found Set
        """
        
        analysis = self.analyzer.analyze_compilation_errors(output)
        
        assert len(analysis.errors) > 0
        assert any(e.category == ErrorCategory.TYPE_ERROR for e in analysis.errors)
    
    def test_analyze_import_error(self):
        """测试导入错误分析"""
        output = """
[ERROR] /path/to/File.java:5:10: error: package org.mockito does not exist
[ERROR] /path/to/File.java:6:1: error: cannot find symbol symbol: class List
        """
        
        analysis = self.analyzer.analyze_compilation_errors(output)
        
        assert len(analysis.errors) > 0
        assert any(e.category == ErrorCategory.IMPORT_ERROR for e in analysis.errors)
    
    def test_analyze_reference_error(self):
        """测试引用错误分析"""
        output = """
[ERROR] /path/to/File.java:10:5: error: cannot find symbol
  symbol:   variable userId
  location: class UserService
[ERROR] /path/to/File.java:15:1: error: cannot find symbol
  symbol:   method calculate()
  location: class Calculator
        """
        
        analysis = self.analyzer.analyze_compilation_errors(output)
        
        assert len(analysis.errors) > 0
        # "cannot find symbol" 应该被分类为 REFERENCE_ERROR 或 IMPORT_ERROR
        assert any(e.category in [ErrorCategory.REFERENCE_ERROR, ErrorCategory.IMPORT_ERROR] for e in analysis.errors)
    
    def test_categorize_error_syntax(self):
        """测试错误分类 - 语法错误"""
        category = self.analyzer._categorize_error("';' expected")
        assert category == ErrorCategory.SYNTAX_ERROR
    
    def test_categorize_error_type(self):
        """测试错误分类 - 类型错误"""
        category = self.analyzer._categorize_error("incompatible types: String cannot be converted to int")
        assert category == ErrorCategory.TYPE_ERROR
    
    def test_categorize_error_import(self):
        """测试错误分类 - 导入错误"""
        category = self.analyzer._categorize_error("package org.mockito does not exist")
        assert category == ErrorCategory.IMPORT_ERROR
    
    def test_determine_severity_critical(self):
        """测试严重程度判断 - 关键"""
        severity = self.analyzer._determine_severity(ErrorCategory.SYNTAX_ERROR, "syntax error")
        assert severity == ErrorSeverity.CRITICAL
    
    def test_determine_severity_high(self):
        """测试严重程度判断 - 高"""
        severity = self.analyzer._determine_severity(ErrorCategory.REFERENCE_ERROR, "cannot find symbol")
        assert severity == ErrorSeverity.HIGH
    
    def test_analyze_test_no_failures(self):
        """测试没有测试失败的情况"""
        output = "BUILD SUCCESS\nTests run: 10, Failures: 0, Errors: 0"
        
        analysis = self.analyzer.analyze_test_failures(output)
        
        assert len(analysis.errors) == 0
        assert len(analysis.root_causes) == 0
        assert "No test failures" in analysis.analysis_summary
    
    def test_analyze_assertion_failure(self):
        """测试断言失败分析"""
        output = """
Tests run: 5, Failures: 1, Errors: 0
UserServiceTest.testGetUser
Expected: 5 but was: 3
        """
        
        analysis = self.analyzer.analyze_test_failures(output)
        
        assert len(analysis.errors) > 0
        assert any(e.category == ErrorCategory.ASSERTION_ERROR for e in analysis.errors)
    
    def test_analyze_null_pointer_exception(self):
        """测试空指针异常分析"""
        output = """
Tests run: 5, Failures: 0, Errors: 1
UserServiceTest.testUpdateUser
NullPointerException was thrown
        """
        
        analysis = self.analyzer.analyze_test_failures(output)
        
        assert len(analysis.errors) > 0
        assert any(e.category == ErrorCategory.NULL_POINTER_ERROR for e in analysis.errors)
    
    def test_identify_root_causes(self):
        """测试识别根本原因"""
        errors = [
            CompilationError(
                error_id="comp_001",
                error_type="compilation",
                message="';' expected",
                file_path="File.java",
                line_number=10,
                column_number=5,
                category=ErrorCategory.SYNTAX_ERROR,
                severity=ErrorSeverity.CRITICAL
            ),
            CompilationError(
                error_id="comp_002",
                error_type="compilation",
                message="'}' expected",
                file_path="File.java",
                line_number=15,
                column_number=3,
                category=ErrorCategory.SYNTAX_ERROR,
                severity=ErrorSeverity.CRITICAL
            )
        ]
        
        root_causes = self.analyzer._identify_root_causes(errors)
        
        assert len(root_causes) > 0
        assert all(isinstance(rc, RootCause) for rc in root_causes)
        assert all(rc.confidence > 0.5 for rc in root_causes)
    
    def test_generate_fix_strategies(self):
        """测试生成修复策略"""
        root_causes = [
            RootCause(
                cause_id="cause_001",
                description="Syntax errors",
                confidence=0.85,
                category=ErrorCategory.SYNTAX_ERROR,
                location="File.java:10",
                contributing_factors=["Missing semicolons"],
                evidence=["';' expected"]
            )
        ]
        
        strategies = self.analyzer._generate_fix_strategies(root_causes)
        
        assert len(strategies) > 0
        assert all(isinstance(s, FixStrategy) for s in strategies)
        # 应该按优先级排序
        assert strategies[0].priority <= strategies[-1].priority
    
    def test_calculate_confidence(self):
        """测试计算置信度"""
        root_causes = [
            RootCause(cause_id="c1", description="desc", confidence=0.9, category=ErrorCategory.SYNTAX_ERROR),
            RootCause(cause_id="c2", description="desc", confidence=0.8, category=ErrorCategory.TYPE_ERROR),
            RootCause(cause_id="c3", description="desc", confidence=0.7, category=ErrorCategory.IMPORT_ERROR)
        ]
        
        confidence = self.analyzer._calculate_confidence(root_causes)
        
        # 应该是平均值
        expected = (0.9 + 0.8 + 0.7) / 3
        assert abs(confidence - expected) < 0.01
    
    def test_generate_summary(self):
        """测试生成摘要"""
        errors = [
            CompilationError(
                error_id="comp_001",
                error_type="compilation",
                message="';' expected",
                file_path="File.java",
                line_number=10,
                column_number=5,
                category=ErrorCategory.SYNTAX_ERROR,
                severity=ErrorSeverity.CRITICAL
            )
        ]
        
        root_causes = [
            RootCause(
                cause_id="cause_001",
                description="Syntax errors",
                confidence=0.85,
                category=ErrorCategory.SYNTAX_ERROR
            )
        ]
        
        suggested_fixes = [
            FixStrategy(
                strategy_id="fix_001",
                strategy_type=FixStrategyType.SYNTAX_FIX,
                description="Fix syntax",
                priority=1,
                estimated_effort="low"
            )
        ]
        
        summary = self.analyzer._generate_summary(errors, root_causes, suggested_fixes)
        
        assert "1 compilation error" in summary
        assert "1 root cause" in summary
        assert "1 fix strategy" in summary
    
    def test_requires_manual_review_low_confidence(self):
        """测试是否需要人工审查 - 低置信度"""
        root_causes = [
            RootCause(cause_id="c1", description="desc", confidence=0.3, category=ErrorCategory.SYNTAX_ERROR)
        ]
        
        requires_review = self.analyzer._requires_manual_review(root_causes, confidence=0.3)
        
        assert requires_review is True
    
    def test_requires_manual_review_high_confidence(self):
        """测试是否需要人工审查 - 高置信度"""
        root_causes = [
            RootCause(cause_id="c1", description="desc", confidence=0.9, category=ErrorCategory.SYNTAX_ERROR)
        ]
        
        requires_review = self.analyzer._requires_manual_review(root_causes, confidence=0.9)
        
        assert requires_review is False
    
    def test_estimate_success_probability(self):
        """测试估算成功概率"""
        cause = RootCause(
            cause_id="c1",
            description="desc",
            confidence=0.8,
            category=ErrorCategory.SYNTAX_ERROR
        )
        
        prob = self.analyzer._estimate_success_probability(FixStrategyType.SYNTAX_FIX, cause)
        
        assert 0.0 <= prob <= 1.0
        # SYNTAX_FIX 基础概率 0.95 * 0.8 = 0.76
        assert prob > 0.7
    
    def test_estimate_effort(self):
        """测试估算工作量"""
        assert self.analyzer._estimate_effort(FixStrategyType.SYNTAX_FIX) == "low"
        assert self.analyzer._estimate_effort(FixStrategyType.IMPORT_ADDITION) == "low"
        assert self.analyzer._estimate_effort(FixStrategyType.LOGIC_CORRECTION) == "high"
    
    def test_get_analysis_history(self):
        """测试获取分析历史"""
        history = self.analyzer.get_analysis_history()
        assert isinstance(history, list)
        assert len(history) == 0
        
        # 执行一次分析
        self.analyzer.analyze_compilation_errors("[ERROR] File.java:10: error: ';' expected")
        
        history = self.analyzer.get_analysis_history()
        assert len(history) == 1
    
    def test_clear_history(self):
        """测试清除历史"""
        # 先添加一些数据
        self.analyzer.analyze_compilation_errors("[ERROR] File.java:10: error: ';' expected")
        
        self.analyzer.clear_history()
        
        assert len(self.analyzer.analysis_history) == 0
    
    def test_parse_compilation_errors_maven_format(self):
        """测试解析 Maven 格式编译错误"""
        output = """
[INFO] Compiling 1 source file
[ERROR] /path/to/UserService.java:15:5: error: ';' expected
[ERROR] /path/to/UserService.java:20:10: error: incompatible types
[INFO] ------------------------------------------------------------------------
        """
        
        errors = self.analyzer._parse_compilation_errors(output)
        
        assert len(errors) > 0
        assert errors[0].file_path.endswith("UserService.java")
        assert errors[0].line_number == 15
    
    def test_analyze_compilation_errors_comprehensive(self):
        """测试全面的编译错误分析"""
        output = """
[ERROR] /path/to/File.java:10:5: error: ';' expected
[ERROR] /path/to/File.java:15:3: error: '}' expected
[ERROR] /path/to/File.java:20:10: error: incompatible types: String cannot be converted to int
[ERROR] /path/to/File.java:25:1: error: cannot find symbol symbol: class List
[ERROR] /path/to/File.java:30:5: error: method does not override or implement a supertype method
        """
        
        analysis = self.analyzer.analyze_compilation_errors(output)
        
        assert len(analysis.errors) == 5
        assert len(analysis.root_causes) > 0
        assert len(analysis.suggested_fixes) > 0
        assert analysis.confidence_score > 0.5
        assert analysis.analysis_summary is not None
    
    def test_analyze_test_failures_comprehensive(self):
        """测试全面的测试失败分析"""
        output = """
Tests run: 10, Failures: 2, Errors: 1, Skipped: 0
UserServiceTest.testGetUser
Expected: 5 but was: 3

UserServiceTest.testUpdateUser
NullPointerException

UserServiceTest.testDeleteUser
MissingMethodCallException
        """
        
        analysis = self.analyzer.analyze_test_failures(output)
        
        assert len(analysis.errors) >= 2
        assert len(analysis.root_causes) > 0
        assert len(analysis.suggested_fixes) > 0


class TestRootCauseAnalyzerEdgeCases:
    """测试边界情况"""
    
    def setup_method(self):
        self.analyzer = RootCauseAnalyzer()
    
    def test_empty_output(self):
        """测试空输出"""
        analysis = self.analyzer.analyze_compilation_errors("")
        
        assert len(analysis.errors) == 0
        assert analysis.confidence_score == 1.0
    
    def test_mixed_warnings_and_errors(self):
        """测试混合警告和错误"""
        output = """
[WARNING] /path/to/File.java:10: warning: unused import
[ERROR] /path/to/File.java:15: error: ';' expected
[WARNING] /path/to/File.java:20: warning: deprecated API
        """
        
        analysis = self.analyzer.analyze_compilation_errors(output)
        
        # 应该只包含错误，不包含警告
        assert len(analysis.errors) >= 1
        assert all(e.severity != ErrorSeverity.LOW for e in analysis.errors)
    
    def test_errors_without_line_numbers(self):
        """测试没有行号的错误"""
        output = """
error: compilation failed
error: internal compiler error
        """
        
        errors = self.analyzer._parse_compilation_errors(output)
        
        # 应该能够解析，但行号为 None
        assert len(errors) > 0
        assert all(e.line_number is None for e in errors)
    
    def test_multiple_errors_same_category(self):
        """测试同一类别的多个错误"""
        output = """
[ERROR] /path/to/File1.java:10: error: ';' expected
[ERROR] /path/to/File2.java:15: error: ';' expected
[ERROR] /path/to/File3.java:20: error: ';' expected
        """
        
        analysis = self.analyzer.analyze_compilation_errors(output)
        
        # 应该聚类为一个根本原因
        assert len(analysis.errors) == 3
        assert len(analysis.root_causes) == 1
        assert analysis.root_causes[0].confidence > 0.7
    
    def test_test_output_without_failures(self):
        """测试没有失败的测试输出"""
        output = """
Tests run: 10, Failures: 0, Errors: 0, Skipped: 0
All tests passed successfully
        """
        
        analysis = self.analyzer.analyze_test_failures(output)
        
        assert len(analysis.errors) == 0
        assert "No test failures" in analysis.analysis_summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
