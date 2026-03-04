"""Tests for LLM coverage evaluator."""

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestLLMCoverageEvaluator:
    """Test suite for LLMCoverageEvaluator."""
    
    def test_init(self):
        """Test initialization."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator
        
        mock_client = Mock()
        evaluator = LLMCoverageEvaluator(mock_client)
        
        assert evaluator.llm_client == mock_client
    
    def test_extract_methods_info_from_class_info(self):
        """Test extracting methods info from class_info."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator
        
        evaluator = LLMCoverageEvaluator(None)
        
        class_info = {
            'methods': [
                {'name': 'getUser', 'visibility': 'public', 'parameters': ['id'], 'is_static': False, 'return_type': 'User'},
                {'name': 'saveUser', 'visibility': 'public', 'parameters': ['user'], 'is_static': False, 'return_type': 'void'},
                {'name': 'validate', 'visibility': 'private', 'parameters': [], 'is_static': True, 'return_type': 'boolean'},
            ]
        }
        
        result = evaluator._extract_methods_info(None, class_info)
        
        assert result['total_count'] == 3
        assert len(result['methods']) == 3
        assert result['methods'][0]['name'] == 'getUser'
        assert result['methods'][2]['is_static'] is True
    
    def test_extract_methods_info_from_source_code(self):
        """Test extracting methods info from source code."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator
        
        evaluator = LLMCoverageEvaluator(None)
        
        source_code = '''
        public class UserService {
            public User getUser(String id) { return null; }
            private void validate(User user) {}
            public static UserService create() { return new UserService(); }
        }
        '''
        
        result = evaluator._extract_methods_info(source_code, None)
        
        assert result['total_count'] >= 3
        method_names = [m['name'] for m in result['methods']]
        assert 'getUser' in method_names
        assert 'validate' in method_names
        assert 'create' in method_names
    
    def test_extract_test_methods_info(self):
        """Test extracting test methods info."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator
        
        evaluator = LLMCoverageEvaluator(None)
        
        test_code = '''
        public class UserServiceTest {
            @Test
            void testGetUser() {}
            
            @Test
            public void testSaveUser() {}
            
            @Test
            void testGetUser_whenNotFound_throwsException() {}
        }
        '''
        
        result = evaluator._extract_test_methods_info(test_code)
        
        assert result['total_count'] == 3
        method_names = [m['name'] for m in result['methods']]
        assert 'testGetUser' in method_names
        assert 'testSaveUser' in method_names
    
    def test_quick_estimate(self):
        """Test quick heuristic estimation."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator, CoverageSource
        
        evaluator = LLMCoverageEvaluator(None)
        
        source_code = '''
        public class Calculator {
            public int add(int a, int b) { return a + b; }
            public int subtract(int a, int b) { return a - b; }
            public int multiply(int a, int b) { return a * b; }
            public int divide(int a, int b) { return a / b; }
        }
        '''
        
        test_code = '''
        public class CalculatorTest {
            @Test
            void testAdd() {}
            @Test
            void testSubtract() {}
        }
        '''
        
        report = evaluator.quick_estimate(source_code, test_code)
        
        assert report.source == CoverageSource.LLM_ESTIMATED
        assert 0.0 <= report.line_coverage <= 1.0
        assert 0.0 <= report.branch_coverage <= 1.0
        assert 0.0 <= report.method_coverage <= 1.0
        assert report.confidence == 0.5
        assert len(report.covered_methods) <= 4
    
    def test_quick_estimate_with_class_info(self):
        """Test quick estimation with class info."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator
        
        evaluator = LLMCoverageEvaluator(None)
        
        source_code = "public class Service {}"
        test_code = "@Test void testMethod() {}"
        
        class_info = {
            'methods': [
                {'name': 'method1'},
                {'name': 'method2'},
                {'name': 'method3'},
            ]
        }
        
        report = evaluator.quick_estimate(source_code, test_code, class_info)
        
        assert report.method_coverage >= 0.0
    
    def test_create_fallback_report(self):
        """Test creating fallback report."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator, CoverageSource
        
        evaluator = LLMCoverageEvaluator(None)
        
        methods_info = {
            'methods': [
                {'name': 'getUser'},
                {'name': 'saveUser'},
                {'name': 'deleteUser'},
            ],
            'total_count': 3
        }
        
        test_methods_info = {
            'methods': [
                {'name': 'testGetUser'},
                {'name': 'testSaveUser'},
            ],
            'total_count': 2
        }
        
        report = evaluator._create_fallback_report(methods_info, test_methods_info)
        
        assert report.source == CoverageSource.LLM_ESTIMATED
        assert 'getUser' in report.covered_methods
        assert 'saveUser' in report.covered_methods
        assert 'deleteUser' in report.uncovered_methods
        assert report.method_coverage == 2 / 3
    
    def test_parse_llm_response_valid_json(self):
        """Test parsing valid LLM JSON response."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator, CoverageSource
        
        evaluator = LLMCoverageEvaluator(None)
        
        response = '''
        {
            "line_coverage": 0.75,
            "branch_coverage": 0.60,
            "method_coverage": 0.80,
            "covered_methods": ["getUser", "saveUser"],
            "uncovered_methods": ["deleteUser"],
            "uncovered_branches": ["if line 10"],
            "confidence": 0.85,
            "reasoning": "Test covers most methods",
            "recommendations": ["Add test for deleteUser"]
        }
        '''
        
        methods_info = {'methods': [], 'total_count': 0}
        report = evaluator._parse_llm_response(response, methods_info)
        
        assert report.line_coverage == 0.75
        assert report.branch_coverage == 0.60
        assert report.method_coverage == 0.80
        assert report.source == CoverageSource.LLM_ESTIMATED
        assert 'getUser' in report.covered_methods
        assert 'deleteUser' in report.uncovered_methods
        assert report.confidence == 0.85
    
    def test_parse_llm_response_clamped_values(self):
        """Test that parsed values are clamped to valid range."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator
        
        evaluator = LLMCoverageEvaluator(None)
        
        response = '''
        {
            "line_coverage": 1.5,
            "branch_coverage": -0.2,
            "method_coverage": 2.0
        }
        '''
        
        methods_info = {'methods': [], 'total_count': 0}
        report = evaluator._parse_llm_response(response, methods_info)
        
        assert 0.0 <= report.line_coverage <= 1.0
        assert 0.0 <= report.branch_coverage <= 1.0
        assert 0.0 <= report.method_coverage <= 1.0
    
    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON falls back to heuristic."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator
        
        evaluator = LLMCoverageEvaluator(None)
        
        response = "This is not valid JSON"
        
        methods_info = {'methods': [{'name': 'method1'}], 'total_count': 1}
        report = evaluator._parse_llm_response(response, methods_info)
        
        assert report.line_coverage >= 0.0
    
    @pytest.mark.asyncio
    async def test_evaluate_coverage_async(self):
        """Test async coverage evaluation."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator, CoverageSource
        
        mock_client = Mock()
        mock_client.agenerate = AsyncMock(return_value='''
        {
            "line_coverage": 0.85,
            "branch_coverage": 0.70,
            "method_coverage": 0.90,
            "covered_methods": ["method1"],
            "uncovered_methods": [],
            "confidence": 0.9,
            "reasoning": "Good coverage"
        }
        ''')
        
        evaluator = LLMCoverageEvaluator(mock_client)
        
        source_code = "public class Test { void method1() {} }"
        test_code = "@Test void testMethod1() {}"
        
        report = await evaluator.evaluate_coverage(source_code, test_code)
        
        assert report.source == CoverageSource.LLM_ESTIMATED
        assert report.line_coverage == 0.85
        mock_client.agenerate.assert_called_once()
    
    def test_evaluate_coverage_sync(self):
        """Test sync coverage evaluation."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageEvaluator, CoverageSource
        
        mock_client = Mock()
        mock_client.generate = Mock(return_value='''
        {
            "line_coverage": 0.75,
            "branch_coverage": 0.60,
            "method_coverage": 0.80,
            "covered_methods": ["method1"],
            "uncovered_methods": [],
            "confidence": 0.8,
            "reasoning": "Decent coverage"
        }
        ''')
        
        evaluator = LLMCoverageEvaluator(mock_client)
        
        source_code = "public class Test { void method1() {} }"
        test_code = "@Test void testMethod1() {}"
        
        report = evaluator.evaluate_coverage_sync(source_code, test_code)
        
        assert report.source == CoverageSource.LLM_ESTIMATED
        assert report.line_coverage == 0.75
        mock_client.generate.assert_called_once()


class TestLLMCoverageReport:
    """Test suite for LLMCoverageReport."""
    
    def test_create_report(self):
        """Test creating a coverage report."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageReport, CoverageSource
        
        report = LLMCoverageReport(
            line_coverage=0.75,
            branch_coverage=0.60,
            method_coverage=0.80,
            source=CoverageSource.LLM_ESTIMATED,
            uncovered_methods=['method3'],
            covered_methods=['method1', 'method2'],
            confidence=0.85,
            reasoning="Good coverage",
            recommendations=["Add test for method3"]
        )
        
        assert report.line_coverage == 0.75
        assert report.branch_coverage == 0.60
        assert report.method_coverage == 0.80
        assert report.source == CoverageSource.LLM_ESTIMATED
        assert 'method3' in report.uncovered_methods
        assert 'method1' in report.covered_methods
        assert report.confidence == 0.85
    
    def test_default_values(self):
        """Test default values."""
        from pyutagent.agent.llm_coverage_evaluator import LLMCoverageReport, CoverageSource
        
        report = LLMCoverageReport(
            line_coverage=0.5,
            branch_coverage=0.5,
            method_coverage=0.5
        )
        
        assert report.source == CoverageSource.LLM_ESTIMATED
        assert report.uncovered_methods == []
        assert report.uncovered_branches == []
        assert report.covered_methods == []
        assert report.confidence == 0.0
        assert report.reasoning == ""
        assert report.recommendations == []


class TestCoverageSource:
    """Test suite for CoverageSource enum."""
    
    def test_coverage_sources(self):
        """Test coverage source values."""
        from pyutagent.agent.llm_coverage_evaluator import CoverageSource
        
        assert CoverageSource.JACOCO.value == "jacoco"
        assert CoverageSource.LLM_ESTIMATED.value == "llm_estimated"
        assert CoverageSource.HYBRID.value == "hybrid"


class TestCreateLLMCoverageReport:
    """Test suite for factory function."""
    
    def test_create_llm_coverage_report(self):
        """Test factory function."""
        from pyutagent.agent.llm_coverage_evaluator import (
            create_llm_coverage_report,
            LLMCoverageReport,
            CoverageSource
        )
        
        report = create_llm_coverage_report(
            line_coverage=0.9,
            branch_coverage=0.8,
            method_coverage=0.95,
            source=CoverageSource.HYBRID,
            confidence=0.9,
            reasoning="Excellent coverage"
        )
        
        assert isinstance(report, LLMCoverageReport)
        assert report.line_coverage == 0.9
        assert report.source == CoverageSource.HYBRID
