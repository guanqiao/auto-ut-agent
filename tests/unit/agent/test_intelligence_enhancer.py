"""单元测试 - IntelligenceEnhancer"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from pyutagent.agent.intelligence_enhancer import (
    IntelligenceEnhancer,
    IntelligenceResult,
    EnhancedGenerationPipeline,
    create_intelligence_enhancer,
    create_enhanced_pipeline
)
from pyutagent.core.semantic_analyzer import TestScenario, BoundaryCondition, BoundaryType
from pyutagent.core.root_cause_analyzer import RootCauseAnalysis, ErrorCategory


class TestIntelligenceResult:
    """测试 IntelligenceResult 数据类"""
    
    def test_intelligence_result_creation(self):
        """测试智能化结果创建"""
        result = IntelligenceResult(
            test_scenarios=[],
            boundary_conditions=[],
            root_cause_analysis=None,
            recommended_fixes=[],
            confidence_score=0.85
        )
        
        assert len(result.test_scenarios) == 0
        assert len(result.boundary_conditions) == 0
        assert result.confidence_score == 0.85
        assert result.requires_manual_review is False


class TestIntelligenceEnhancer:
    """测试 IntelligenceEnhancer 类"""
    
    def setup_method(self):
        """测试前准备"""
        self.enhancer = IntelligenceEnhancer()
    
    def teardown_method(self):
        """测试后清理"""
        self.enhancer.clear_cache()
    
    def test_initialization(self):
        """测试初始化"""
        enhancer = IntelligenceEnhancer()
        
        assert enhancer.semantic_analyzer is not None
        assert enhancer.root_cause_analyzer is not None
        assert len(enhancer.analysis_cache) == 0
    
    def test_analyze_target_code(self):
        """测试分析目标代码"""
        # 创建模拟的 JavaClass 对象
        mock_method = Mock()
        mock_method.name = "calculateTotal"
        mock_method.return_type = "int"
        mock_method.parameters = [("int", "amount"), ("int", "tax")]
        mock_method.modifiers = ["public"]
        mock_method.annotations = []
        mock_method.start_line = 10
        mock_method.end_line = 20
        
        mock_class = Mock()
        mock_class.name = "Calculator"
        mock_class.package = "com.example"
        mock_class.methods = [mock_method]
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        result = self.enhancer.analyze_target_code(
            "/path/to/Calculator.java",
            mock_class
        )
        
        assert isinstance(result, IntelligenceResult)
        assert len(result.test_scenarios) > 0
        assert len(result.boundary_conditions) > 0
        assert result.confidence_score > 0.5
    
    def test_analyze_target_code_caching(self):
        """测试分析结果缓存"""
        mock_method = Mock()
        mock_method.name = "test"
        mock_method.return_type = "void"
        mock_method.parameters = []
        mock_method.modifiers = ["public"]
        mock_method.annotations = []
        mock_method.start_line = 1
        mock_method.end_line = 5
        
        mock_class = Mock()
        mock_class.name = "TestClass"
        mock_class.package = "com.example"
        mock_class.methods = [mock_method]
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        # 第一次分析
        result1 = self.enhancer.analyze_target_code("/path/to/TestClass.java", mock_class)
        
        # 第二次分析 (应该使用缓存)
        result2 = self.enhancer.analyze_target_code("/path/to/TestClass.java", mock_class)
        
        # 应该是同一个对象
        assert result1 is result2
        
        # 检查缓存统计
        stats = self.enhancer.get_cache_stats()
        assert stats["cache_size"] >= 1
    
    def test_analyze_compilation_errors(self):
        """测试分析编译错误"""
        compiler_output = """
[ERROR] /path/to/File.java:10:5: error: ';' expected
[ERROR] /path/to/File.java:15:3: error: '}' expected
        """
        
        result = self.enhancer.analyze_compilation_errors(compiler_output)
        
        assert isinstance(result, IntelligenceResult)
        assert result.root_cause_analysis is not None
        assert len(result.root_cause_analysis.errors) > 0
        assert len(result.recommended_fixes) > 0
        assert result.confidence_score > 0.5
    
    def test_analyze_test_failures(self):
        """测试分析测试失败"""
        test_output = """
Tests run: 5, Failures: 1
CalculatorTest.testAdd
Expected: 5 but was: 3
        """
        
        result = self.enhancer.analyze_test_failures(test_output)
        
        assert isinstance(result, IntelligenceResult)
        assert result.root_cause_analysis is not None
        assert len(result.root_cause_analysis.errors) > 0
    
    def test_generate_enhanced_prompt(self):
        """测试生成增强提示词"""
        # 创建测试场景
        scenario = TestScenario(
            scenario_id="Test_001",
            description="Test with normal input",
            target_method="calculate",
            test_type="normal",
            setup_steps=["Create instance"],
            test_steps=["Call method", "Verify result"],
            expected_result="Success",
            priority=1
        )
        
        # 创建边界条件
        boundary = BoundaryCondition(
            parameter_name="amount",
            boundary_type=BoundaryType.RANGE_CHECK,
            description="Test with zero value",
            test_value=0,
            expected_behavior="Should handle zero correctly"
        )
        
        result = IntelligenceResult(
            test_scenarios=[scenario],
            boundary_conditions=[boundary],
            root_cause_analysis=None,
            recommended_fixes=["Fix syntax error"],
            confidence_score=0.85
        )
        
        base_prompt = "Generate unit tests for Calculator class"
        enhanced_prompt = self.enhancer.generate_enhanced_prompt(base_prompt, result)
        
        assert "Recommended Test Scenarios" in enhanced_prompt
        assert "Boundary Conditions to Test" in enhanced_prompt
        assert "Recommended Fixes" in enhanced_prompt
        assert "Analysis Confidence" in enhanced_prompt
        assert "Test with normal input" in enhanced_prompt
    
    def test_generate_enhanced_prompt_with_manual_review(self):
        """测试生成需要人工审查的增强提示词"""
        result = IntelligenceResult(
            test_scenarios=[],
            boundary_conditions=[],
            root_cause_analysis=None,
            recommended_fixes=[],
            confidence_score=0.3,
            requires_manual_review=True
        )
        
        base_prompt = "Generate tests"
        enhanced_prompt = self.enhancer.generate_enhanced_prompt(base_prompt, result)
        
        assert "Manual review recommended" in enhanced_prompt
    
    def test_get_test_scenario_for_method(self):
        """测试获取特定方法的测试场景"""
        scenario1 = TestScenario(
            scenario_id="Test_001",
            description="Test calculate method",
            target_method="calculate",
            test_type="normal",
            setup_steps=[],
            test_steps=[],
            expected_result="Success",
            priority=1
        )
        
        scenario2 = TestScenario(
            scenario_id="Test_002",
            description="Test validate method",
            target_method="validate",
            test_type="normal",
            setup_steps=[],
            test_steps=[],
            expected_result="Success",
            priority=1
        )
        
        result = IntelligenceResult(
            test_scenarios=[scenario1, scenario2],
            boundary_conditions=[],
            root_cause_analysis=None,
            recommended_fixes=[],
            confidence_score=0.85
        )
        
        scenarios = self.enhancer.get_test_scenario_for_method("calculate", result)
        
        assert len(scenarios) == 1
        assert scenarios[0].target_method == "calculate"
    
    def test_get_priority_scenarios(self):
        """测试获取高优先级测试场景"""
        scenarios = [
            TestScenario(
                scenario_id=f"Test_{i:03d}",
                description=f"Scenario {i}",
                target_method="method",
                test_type="normal" if i % 2 == 0 else "edge",
                setup_steps=[],
                test_steps=[],
                expected_result="Success",
                priority=i % 3 + 1
            )
            for i in range(10)
        ]
        
        result = IntelligenceResult(
            test_scenarios=scenarios,
            boundary_conditions=[],
            root_cause_analysis=None,
            recommended_fixes=[],
            confidence_score=0.85
        )
        
        priority_scenarios = self.enhancer.get_priority_scenarios(result, max_scenarios=5)
        
        assert len(priority_scenarios) == 5
        # 应该按优先级排序
        assert all(
            priority_scenarios[i].priority <= priority_scenarios[i+1].priority
            for i in range(len(priority_scenarios)-1)
        )
    
    def test_should_retry_generation_high_confidence(self):
        """测试是否应该重试 - 高置信度"""
        result = IntelligenceResult(
            test_scenarios=[],
            boundary_conditions=[],
            root_cause_analysis=None,
            recommended_fixes=[],
            confidence_score=0.9,
            requires_manual_review=False
        )
        
        should_retry = self.enhancer.should_retry_generation(result, max_retries=3, current_retry=1)
        
        assert should_retry is True
    
    def test_should_retry_generation_low_confidence(self):
        """测试是否应该重试 - 低置信度"""
        result = IntelligenceResult(
            test_scenarios=[],
            boundary_conditions=[],
            root_cause_analysis=None,
            recommended_fixes=[],
            confidence_score=0.3,
            requires_manual_review=False
        )
        
        should_retry = self.enhancer.should_retry_generation(result, max_retries=3, current_retry=1)
        
        assert should_retry is False
    
    def test_should_retry_generation_max_retries(self):
        """测试是否应该重试 - 达到最大重试次数"""
        result = IntelligenceResult(
            test_scenarios=[],
            boundary_conditions=[],
            root_cause_analysis=None,
            recommended_fixes=[],
            confidence_score=0.8,
            requires_manual_review=False
        )
        
        should_retry = self.enhancer.should_retry_generation(result, max_retries=3, current_retry=3)
        
        assert should_retry is False
    
    def test_clear_cache(self):
        """测试清除缓存"""
        # 先添加一些数据到缓存
        mock_class = Mock()
        mock_class.name = "Test"
        mock_class.methods = []
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        self.enhancer.analyze_target_code("/path/to/Test.java", mock_class)
        
        assert len(self.enhancer.analysis_cache) > 0
        
        # 清除缓存
        self.enhancer.clear_cache()
        
        assert len(self.enhancer.analysis_cache) == 0
    
    def test_get_cache_stats(self):
        """测试获取缓存统计"""
        stats = self.enhancer.get_cache_stats()
        
        assert "cache_size" in stats
        assert "cache_keys" in stats
        assert isinstance(stats["cache_size"], int)
        assert isinstance(stats["cache_keys"], list)


class TestEnhancedGenerationPipeline:
    """测试 EnhancedGenerationPipeline 类"""
    
    def setup_method(self):
        """测试前准备"""
        self.enhancer = IntelligenceEnhancer()
        self.pipeline = EnhancedGenerationPipeline(self.enhancer)
    
    def test_initialization(self):
        """测试初始化"""
        pipeline = EnhancedGenerationPipeline(self.enhancer)
        
        assert pipeline.enhancer is self.enhancer
    
    def test_generate_with_intelligence(self):
        """测试使用智能化增强生成"""
        # 创建模拟对象
        mock_method = Mock()
        mock_method.name = "calculate"
        mock_method.return_type = "int"
        mock_method.parameters = [("int", "a")]
        mock_method.modifiers = ["public"]
        mock_method.annotations = []
        mock_method.start_line = 1
        mock_method.end_line = 10
        
        mock_class = Mock()
        mock_class.name = "Calculator"
        mock_class.package = "com.example"
        mock_class.methods = [mock_method]
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        # 模拟 LLM 客户端
        mock_llm = Mock()
        mock_llm.generate.return_value = "public class CalculatorTest { }"
        
        # 模拟提示词构建器
        mock_prompt_builder = Mock()
        mock_prompt_builder.build_prompt.return_value = "Generate tests for Calculator"
        
        result = self.pipeline.generate_with_intelligence(
            "/path/to/Calculator.java",
            mock_class,
            mock_llm,
            mock_prompt_builder,
            max_retries=3
        )
        
        assert "success" in result
        assert "intelligence_result" in result
        assert "scenarios_covered" in result
        assert "retry_count" in result
        assert isinstance(result["intelligence_result"], IntelligenceResult)


class TestFactoryFunctions:
    """测试工厂函数"""
    
    def test_create_intelligence_enhancer(self):
        """测试创建智能化增强器"""
        enhancer = create_intelligence_enhancer()
        
        assert isinstance(enhancer, IntelligenceEnhancer)
    
    def test_create_enhanced_pipeline(self):
        """测试创建增强流水线"""
        pipeline = create_enhanced_pipeline()
        
        assert isinstance(pipeline, EnhancedGenerationPipeline)
        assert isinstance(pipeline.enhancer, IntelligenceEnhancer)


class TestIntelligenceEnhancerEdgeCases:
    """测试边界情况"""
    
    def setup_method(self):
        self.enhancer = IntelligenceEnhancer()
    
    def teardown_method(self):
        self.enhancer.clear_cache()
    
    def test_empty_class_analysis(self):
        """测试空类分析"""
        mock_class = Mock()
        mock_class.name = "EmptyClass"
        mock_class.methods = []
        mock_class.fields = []
        mock_class.imports = []
        mock_class.annotations = []
        
        result = self.enhancer.analyze_target_code("/path/to/EmptyClass.java", mock_class)
        
        assert isinstance(result, IntelligenceResult)
        assert len(result.test_scenarios) == 0
    
    def test_no_compilation_errors(self):
        """测试没有编译错误的情况"""
        result = self.enhancer.analyze_compilation_errors("BUILD SUCCESS")
        
        assert len(result.recommended_fixes) == 0
        assert result.confidence_score == 1.0
    
    def test_no_test_failures(self):
        """测试没有测试失败的情况"""
        result = self.enhancer.analyze_test_failures("BUILD SUCCESS")
        
        assert len(result.recommended_fixes) == 0
        assert result.confidence_score == 1.0
    
    def test_enhanced_prompt_without_scenarios(self):
        """测试没有测试场景时的增强提示词"""
        result = IntelligenceResult(
            test_scenarios=[],
            boundary_conditions=[],
            root_cause_analysis=None,
            recommended_fixes=[],
            confidence_score=0.85
        )
        
        enhanced_prompt = self.enhancer.generate_enhanced_prompt("Base prompt", result)
        
        assert "Base prompt" in enhanced_prompt
        assert "Confidence Score" in enhanced_prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
