"""智能化增强模块 - 集成 SemanticAnalyzer 和 RootCauseAnalyzer

本模块将智能化分析组件集成到 UT 生成流程中:
- 使用 SemanticAnalyzer 进行代码语义分析和测试场景识别
- 使用 RootCauseAnalyzer 进行错误根因分析和修复策略推荐
- 增强测试生成的智能性和准确性
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..core.semantic_analyzer import SemanticAnalyzer, TestScenario, BoundaryCondition
from ..core.root_cause_analyzer import (
    RootCauseAnalyzer,
    RootCauseAnalysis,
    CompilationError,
    TestFailure
)

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceResult:
    """智能化分析结果"""
    test_scenarios: List[TestScenario]
    boundary_conditions: List[BoundaryCondition]
    root_cause_analysis: Optional[RootCauseAnalysis]
    recommended_fixes: List[str]
    confidence_score: float
    requires_manual_review: bool = False


class IntelligenceEnhancer:
    """智能化增强器
    
    功能:
    - 代码语义分析
    - 测试场景智能识别
    - 错误根因分析
    - 修复策略推荐
    - 智能决策支持
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.semantic_analyzer = SemanticAnalyzer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.analysis_cache: Dict[str, IntelligenceResult] = {}
    
    def analyze_target_code(
        self,
        file_path: str,
        java_class: Any
    ) -> IntelligenceResult:
        """分析目标代码
        
        Args:
            file_path: Java 文件路径
            java_class: 解析后的 JavaClass 对象
            
        Returns:
            智能化分析结果
        """
        self.logger.info(f"[IntelligenceEnhancer] Analyzing target code: {file_path}")
        
        # 检查缓存
        cache_key = f"{file_path}:{hash(str(java_class))}"
        if cache_key in self.analysis_cache:
            self.logger.info("[IntelligenceEnhancer] Using cached analysis result")
            return self.analysis_cache[cache_key]
        
        # 使用 SemanticAnalyzer 分析代码
        semantic_result = self.semantic_analyzer.analyze_file(file_path, java_class)
        
        # 从 semantic_result 中提取测试场景
        test_scenarios_data = semantic_result.get("test_scenarios", [])
        test_scenarios = [
            TestScenario(
                scenario_id=ts["id"],
                description=ts["description"],
                target_method=ts["target"],
                test_type=ts["type"],
                setup_steps=ts.get("setup", []),
                test_steps=ts.get("steps", []),
                expected_result=ts.get("expected", ""),
                priority=ts.get("priority", 1)
            )
            for ts in test_scenarios_data
        ]
        
        # 提取边界条件
        boundary_conditions = semantic_result.get("boundary_conditions", [])
        boundary_objs = [
            BoundaryCondition(
                parameter_name=bc["parameter"],
                boundary_type=self._map_boundary_type(bc["type"]),
                description=bc["description"],
                test_value=bc["test_value"],
                expected_behavior=bc["expected_behavior"]
            )
            for bc in boundary_conditions
        ]
        
        # 创建分析结果
        result = IntelligenceResult(
            test_scenarios=test_scenarios,
            boundary_conditions=boundary_objs,
            root_cause_analysis=None,
            recommended_fixes=[],
            confidence_score=semantic_result.get("confidence", 0.8),
            requires_manual_review=False
        )
        
        # 缓存结果
        self.analysis_cache[cache_key] = result
        
        self.logger.info(f"[IntelligenceEnhancer] Analysis complete - "
                        f"Scenarios: {len(test_scenarios)}, "
                        f"Boundaries: {len(boundary_objs)}")
        
        return result
    
    def analyze_compilation_errors(
        self,
        compiler_output: str,
        source_code: Optional[str] = None
    ) -> IntelligenceResult:
        """分析编译错误
        
        Args:
            compiler_output: 编译器输出
            source_code: 可选的源代码
            
        Returns:
            智能化分析结果
        """
        self.logger.info("[IntelligenceEnhancer] Analyzing compilation errors")
        
        # 使用 RootCauseAnalyzer 分析错误
        rca_result = self.root_cause_analyzer.analyze_compilation_errors(
            compiler_output,
            source_code
        )
        
        # 提取推荐修复
        recommended_fixes = [
            f"[Priority {fix.priority}] {fix.description} "
            f"(Effort: {fix.estimated_effort}, Success: {fix.success_probability:.0%})"
            for fix in rca_result.suggested_fixes[:5]  # 前 5 个修复建议
        ]
        
        result = IntelligenceResult(
            test_scenarios=[],
            boundary_conditions=[],
            root_cause_analysis=rca_result,
            recommended_fixes=recommended_fixes,
            confidence_score=rca_result.confidence_score,
            requires_manual_review=rca_result.requires_manual_review
        )
        
        self.logger.info(f"[IntelligenceEnhancer] Compilation error analysis complete - "
                        f"Errors: {len(rca_result.errors)}, "
                        f"Root Causes: {len(rca_result.root_causes)}, "
                        f"Confidence: {rca_result.confidence_score:.2f}")
        
        return result
    
    def analyze_test_failures(
        self,
        test_output: str,
        test_code: Optional[str] = None,
        source_code: Optional[str] = None
    ) -> IntelligenceResult:
        """分析测试失败
        
        Args:
            test_output: 测试输出
            test_code: 可选的测试代码
            source_code: 可选的源代码
            
        Returns:
            智能化分析结果
        """
        self.logger.info("[IntelligenceEnhancer] Analyzing test failures")
        
        # 使用 RootCauseAnalyzer 分析失败
        rca_result = self.root_cause_analyzer.analyze_test_failures(
            test_output,
            test_code,
            source_code
        )
        
        # 提取推荐修复
        recommended_fixes = [
            f"[Priority {fix.priority}] {fix.description} "
            f"(Effort: {fix.estimated_effort}, Success: {fix.success_probability:.0%})"
            for fix in rca_result.suggested_fixes[:5]  # 前 5 个修复建议
        ]
        
        result = IntelligenceResult(
            test_scenarios=[],
            boundary_conditions=[],
            root_cause_analysis=rca_result,
            recommended_fixes=recommended_fixes,
            confidence_score=rca_result.confidence_score,
            requires_manual_review=rca_result.requires_manual_review
        )
        
        self.logger.info(f"[IntelligenceEnhancer] Test failure analysis complete - "
                        f"Failures: {len(rca_result.errors)}, "
                        f"Root Causes: {len(rca_result.root_causes)}, "
                        f"Confidence: {rca_result.confidence_score:.2f}")
        
        return result
    
    def generate_enhanced_prompt(
        self,
        base_prompt: str,
        intelligence_result: IntelligenceResult,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成增强提示词
        
        Args:
            base_prompt: 基础提示词
            intelligence_result: 智能化分析结果
            context: 可选的上下文信息
            
        Returns:
            增强后的提示词
        """
        enhanced_prompt = base_prompt
        
        # 添加测试场景信息
        if intelligence_result.test_scenarios:
            enhanced_prompt += "\n\n=== Recommended Test Scenarios ===\n"
            for i, scenario in enumerate(intelligence_result.test_scenarios[:10], 1):
                enhanced_prompt += (
                    f"\n{i}. {scenario.description}\n"
                    f"   Target: {scenario.target_method}\n"
                    f"   Type: {scenario.test_type}\n"
                    f"   Priority: {scenario.priority}\n"
                    f"   Steps: {'; '.join(scenario.test_steps[:2])}\n"
                )
        
        # 添加边界条件信息
        if intelligence_result.boundary_conditions:
            enhanced_prompt += "\n\n=== Boundary Conditions to Test ===\n"
            for i, bc in enumerate(intelligence_result.boundary_conditions[:10], 1):
                enhanced_prompt += (
                    f"\n{i}. Parameter: {bc.parameter_name}\n"
                    f"   Type: {bc.boundary_type.name}\n"
                    f"   Test Value: {bc.test_value}\n"
                    f"   Expected: {bc.expected_behavior}\n"
                )
        
        # 添加修复建议
        if intelligence_result.recommended_fixes:
            enhanced_prompt += "\n\n=== Recommended Fixes ===\n"
            for i, fix in enumerate(intelligence_result.recommended_fixes, 1):
                enhanced_prompt += f"\n{i}. {fix}\n"
        
        # 添加置信度信息
        enhanced_prompt += (
            f"\n\n=== Analysis Confidence ===\n"
            f"Confidence Score: {intelligence_result.confidence_score:.0%}\n"
        )
        
        if intelligence_result.requires_manual_review:
            enhanced_prompt += "⚠️ Manual review recommended due to low confidence or complex issues.\n"
        
        return enhanced_prompt
    
    def get_test_scenario_for_method(
        self,
        method_name: str,
        intelligence_result: IntelligenceResult
    ) -> List[TestScenario]:
        """获取特定方法的测试场景
        
        Args:
            method_name: 方法名
            intelligence_result: 智能化分析结果
            
        Returns:
            相关的测试场景列表
        """
        return [
            scenario for scenario in intelligence_result.test_scenarios
            if scenario.target_method == method_name
        ]
    
    def get_priority_scenarios(
        self,
        intelligence_result: IntelligenceResult,
        max_scenarios: int = 5
    ) -> List[TestScenario]:
        """获取高优先级测试场景
        
        Args:
            intelligence_result: 智能化分析结果
            max_scenarios: 最大场景数量
            
        Returns:
            高优先级测试场景列表
        """
        scenarios = sorted(
            intelligence_result.test_scenarios,
            key=lambda s: (s.priority, s.test_type == "normal")
        )
        return scenarios[:max_scenarios]
    
    def should_retry_generation(
        self,
        intelligence_result: IntelligenceResult,
        max_retries: int = 3,
        current_retry: int = 0
    ) -> bool:
        """判断是否应该重试生成
        
        Args:
            intelligence_result: 智能化分析结果
            max_retries: 最大重试次数
            current_retry: 当前重试次数
            
        Returns:
            是否应该重试
        """
        # 置信度高且不需要人工审查，建议重试
        if (intelligence_result.confidence_score > 0.7 and
            not intelligence_result.requires_manual_review):
            return current_retry < max_retries
        
        # 置信度低，建议人工审查
        if intelligence_result.confidence_score < 0.4:
            logger.warning("[IntelligenceEnhancer] Low confidence, manual review recommended")
            return False
        
        # 默认继续重试
        return current_retry < max_retries
    
    def _map_boundary_type(self, type_name: str) -> Any:
        """映射边界条件类型"""
        from ..core.semantic_analyzer import BoundaryType
        
        type_mapping = {
            "NULL_CHECK": BoundaryType.NULL_CHECK,
            "EMPTY_CHECK": BoundaryType.EMPTY_CHECK,
            "RANGE_CHECK": BoundaryType.RANGE_CHECK,
            "TYPE_CHECK": BoundaryType.TYPE_CHECK,
            "STATE_CHECK": BoundaryType.STATE_CHECK,
            "PERMISSION_CHECK": BoundaryType.PERMISSION_CHECK,
        }
        return type_mapping.get(type_name, BoundaryType.TYPE_CHECK)
    
    def clear_cache(self):
        """清除缓存"""
        self.analysis_cache.clear()
        self.logger.info("[IntelligenceEnhancer] Analysis cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "cache_size": len(self.analysis_cache),
            "cache_keys": list(self.analysis_cache.keys())
        }


class EnhancedGenerationPipeline:
    """增强生成流水线
    
    将智能化分析集成到 UT 生成流程中:
    1. 语义分析 -> 2. 测试场景识别 -> 3. 智能生成 -> 4. 错误分析 -> 5. 智能修复
    """
    
    def __init__(self, intelligence_enhancer: IntelligenceEnhancer):
        self.enhancer = intelligence_enhancer
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_with_intelligence(
        self,
        file_path: str,
        java_class: Any,
        llm_client: Any,
        prompt_builder: Any,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """使用智能化增强生成测试
        
        Args:
            file_path: Java 文件路径
            java_class: 解析后的 JavaClass 对象
            llm_client: LLM 客户端
            prompt_builder: 提示词构建器
            max_retries: 最大重试次数
            
        Returns:
            生成结果
        """
        self.logger.info(f"[EnhancedGenerationPipeline] Starting intelligent generation for {file_path}")
        
        # 1. 语义分析
        intelligence_result = self.enhancer.analyze_target_code(file_path, java_class)
        
        # 2. 获取高优先级测试场景
        priority_scenarios = self.enhancer.get_priority_scenarios(intelligence_result)
        
        # 3. 构建增强提示词
        base_prompt = prompt_builder.build_prompt(java_class)
        enhanced_prompt = self.enhancer.generate_enhanced_prompt(
            base_prompt,
            intelligence_result
        )
        
        # 4. LLM 生成
        retry_count = 0
        while retry_count < max_retries:
            # 调用 LLM
            response = llm_client.generate(enhanced_prompt)
            
            # 这里应该集成编译和测试步骤
            # 简化版本，直接返回
            if self._validate_generation(response):
                return {
                    "success": True,
                    "test_code": response,
                    "intelligence_result": intelligence_result,
                    "scenarios_covered": len(priority_scenarios),
                    "retry_count": retry_count
                }
            
            retry_count += 1
        
        return {
            "success": False,
            "test_code": None,
            "intelligence_result": intelligence_result,
            "scenarios_covered": 0,
            "retry_count": retry_count
        }
    
    def _validate_generation(self, code: str) -> bool:
        """验证生成的代码"""
        # 简化验证，实际应该编译和运行测试
        return code is not None and len(code) > 0


def create_intelligence_enhancer() -> IntelligenceEnhancer:
    """创建智能化增强器实例"""
    return IntelligenceEnhancer()


def create_enhanced_pipeline() -> EnhancedGenerationPipeline:
    """创建增强生成流水线实例"""
    enhancer = create_intelligence_enhancer()
    return EnhancedGenerationPipeline(enhancer)
