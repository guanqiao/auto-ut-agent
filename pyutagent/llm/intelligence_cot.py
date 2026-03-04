"""智能化 CoT 提示工程 - 集成语义分析和根因分析的 Chain-of-Thought

本模块将 SemanticAnalyzer 和 RootCauseAnalyzer 的智能分析结果
集成到 Chain-of-Thought 提示中，增强 LLM 的理解和推理能力。
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .chain_of_thought import (
    ChainOfThoughtEngine,
    ChainOfThoughtPrompt,
    PromptCategory,
    ReasoningStep,
    ThoughtStep
)
from ..core.semantic_analyzer import SemanticAnalyzer, TestScenario, BoundaryCondition
from ..core.root_cause_analyzer import RootCauseAnalyzer, RootCauseAnalysis

logger = logging.getLogger(__name__)


@dataclass
class IntelligenceContext:
    """智能化上下文信息"""
    test_scenarios: List[Dict[str, Any]]
    boundary_conditions: List[Dict[str, Any]]
    business_logic: List[Dict[str, Any]]
    call_graph_info: Dict[str, Any]
    root_cause_analysis: Optional[RootCauseAnalysis]
    code_complexity: int
    recommended_focus_areas: List[str]


class IntelligenceEnhancedCoT:
    """智能化增强的 Chain-of-Thought 引擎
    
    功能:
    - 集成语义分析结果到 CoT 提示
    - 集成错误根因分析到修复提示
    - 动态生成针对性的推理步骤
    - 提供上下文感知的智能提示
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cot_engine = ChainOfThoughtEngine()
        self.semantic_analyzer = SemanticAnalyzer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        
        # 注册增强的提示模板
        self._register_enhanced_prompts()
        
        self.logger.info("[IntelligenceEnhancedCoT] Initialized with intelligence enhancements")
    
    def _register_enhanced_prompts(self):
        """注册增强的提示模板"""
        
        # 1. 语义分析增强的测试生成提示
        self.cot_engine.add_custom_prompt(
            name="semantic_enhanced_test_generation",
            category=PromptCategory.TEST_GENERATION,
            description="Test generation enhanced with semantic analysis insights",
            base_template="""Generate comprehensive unit tests using the following semantic analysis insights.

**Source Code:**
```java
${source_code}
```

**Semantic Analysis Results:**
- Class: ${class_name}
- Business Logic Types: ${business_logic_types}
- Method Complexity: ${complexity_scores}
- Data Flow Patterns: ${data_flows}

**Recommended Test Scenarios:**
${test_scenarios}

**Boundary Conditions to Test:**
${boundary_conditions}

**Call Graph Insights:**
${call_graph_info}

**Requirements:**
- Prioritize tests for high-complexity methods
- Cover all identified business logic types
- Test all boundary conditions
- Use appropriate mock strategies based on dependencies
""",
            variables=[
                "source_code", "class_name", "business_logic_types",
                "complexity_scores", "data_flows", "test_scenarios",
                "boundary_conditions", "call_graph_info"
            ],
            thought_steps=[
                {
                    "step_type": "analyze",
                    "instruction": "Review the semantic analysis results to understand the code structure, business logic types, and complexity.",
                    "expected_output": "Understanding of code semantics and test priorities",
                    "examples": [
                        "This is a service class with CALCULATION and PERSISTENCE logic",
                        "The calculateTotal method has high complexity (5) and should be prioritized"
                    ]
                },
                {
                    "step_type": "identify",
                    "instruction": "Identify test scenarios from the semantic analysis, focusing on business logic, boundary conditions, and data flow.",
                    "expected_output": "Comprehensive list of test scenarios with priorities",
                    "examples": [
                        "Test calculateTotal with normal values (Priority 1)",
                        "Test calculateTotal with negative values - boundary condition (Priority 2)",
                        "Test calculateTotal with null parameters - edge case (Priority 2)"
                    ]
                },
                {
                    "step_type": "plan",
                    "instruction": "Plan test structure based on call graph and data flow analysis. Identify what needs mocking.",
                    "expected_output": "Test plan with mock strategy",
                    "examples": [
                        "Mock UserRepository for database operations",
                        "Create separate test class for complex calculation logic"
                    ]
                },
                {
                    "step_type": "generate",
                    "instruction": "Generate tests following the semantic insights, ensuring coverage of all identified scenarios and boundaries.",
                    "expected_output": "Complete test suite with semantic coverage",
                    "examples": [
                        "@Test @DisplayName(\"Should calculate total with valid inputs\")",
                        "@Test @DisplayName(\"Should handle negative values gracefully\")"
                    ]
                },
                {
                    "step_type": "verify",
                    "instruction": "Verify all semantic scenarios are covered, boundary conditions tested, and complexity addressed.",
                    "expected_output": "Coverage verification against semantic analysis",
                    "examples": [
                        "✓ All business logic types covered",
                        "✓ All boundary conditions tested",
                        "✓ High-complexity methods have comprehensive tests"
                    ]
                }
            ]
        )
        
        # 2. 根因分析增强的错误修复提示
        self.cot_engine.add_custom_prompt(
            name="rca_enhanced_error_fix",
            category=PromptCategory.ERROR_ANALYSIS,
            description="Error fix enhanced with root cause analysis insights",
            base_template="""Fix test errors using systematic root cause analysis.

**Error Information:**
```
${error_message}
```

**Root Cause Analysis:**
- Category: ${error_category}
- Confidence: ${confidence_score}
- Contributing Factors: ${contributing_factors}
- Evidence: ${evidence}

**Recommended Fix Strategies:**
${recommended_fixes}

**Failing Test Code:**
```java
${test_code}
```

**Source Code:**
```java
${source_code}
```

**Fix Priority:** ${fix_priority}
**Estimated Effort:** ${estimated_effort}
**Success Probability:** ${success_probability}
""",
            variables=[
                "error_message", "error_category", "confidence_score",
                "contributing_factors", "evidence", "recommended_fixes",
                "test_code", "source_code", "fix_priority",
                "estimated_effort", "success_probability"
            ],
            thought_steps=[
                {
                    "step_type": "analyze",
                    "instruction": "Analyze the error using the root cause analysis. Understand the error category, confidence, and contributing factors.",
                    "expected_output": "Clear understanding of error root cause",
                    "examples": [
                        "TYPE_ERROR with 85% confidence - type mismatch in assignment",
                        "MOCK_ERROR - MissingMethodCallException indicates unstubbed mock method"
                    ]
                },
                {
                    "step_type": "understand",
                    "instruction": "Understand the context by reviewing the evidence and test intent.",
                    "expected_output": "Understanding of what the test was trying to verify",
                    "examples": [
                        "Test was verifying exception handling but mock not configured correctly",
                        "Expected type conversion but incompatible types used"
                    ]
                },
                {
                    "step_type": "identify",
                    "instruction": "Identify the best fix strategy from the recommendations based on success probability and effort.",
                    "expected_output": "Selected fix strategy with rationale",
                    "examples": [
                        "Choose TYPE_CORRECTION (90% success, low effort)",
                        "Apply MOCK_SETUP strategy (85% success, medium effort)"
                    ]
                },
                {
                    "step_type": "refine",
                    "instruction": "Apply the fix and verify it resolves the issue without side effects.",
                    "expected_output": "Fixed code with verification",
                    "examples": [
                        "Added type cast and verified compilation",
                        "Configured mock with when().thenReturn() and test passes"
                    ]
                }
            ]
        )
        
        # 3. 边界条件增强的测试生成提示
        self.cot_engine.add_custom_prompt(
            name="boundary_enhanced_generation",
            category=PromptCategory.BOUNDARY_ANALYSIS,
            description="Test generation focused on boundary condition coverage",
            base_template="""Generate tests with comprehensive boundary condition coverage.

**Method Signature:**
```java
${method_signature}
```

**Source Code:**
```java
${source_code}
```

**Identified Boundary Conditions:**
${boundary_conditions}

**Parameter Analysis:**
${parameter_analysis}

**Boundary Test Matrix:**
${boundary_matrix}

**Requirements:**
- Test each boundary condition explicitly
- Include both valid and invalid boundaries
- Verify error handling for out-of-bounds inputs
- Document boundary assumptions
""",
            variables=[
                "method_signature", "source_code", "boundary_conditions",
                "parameter_analysis", "boundary_matrix"
            ],
            thought_steps=[
                {
                    "step_type": "analyze",
                    "instruction": "Analyze method parameters and identify all boundary conditions from semantic analysis.",
                    "expected_output": "Complete boundary condition inventory",
                    "examples": [
                        "String parameter: null, empty, whitespace-only, very long",
                        "int parameter: min value, zero, max value, negative"
                    ]
                },
                {
                    "step_type": "identify",
                    "instruction": "Identify which boundaries are critical vs. nice-to-have based on business logic.",
                    "expected_output": "Prioritized boundary conditions",
                    "examples": [
                        "Critical: null check for required parameters",
                        "Important: empty collection handling",
                        "Nice-to-have: very large inputs"
                    ]
                },
                {
                    "step_type": "plan",
                    "instruction": "Plan test methods to cover boundary conditions efficiently, grouping related boundaries.",
                    "expected_output": "Test plan for boundary coverage",
                    "examples": [
                        "Group all null checks in one test class",
                        "Separate tests for different boundary types"
                    ]
                },
                {
                    "step_type": "generate",
                    "instruction": "Generate boundary tests with clear naming and documentation.",
                    "expected_output": "Boundary test suite",
                    "examples": [
                        "@Test @DisplayName(\"Should reject null username\")",
                        "@ParameterizedTest for multiple boundary values"
                    ]
                }
            ]
        )
        
        # 4. 业务逻辑增强的测试设计提示
        self.cot_engine.add_custom_prompt(
            name="business_logic_test_design",
            category=PromptCategory.TEST_GENERATION,
            description="Test design based on business logic analysis",
            base_template="""Design tests that thoroughly verify business logic.

**Class Under Test:**
```java
${class_code}
```

**Business Logic Analysis:**
${business_logic_analysis}

**Logic Type Distribution:**
${logic_distribution}

**Preconditions and Postconditions:**
${conditions}

**Side Effects:**
${side_effects}

**Test Design Requirements:**
- Verify each business logic type
- Test preconditions and postconditions
- Verify side effects occur correctly
- Use appropriate assertion strategies
""",
            variables=[
                "class_code", "business_logic_analysis", "logic_distribution",
                "conditions", "side_effects"
            ],
            thought_steps=[
                {
                    "step_type": "analyze",
                    "instruction": "Analyze the business logic types identified (VALIDATION, TRANSFORMATION, PERSISTENCE, etc.).",
                    "expected_output": "Understanding of business responsibilities",
                    "examples": [
                        "VALIDATION logic: input sanitization, constraint checking",
                        "TRANSFORMATION logic: data conversion, calculation"
                    ]
                },
                {
                    "step_type": "identify",
                    "instruction": "Identify the key business rules and invariants that must be tested.",
                    "expected_output": "Business rules inventory",
                    "examples": [
                        "Business rule: Email must be unique",
                        "Invariant: Account balance cannot be negative"
                    ]
                },
                {
                    "step_type": "plan",
                    "instruction": "Plan tests to verify each business rule with appropriate scenarios.",
                    "expected_output": "Business logic test plan",
                    "examples": [
                        "Test validation rules with valid and invalid inputs",
                        "Test transformations with edge cases"
                    ]
                },
                {
                    "step_type": "generate",
                    "instruction": "Generate tests that clearly express business intent and verify rules.",
                    "expected_output": "Business-focused test suite",
                    "examples": [
                        "@Test void shouldRejectDuplicateEmail()",
                        "@Test void shouldCalculateDiscountCorrectly()"
                    ]
                }
            ]
        )
        
        self.logger.info("[IntelligenceEnhancedCoT] Registered 4 enhanced prompt templates")
    
    def generate_enhanced_test_prompt(
        self,
        source_code: str,
        java_class: Any,
        file_path: str
    ) -> str:
        """生成增强的测试生成提示
        
        Args:
            source_code: 源代码
            java_class: 解析后的 JavaClass 对象
            file_path: 文件路径
            
        Returns:
            增强的 CoT 提示
        """
        self.logger.info(f"[IntelligenceEnhancedCoT] Generating enhanced test prompt for {file_path}")
        
        # 1. 语义分析
        semantic_result = self.semantic_analyzer.analyze_file(file_path, java_class)
        
        # 2. 构建智能上下文
        intelligence_context = self._build_intelligence_context(semantic_result)
        
        # 3. 准备提示变量
        context = self._prepare_test_generation_context(
            source_code, java_class, semantic_result, intelligence_context
        )
        
        # 4. 渲染增强的提示
        enhanced_prompt = self.cot_engine.render_prompt(
            "semantic_enhanced_test_generation",
            context
        )
        
        self.logger.info(f"[IntelligenceEnhancedCoT] Generated enhanced prompt with "
                        f"{len(intelligence_context.test_scenarios)} scenarios, "
                        f"{len(intelligence_context.boundary_conditions)} boundaries")
        
        return enhanced_prompt
    
    def generate_enhanced_fix_prompt(
        self,
        error_message: str,
        test_code: str,
        source_code: str,
        test_method: str
    ) -> str:
        """生成增强的错误修复提示
        
        Args:
            error_message: 错误消息
            test_code: 测试代码
            source_code: 源代码
            test_method: 测试方法名
            
        Returns:
            增强的 CoT 修复提示
        """
        self.logger.info(f"[IntelligenceEnhancedCoT] Generating enhanced fix prompt for {test_method}")
        
        # 1. 根因分析
        rca_result = self.root_cause_analyzer.analyze_test_failures(
            error_message,
            test_code,
            source_code
        )
        
        # 2. 准备提示变量
        context = self._prepare_error_fix_context(
            error_message, test_code, source_code, test_method, rca_result
        )
        
        # 3. 渲染增强的提示
        enhanced_prompt = self.cot_engine.render_prompt(
            "rca_enhanced_error_fix",
            context
        )
        
        self.logger.info(f"[IntelligenceEnhancedCoT] Generated enhanced fix prompt with "
                        f"{len(rca_result.root_causes)} root causes, "
                        f"{len(rca_result.suggested_fixes)} fix strategies")
        
        return enhanced_prompt
    
    def generate_boundary_focused_prompt(
        self,
        source_code: str,
        java_class: Any,
        file_path: str,
        target_method: Optional[str] = None
    ) -> str:
        """生成边界条件聚焦的提示
        
        Args:
            source_code: 源代码
            java_class: 解析后的 JavaClass 对象
            file_path: 文件路径
            target_method: 可选的目标方法名
            
        Returns:
            边界条件聚焦的 CoT 提示
        """
        self.logger.info(f"[IntelligenceEnhancedCoT] Generating boundary-focused prompt")
        
        # 1. 语义分析
        semantic_result = self.semantic_analyzer.analyze_file(file_path, java_class)
        
        # 2. 过滤目标方法的边界条件
        boundaries = semantic_result.get("boundary_conditions", [])
        if target_method:
            # 简化版本，实际应该根据方法参数过滤
            boundaries = boundaries[:10]  # 限制数量
        
        # 3. 准备提示变量
        context = {
            "method_signature": self._extract_method_signature(java_class, target_method),
            "source_code": source_code,
            "boundary_conditions": self._format_boundary_conditions(boundaries),
            "parameter_analysis": self._analyze_parameters(java_class, target_method),
            "boundary_matrix": self._create_boundary_matrix(boundaries)
        }
        
        # 4. 渲染提示
        enhanced_prompt = self.cot_engine.render_prompt(
            "boundary_enhanced_generation",
            context
        )
        
        return enhanced_prompt
    
    def _build_intelligence_context(self, semantic_result: Dict[str, Any]) -> IntelligenceContext:
        """构建智能上下文"""
        return IntelligenceContext(
            test_scenarios=semantic_result.get("test_scenarios", []),
            boundary_conditions=semantic_result.get("boundary_conditions", []),
            business_logic=semantic_result.get("business_logic", []),
            call_graph_info=semantic_result.get("call_graph", {}),
            root_cause_analysis=None,
            code_complexity=self._calculate_overall_complexity(semantic_result),
            recommended_focus_areas=self._identify_focus_areas(semantic_result)
        )
    
    def _prepare_test_generation_context(
        self,
        source_code: str,
        java_class: Any,
        semantic_result: Dict[str, Any],
        intelligence_context: IntelligenceContext
    ) -> Dict[str, Any]:
        """准备测试生成上下文"""
        return {
            "source_code": source_code,
            "class_name": java_class.name,
            "business_logic_types": self._format_business_logic_types(intelligence_context.business_logic),
            "complexity_scores": self._format_complexity_scores(intelligence_context.call_graph_info),
            "data_flows": self._format_data_flows(semantic_result.get("data_flows", [])),
            "test_scenarios": self._format_test_scenarios(intelligence_context.test_scenarios),
            "boundary_conditions": self._format_boundary_conditions(intelligence_context.boundary_conditions),
            "call_graph_info": self._format_call_graph_info(intelligence_context.call_graph_info)
        }
    
    def _prepare_error_fix_context(
        self,
        error_message: str,
        test_code: str,
        source_code: str,
        test_method: str,
        rca_result: RootCauseAnalysis
    ) -> Dict[str, Any]:
        """准备错误修复上下文"""
        root_cause = rca_result.root_causes[0] if rca_result.root_causes else None
        fix_strategy = rca_result.suggested_fixes[0] if rca_result.suggested_fixes else None
        
        return {
            "error_message": error_message,
            "error_category": root_cause.category.name if root_cause else "UNKNOWN",
            "confidence_score": f"{rca_result.confidence_score:.0%}",
            "contributing_factors": ", ".join(root_cause.contributing_factors) if root_cause else "N/A",
            "evidence": "; ".join(root_cause.evidence[:3]) if root_cause else "N/A",
            "recommended_fixes": self._format_recommended_fixes(rca_result.suggested_fixes),
            "test_code": test_code,
            "source_code": source_code,
            "test_method": test_method,
            "fix_priority": fix_strategy.priority if fix_strategy else "N/A",
            "estimated_effort": fix_strategy.estimated_effort if fix_strategy else "N/A",
            "success_probability": f"{fix_strategy.success_probability:.0%}" if fix_strategy else "N/A"
        }
    
    def _format_business_logic_types(self, business_logic: List[Dict[str, Any]]) -> str:
        """格式化业务逻辑类型"""
        if not business_logic:
            return "No business logic identified"
        
        logic_types = set(bl.get("type", "UNKNOWN") for bl in business_logic)
        return ", ".join(logic_types)
    
    def _format_complexity_scores(self, call_graph_info: Dict[str, Any]) -> str:
        """格式化复杂度分数"""
        if not call_graph_info or "nodes" not in call_graph_info:
            return "No complexity data"
        
        nodes = call_graph_info.get("nodes", [])
        complexity_info = [
            f"{node.get('method', 'unknown')}: {node.get('complexity', 1)}"
            for node in nodes[:5]  # 限制数量
        ]
        return "; ".join(complexity_info)
    
    def _format_data_flows(self, data_flows: List[Dict[str, Any]]) -> str:
        """格式化数据流"""
        if not data_flows:
            return "No data flows identified"
        
        flow_info = [
            f"{flow.get('source', '?')} -> {flow.get('sink', '?')}"
            for flow in data_flows[:5]
        ]
        return "; ".join(flow_info)
    
    def _format_test_scenarios(self, scenarios: List[Dict[str, Any]]) -> str:
        """格式化测试场景"""
        if not scenarios:
            return "No test scenarios identified"
        
        formatted = []
        for scenario in scenarios[:10]:  # 限制数量
            formatted.append(
                f"- {scenario.get('description', 'Unknown scenario')}\n"
                f"  Target: {scenario.get('target', 'N/A')}\n"
                f"  Type: {scenario.get('type', 'normal')}\n"
                f"  Priority: {scenario.get('priority', 3)}"
            )
        return "\n\n".join(formatted)
    
    def _format_boundary_conditions(self, boundaries: List[Dict[str, Any]]) -> str:
        """格式化边界条件"""
        if not boundaries:
            return "No boundary conditions identified"
        
        formatted = []
        for boundary in boundaries[:10]:  # 限制数量
            formatted.append(
                f"- Parameter: {boundary.get('parameter', 'N/A')}\n"
                f"  Type: {boundary.get('type', 'UNKNOWN')}\n"
                f"  Test Value: {boundary.get('test_value', 'N/A')}\n"
                f"  Expected: {boundary.get('expected_behavior', 'Handle gracefully')}"
            )
        return "\n\n".join(formatted)
    
    def _format_call_graph_info(self, call_graph_info: Dict[str, Any]) -> str:
        """格式化调用图信息"""
        if not call_graph_info:
            return "No call graph information"
        
        entry_points = call_graph_info.get("entry_points", [])
        nodes = call_graph_info.get("nodes", [])
        
        info_parts = [
            f"Entry Points: {', '.join(entry_points[:5])}" if entry_points else "",
            f"Total Methods: {len(nodes)}"
        ]
        
        return "; ".join(filter(None, info_parts))
    
    def _format_recommended_fixes(self, fixes: List[Any]) -> str:
        """格式化推荐修复"""
        if not fixes:
            return "No fix strategies recommended"
        
        formatted = []
        for i, fix in enumerate(fixes[:5], 1):
            formatted.append(
                f"{i}. {fix.description}\n"
                f"   Priority: {fix.priority}, Effort: {fix.estimated_effort}, "
                f"Success: {fix.success_probability:.0%}"
            )
        return "\n\n".join(formatted)
    
    def _calculate_overall_complexity(self, semantic_result: Dict[str, Any]) -> int:
        """计算整体复杂度"""
        call_graph = semantic_result.get("call_graph", {})
        nodes = call_graph.get("nodes", [])
        
        if not nodes:
            return 1
        
        complexities = [node.get("complexity", 1) for node in nodes]
        return sum(complexities) // len(complexities)
    
    def _identify_focus_areas(self, semantic_result: Dict[str, Any]) -> List[str]:
        """识别重点关注区域"""
        focus_areas = []
        
        # 高复杂度方法
        call_graph = semantic_result.get("call_graph", {})
        nodes = call_graph.get("nodes", [])
        high_complexity = [
            node.get("method", "unknown")
            for node in nodes
            if node.get("complexity", 1) > 3
        ]
        if high_complexity:
            focus_areas.append(f"High complexity methods: {', '.join(high_complexity[:3])}")
        
        # 边界条件多的参数
        boundaries = semantic_result.get("boundary_conditions", [])
        if len(boundaries) > 5:
            focus_areas.append(f"Parameters with multiple boundary conditions ({len(boundaries)})")
        
        # 业务逻辑类型
        business_logic = semantic_result.get("business_logic", [])
        logic_types = set(bl.get("type", "") for bl in business_logic)
        if len(logic_types) > 2:
            focus_areas.append(f"Multiple business logic types: {', '.join(logic_types)}")
        
        return focus_areas
    
    def _extract_method_signature(self, java_class: Any, method_name: Optional[str]) -> str:
        """提取方法签名"""
        if not method_name:
            return f"class {java_class.name}"
        
        for method in java_class.methods:
            if method.name == method_name:
                params = ", ".join([f"{t} {n}" for t, n in method.parameters])
                return f"{method.return_type} {method.name}({params})"
        
        return f"method {method_name} not found"
    
    def _analyze_parameters(self, java_class: Any, method_name: Optional[str]) -> str:
        """分析参数"""
        if not method_name:
            return "No specific method targeted"
        
        for method in java_class.methods:
            if method.name == method_name:
                if not method.parameters:
                    return "No parameters"
                param_info = [
                    f"{t} {n} - requires boundary testing"
                    for t, n in method.parameters
                ]
                return "\n".join(param_info)
        
        return "Method not found"
    
    def _create_boundary_matrix(self, boundaries: List[Dict[str, Any]]) -> str:
        """创建边界矩阵"""
        if not boundaries:
            return "No boundary conditions"
        
        matrix = []
        for boundary in boundaries[:5]:
            matrix.append(
                f"| {boundary.get('parameter', '?')} | "
                f"{boundary.get('type', '?')} | "
                f"{boundary.get('test_value', '?')} |"
            )
        
        return "Parameter | Boundary | Value\n" + "\n".join(matrix)
    
    def get_available_enhanced_prompts(self) -> List[Dict[str, str]]:
        """获取可用的增强提示列表"""
        return [
            {
                "name": "semantic_enhanced_test_generation",
                "category": "test_generation",
                "description": "Test generation with semantic analysis insights"
            },
            {
                "name": "rca_enhanced_error_fix",
                "category": "error_analysis",
                "description": "Error fix with root cause analysis"
            },
            {
                "name": "boundary_enhanced_generation",
                "category": "boundary_analysis",
                "description": "Boundary-focused test generation"
            },
            {
                "name": "business_logic_test_design",
                "category": "test_generation",
                "description": "Business logic-driven test design"
            }
        ]


def create_intelligence_enhanced_cot() -> IntelligenceEnhancedCoT:
    """创建智能化增强的 CoT 引擎实例"""
    return IntelligenceEnhancedCoT()
