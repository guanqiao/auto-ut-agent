"""Test Design Agent - 测试设计专家

负责分析源代码并设计测试策略和测试用例。
"""

import asyncio
import logging
from typing import Dict, Any, Set, Optional

from ..multi_agent.specialized_agent import (
    SpecializedAgent,
    AgentCapability,
    AgentTask,
    TaskResult,
)
from ..multi_agent.message_bus import MessageBus
from ..multi_agent.shared_knowledge import SharedKnowledgeBase, ExperienceReplay

logger = logging.getLogger(__name__)


class TestDesignAgent(SpecializedAgent):
    """测试设计专家
    
    职责：
    - 分析源代码结构和方法签名
    - 识别测试场景（正向、负向、边界）
    - 设计测试用例和测试数据
    - 确定Mock策略
    - 评估测试覆盖率目标
    """
    
    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        knowledge_base: SharedKnowledgeBase,
        experience_replay: Optional[ExperienceReplay] = None,
        llm_client: Optional[Any] = None
    ):
        super().__init__(
            agent_id=agent_id,
            capabilities={
                AgentCapability.TEST_DESIGN,
                AgentCapability.DEPENDENCY_ANALYSIS,
                AgentCapability.MOCK_GENERATION,
            },
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        self.llm_client = llm_client
        self.design_history: list = []
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """执行测试设计任务
        
        Args:
            task: 包含源代码信息的任务
            
        Returns:
            测试设计方案
        """
        task_type = task.task_type
        payload = task.payload
        
        if task_type == "design_tests":
            return await self._design_tests(payload)
        elif task_type == "analyze_coverage_gaps":
            return await self._analyze_coverage_gaps(payload)
        elif task_type == "design_edge_cases":
            return await self._design_edge_cases(payload)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
    
    async def _design_tests(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """设计测试用例
        
        Args:
            payload: 包含 class_info, source_code 等
            
        Returns:
            测试设计方案
        """
        class_info = payload.get("class_info", {})
        source_code = payload.get("source_code", "")
        target_coverage = payload.get("target_coverage", 0.8)
        
        logger.info(f"[TestDesignAgent] Designing tests for class: {class_info.get('name', 'Unknown')}")
        
        design = {
            "class_name": class_info.get("name", ""),
            "package": class_info.get("package", ""),
            "test_scenarios": [],
            "mock_strategy": {},
            "test_data_requirements": [],
            "coverage_strategy": {},
            "priority_methods": [],
        }
        
        methods = class_info.get("methods", [])
        for method in methods:
            method_name = method.get("name", "")
            method_params = method.get("parameters", [])
            return_type = method.get("return_type", "void")
            
            scenarios = await self._design_method_scenarios(
                method_name=method_name,
                parameters=method_params,
                return_type=return_type,
                method_annotations=method.get("annotations", [])
            )
            
            design["test_scenarios"].extend(scenarios)
            
            if self._is_high_priority_method(method):
                design["priority_methods"].append(method_name)
        
        design["mock_strategy"] = self._determine_mock_strategy(class_info)
        design["coverage_strategy"] = {
            "target": target_coverage,
            "focus_areas": design["priority_methods"],
            "approach": "boundary_value_analysis" if len(methods) > 5 else "exhaustive"
        }
        
        self.design_history.append(design)
        
        self.share_knowledge(
            item_type="test_design",
            content=design,
            confidence=0.85,
            tags=["test_design", class_info.get("name", "unknown")]
        )
        
        return {
            "success": True,
            "output": design,
            "metadata": {
                "scenarios_count": len(design["test_scenarios"]),
                "priority_methods": len(design["priority_methods"])
            }
        }
    
    async def _design_method_scenarios(
        self,
        method_name: str,
        parameters: list,
        return_type: str,
        method_annotations: list
    ) -> list:
        """为方法设计测试场景"""
        scenarios = []
        
        scenarios.append({
            "method": method_name,
            "type": "positive",
            "name": f"test{self._capitalize(method_name)}_success",
            "description": f"Test {method_name} with valid inputs",
            "parameters": self._generate_valid_params(parameters),
            "expected_result": "success",
            "priority": "high"
        })
        
        for param in parameters:
            if self._is_nullable_param(param):
                scenarios.append({
                    "method": method_name,
                    "type": "negative",
                    "name": f"test{self._capitalize(method_name)}_null{self._capitalize(param.get('name', ''))}",
                    "description": f"Test {method_name} with null {param.get('name', '')}",
                    "parameters": self._generate_null_params(parameters, param),
                    "expected_result": "exception_or_handled",
                    "priority": "medium"
                })
        
        for param in parameters:
            if self._has_boundary_values(param):
                boundary_values = self._get_boundary_values(param)
                for bv in boundary_values[:2]:
                    scenarios.append({
                        "method": method_name,
                        "type": "boundary",
                        "name": f"test{self._capitalize(method_name)}_{param.get('name', '')}_{bv['name']}",
                        "description": f"Test {method_name} with {bv['name']} {param.get('name', '')}",
                        "parameters": self._generate_boundary_params(parameters, param, bv['value']),
                        "expected_result": "handled",
                        "priority": "medium"
                    })
        
        return scenarios
    
    async def _analyze_coverage_gaps(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """分析覆盖率缺口"""
        coverage_data = payload.get("coverage_data", {})
        class_info = payload.get("class_info", {})
        
        gaps = []
        for method, coverage in coverage_data.get("method_coverage", {}).items():
            if coverage < 0.8:
                gaps.append({
                    "method": method,
                    "current_coverage": coverage,
                    "missing_scenarios": self._identify_missing_scenarios(method, coverage)
                })
        
        return {
            "success": True,
            "output": {
                "gaps": gaps,
                "recommendations": self._generate_gap_recommendations(gaps)
            }
        }
    
    async def _design_edge_cases(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """设计边界测试用例"""
        method_info = payload.get("method_info", {})
        
        edge_cases = []
        parameters = method_info.get("parameters", [])
        
        for param in parameters:
            param_type = param.get("type", "")
            edge_cases.extend(self._get_edge_cases_for_type(param_type, param.get("name", "")))
        
        return {
            "success": True,
            "output": {
                "edge_cases": edge_cases
            }
        }
    
    def _determine_mock_strategy(self, class_info: Dict[str, Any]) -> Dict[str, Any]:
        """确定Mock策略"""
        dependencies = class_info.get("dependencies", [])
        
        mock_strategy = {
            "required_mocks": [],
            "mock_framework": "mockito",
            "approach": "interface_mocking"
        }
        
        for dep in dependencies:
            if self._should_mock(dep):
                mock_strategy["required_mocks"].append({
                    "type": dep.get("type", ""),
                    "name": dep.get("name", ""),
                    "methods_to_mock": dep.get("used_methods", [])
                })
        
        return mock_strategy
    
    def _is_high_priority_method(self, method: Dict[str, Any]) -> bool:
        """判断是否为高优先级方法"""
        name = method.get("name", "")
        annotations = method.get("annotations", [])
        
        if name.startswith(("get", "set", "is")):
            return False
        
        if "Deprecated" in annotations:
            return False
        
        return True
    
    def _capitalize(self, s: str) -> str:
        return s[0].upper() + s[1:] if s else ""
    
    def _generate_valid_params(self, parameters: list) -> list:
        result = []
        for p in parameters:
            p_type = p.get("type", "Object")
            result.append({
                "name": p.get("name", ""),
                "type": p_type,
                "value": self._get_default_value(p_type)
            })
        return result
    
    def _get_default_value(self, type_name: str) -> Any:
        defaults = {
            "int": 1, "Integer": 1,
            "long": 1L, "Long": 1L,
            "double": 1.0, "Double": 1.0,
            "float": 1.0, "Float": 1.0,
            "boolean": True, "Boolean": True,
            "String": "\"test\"",
            "List": "Collections.emptyList()",
            "Map": "Collections.emptyMap()",
        }
        return defaults.get(type_name, "null")
    
    def _is_nullable_param(self, param: Dict[str, Any]) -> bool:
        annotations = param.get("annotations", [])
        return "Nullable" in annotations or param.get("type", "").startswith("Optional")
    
    def _generate_null_params(self, parameters: list, null_param: Dict[str, Any]) -> list:
        result = []
        for p in parameters:
            if p.get("name") == null_param.get("name"):
                result.append({"name": p.get("name", ""), "type": p.get("type", ""), "value": "null"})
            else:
                result.append({"name": p.get("name", ""), "type": p.get("type", ""), "value": self._get_default_value(p.get("type", "Object"))})
        return result
    
    def _has_boundary_values(self, param: Dict[str, Any]) -> bool:
        p_type = param.get("type", "")
        return p_type in ["int", "Integer", "long", "Long", "double", "Double", "String"]
    
    def _get_boundary_values(self, param: Dict[str, Any]) -> list:
        p_type = param.get("type", "")
        boundaries = {
            "int": [{"name": "min", "value": 0}, {"name": "max", "value": 2147483647}, {"name": "negative", "value": -1}],
            "Integer": [{"name": "min", "value": 0}, {"name": "max", "value": 2147483647}],
            "String": [{"name": "empty", "value": "\"\""}, {"name": "long", "value": "\"a\".repeat(1000)"}],
        }
        return boundaries.get(p_type, [])
    
    def _generate_boundary_params(self, parameters: list, target_param: Dict[str, Any], value: Any) -> list:
        result = []
        for p in parameters:
            if p.get("name") == target_param.get("name"):
                result.append({"name": p.get("name", ""), "type": p.get("type", ""), "value": value})
            else:
                result.append({"name": p.get("name", ""), "type": p.get("type", ""), "value": self._get_default_value(p.get("type", "Object"))})
        return result
    
    def _identify_missing_scenarios(self, method: str, coverage: float) -> list:
        if coverage < 0.3:
            return ["basic_positive", "basic_negative", "exception_handling"]
        elif coverage < 0.6:
            return ["edge_cases", "null_handling"]
        else:
            return ["rare_edge_cases"]
    
    def _generate_gap_recommendations(self, gaps: list) -> list:
        recommendations = []
        for gap in gaps:
            recommendations.append(f"Add tests for {gap['method']} to cover {', '.join(gap['missing_scenarios'])}")
        return recommendations
    
    def _get_edge_cases_for_type(self, type_name: str, param_name: str) -> list:
        edge_cases = []
        if type_name in ["int", "Integer", "long", "Long"]:
            edge_cases.extend([
                {"param": param_name, "value": 0, "description": "Zero value"},
                {"param": param_name, "value": -1, "description": "Negative value"},
                {"param": param_name, "value": "Integer.MAX_VALUE", "description": "Max integer"},
            ])
        elif type_name in ["String"]:
            edge_cases.extend([
                {"param": param_name, "value": "\"\"", "description": "Empty string"},
                {"param": param_name, "value": "\" \"", "description": "Whitespace only"},
                {"param": param_name, "value": "\"a\".repeat(10000)", "description": "Very long string"},
            ])
        elif type_name in ["List", "Collection"]:
            edge_cases.extend([
                {"param": param_name, "value": "Collections.emptyList()", "description": "Empty list"},
                {"param": param_name, "value": "null", "description": "Null list"},
            ])
        return edge_cases
    
    def _should_mock(self, dependency: Dict[str, Any]) -> bool:
        dep_type = dependency.get("type", "")
        non_mock_prefixes = ["java.lang", "java.util", "java.time"]
        return not any(dep_type.startswith(prefix) for prefix in non_mock_prefixes)
