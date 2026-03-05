"""Test Implement Agent - 测试实现专家

负责根据设计方案实现具体的测试代码。
"""

import asyncio
import logging
from typing import Dict, Any, Set, Optional, List

from ..multi_agent.specialized_agent import (
    SpecializedAgent,
    AgentCapability,
    AgentTask,
)
from ..multi_agent.message_bus import MessageBus
from ..multi_agent.shared_knowledge import SharedKnowledgeBase, ExperienceReplay

logger = logging.getLogger(__name__)


class TestImplementAgent(SpecializedAgent):
    """测试实现专家
    
    职责：
    - 根据测试设计实现测试代码
    - 生成Mock对象和测试数据
    - 实现测试断言
    - 确保测试代码符合规范
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
                AgentCapability.TEST_IMPLEMENTATION,
                AgentCapability.MOCK_GENERATION,
            },
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        self.llm_client = llm_client
        self.implementation_history: list = []
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """执行测试实现任务"""
        task_type = task.task_type
        payload = task.payload
        
        if task_type == "implement_tests":
            return await self._implement_tests(payload)
        elif task_type == "implement_single_test":
            return await self._implement_single_test(payload)
        elif task_type == "generate_mocks":
            return await self._generate_mocks(payload)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
    
    async def _implement_tests(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """根据设计实现测试代码
        
        Args:
            payload: 包含 test_design, class_info 等
            
        Returns:
            实现的测试代码
        """
        test_design = payload.get("test_design", {})
        class_info = payload.get("class_info", {})
        source_code = payload.get("source_code", "")
        
        logger.info(f"[TestImplementAgent] Implementing tests for: {class_info.get('name', 'Unknown')}")
        
        class_name = class_info.get("name", "UnknownClass")
        package = class_info.get("package", "")
        test_scenarios = test_design.get("test_scenarios", [])
        mock_strategy = test_design.get("mock_strategy", {})
        
        test_code = self._generate_test_class(
            class_name=class_name,
            package=package,
            test_scenarios=test_scenarios,
            mock_strategy=mock_strategy,
            class_info=class_info
        )
        
        implementation = {
            "test_class_name": f"{class_name}Test",
            "test_package": package,
            "test_code": test_code,
            "imports": self._generate_imports(class_info, mock_strategy),
            "mocks_generated": len(mock_strategy.get("required_mocks", [])),
            "test_methods_count": len(test_scenarios),
        }
        
        self.implementation_history.append(implementation)
        
        self.share_knowledge(
            item_type="test_implementation",
            content=implementation,
            confidence=0.9,
            tags=["test_implementation", class_name]
        )
        
        return {
            "success": True,
            "output": implementation,
            "metadata": {
                "test_methods": len(test_scenarios),
                "mocks": len(mock_strategy.get("required_mocks", []))
            }
        }
    
    async def _implement_single_test(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """实现单个测试方法"""
        scenario = payload.get("scenario", {})
        class_info = payload.get("class_info", {})
        
        test_method = self._generate_test_method(scenario, class_info)
        
        return {
            "success": True,
            "output": {
                "test_method": test_method,
                "method_name": scenario.get("name", "unknown_test")
            }
        }
    
    async def _generate_mocks(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """生成Mock对象"""
        mock_requirements = payload.get("mock_requirements", [])
        
        mocks = []
        for req in mock_requirements:
            mock_code = self._generate_mock_for_type(
                type_name=req.get("type", ""),
                methods=req.get("methods_to_mock", [])
            )
            mocks.append({
                "type": req.get("type", ""),
                "name": req.get("name", ""),
                "code": mock_code
            })
        
        return {
            "success": True,
            "output": {
                "mocks": mocks
            }
        }
    
    def _generate_test_class(
        self,
        class_name: str,
        package: str,
        test_scenarios: list,
        mock_strategy: Dict[str, Any],
        class_info: Dict[str, Any]
    ) -> str:
        """生成完整的测试类"""
        lines = []
        
        if package:
            lines.append(f"package {package};")
            lines.append("")
        
        imports = self._generate_imports(class_info, mock_strategy)
        for imp in imports:
            lines.append(f"import {imp};")
        lines.append("")
        
        test_class_name = f"{class_name}Test"
        lines.append(f"@ExtendWith(MockitoExtension.class)")
        lines.append(f"public class {test_class_name} {{")
        lines.append("")
        
        fields = self._generate_fields(class_info, mock_strategy)
        for field in fields:
            lines.append(f"    {field}")
        lines.append("")
        
        setup = self._generate_setup(class_info, mock_strategy)
        if setup:
            lines.append("    @BeforeEach")
            lines.append("    void setUp() {")
            for line in setup:
                lines.append(f"        {line}")
            lines.append("    }")
            lines.append("")
        
        for scenario in test_scenarios:
            method = self._generate_test_method(scenario, class_info)
            lines.extend(method)
            lines.append("")
        
        lines.append("}")
        
        return '\n'.join(lines)
    
    def _generate_imports(self, class_info: Dict[str, Any], mock_strategy: Dict[str, Any]) -> List[str]:
        """生成导入语句"""
        imports = [
            "org.junit.jupiter.api.Test",
            "org.junit.jupiter.api.BeforeEach",
            "org.junit.jupiter.api.DisplayName",
            "org.junit.jupiter.api.extension.ExtendWith",
            "org.mockito.Mock",
            "org.mockito.InjectMocks",
            "org.mockito.junit.jupiter.MockitoExtension",
            "static org.junit.jupiter.api.Assertions.*",
            "static org.mockito.Mockito.*",
        ]
        
        class_package = class_info.get("package", "")
        class_name = class_info.get("name", "")
        if class_package:
            imports.insert(0, f"{class_package}.{class_name}")
        
        return imports
    
    def _generate_fields(self, class_info: Dict[str, Any], mock_strategy: Dict[str, Any]) -> List[str]:
        """生成测试类字段"""
        fields = []
        
        class_name = class_info.get("name", "")
        fields.append(f"    @InjectMocks")
        fields.append(f"    private {class_name} underTest;")
        fields.append("")
        
        for mock in mock_strategy.get("required_mocks", []):
            mock_type = mock.get("type", "")
            mock_name = self._to_camel_case(mock.get("name", mock_type))
            fields.append(f"    @Mock")
            fields.append(f"    private {mock_type} {mock_name};")
        
        return fields
    
    def _generate_setup(self, class_info: Dict[str, Any], mock_strategy: Dict[str, Any]) -> List[str]:
        """生成setUp方法"""
        setup_lines = []
        
        for mock in mock_strategy.get("required_mocks", []):
            mock_type = mock.get("type", "")
            mock_name = self._to_camel_case(mock.get("name", mock_type))
            methods = mock.get("methods_to_mock", [])
            
            for method in methods[:3]:
                setup_lines.append(f"// when({mock_name}.{method}).thenReturn(...);")
        
        return setup_lines
    
    def _generate_test_method(self, scenario: Dict[str, Any], class_info: Dict[str, Any]) -> List[str]:
        """生成测试方法"""
        lines = []
        
        method_name = scenario.get("name", "testMethod")
        description = scenario.get("description", "")
        test_type = scenario.get("type", "positive")
        parameters = scenario.get("parameters", [])
        expected_result = scenario.get("expected_result", "success")
        
        lines.append(f'    @Test')
        lines.append(f'    @DisplayName("{description}")')
        lines.append(f'    void {method_name}() {{')
        
        for param in parameters:
            param_name = param.get("name", "")
            param_type = param.get("type", "Object")
            param_value = param.get("value", "null")
            lines.append(f'        {param_type} {param_name} = {param_value};')
        
        lines.append('')
        
        class_name = class_info.get("name", "UnderTest")
        target_method = scenario.get("method", "method")
        
        if expected_result == "exception":
            lines.append(f'        assertThrows(Exception.class, () -> {{')
            lines.append(f'            underTest.{target_method}({self._param_names(parameters)});')
            lines.append(f'        }});')
        else:
            lines.append(f'        // When')
            if scenario.get("return_type", "void") != "void":
                lines.append(f'        var result = underTest.{target_method}({self._param_names(parameters)});')
                lines.append('')
                lines.append(f'        // Then')
                lines.append(f'        assertNotNull(result);')
            else:
                lines.append(f'        underTest.{target_method}({self._param_names(parameters)});')
                lines.append('')
                lines.append(f'        // Then - verify interactions')
                lines.append(f'        // verify(...).someMethod(...);')
        
        lines.append(f'    }}')
        
        return lines
    
    def _generate_mock_for_type(self, type_name: str, methods: List[str]) -> str:
        """为类型生成Mock代码"""
        mock_name = self._to_camel_case(type_name)
        lines = [f"// Mock for {type_name}"]
        
        for method in methods:
            lines.append(f"when({mock_name}.{method}).thenReturn(/* mock return value */);")
        
        return '\n'.join(lines)
    
    def _to_camel_case(self, s: str) -> str:
        if not s:
            return ""
        if '.' in s:
            s = s.split('.')[-1]
        return s[0].lower() + s[1:] if s else ""
    
    def _param_names(self, parameters: list) -> str:
        return ', '.join(p.get("name", "") for p in parameters)
