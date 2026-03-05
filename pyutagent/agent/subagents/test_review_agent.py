"""Test Review Agent - 测试审查专家

负责审查测试代码质量，确保符合最佳实践。
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


class TestReviewAgent(SpecializedAgent):
    """测试审查专家
    
    职责：
    - 审查测试代码质量
    - 检查测试覆盖完整性
    - 验证测试命名规范
    - 检查断言有效性
    - 识别测试代码坏味道
    """
    
    QUALITY_CRITERIA = {
        "naming": {
            "test_method_prefix": "test_或should_开头",
            "descriptive_names": "测试方法名应描述测试场景",
            "class_naming": "测试类名应以Test结尾",
        },
        "structure": {
            "single_responsibility": "每个测试方法只测试一个场景",
            "arrange_act_assert": "遵循AAA模式",
            "no_logic_in_tests": "测试中不应有复杂逻辑",
        },
        "assertions": {
            "meaningful_assertions": "断言应有意义且充分",
            "assertion_messages": "断言应包含失败消息",
            "verify_mocks": "Mock交互应被验证",
        },
        "coverage": {
            "happy_path": "覆盖正常路径",
            "error_path": "覆盖异常路径",
            "edge_cases": "覆盖边界情况",
        }
    }
    
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
                AgentCapability.TEST_REVIEW,
                AgentCapability.COVERAGE_ANALYSIS,
            },
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        self.llm_client = llm_client
        self.review_history: list = []
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """执行测试审查任务"""
        task_type = task.task_type
        payload = task.payload
        
        if task_type == "review_tests":
            return await self._review_tests(payload)
        elif task_type == "check_coverage":
            return await self._check_coverage(payload)
        elif task_type == "identify_smells":
            return await self._identify_smells(payload)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
    
    async def _review_tests(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """审查测试代码
        
        Args:
            payload: 包含 test_code, class_info 等
            
        Returns:
            审查结果和改进建议
        """
        test_code = payload.get("test_code", "")
        class_info = payload.get("class_info", {})
        test_design = payload.get("test_design", {})
        
        logger.info(f"[TestReviewAgent] Reviewing tests for: {class_info.get('name', 'Unknown')}")
        
        issues = []
        suggestions = []
        
        naming_issues = self._check_naming_conventions(test_code, class_info)
        issues.extend(naming_issues)
        
        structure_issues = self._check_structure(test_code)
        issues.extend(structure_issues)
        
        assertion_issues = self._check_assertions(test_code)
        issues.extend(assertion_issues)
        
        coverage_issues = self._check_coverage_completeness(test_code, test_design, class_info)
        issues.extend(coverage_issues)
        
        for issue in issues:
            suggestions.append(self._generate_suggestion(issue))
        
        quality_score = self._calculate_quality_score(issues)
        
        review_result = {
            "test_class": class_info.get("name", ""),
            "quality_score": quality_score,
            "issues": issues,
            "suggestions": suggestions,
            "passed": quality_score >= 0.7,
            "summary": self._generate_review_summary(issues, quality_score)
        }
        
        self.review_history.append(review_result)
        
        self.share_knowledge(
            item_type="test_review",
            content=review_result,
            confidence=0.85,
            tags=["test_review", class_info.get("name", "unknown")]
        )
        
        return {
            "success": True,
            "output": review_result,
            "metadata": {
                "issues_count": len(issues),
                "quality_score": quality_score
            }
        }
    
    async def _check_coverage(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """检查测试覆盖率"""
        test_code = payload.get("test_code", "")
        class_info = payload.get("class_info", {})
        coverage_data = payload.get("coverage_data", {})
        
        uncovered_methods = []
        methods = class_info.get("methods", [])
        
        for method in methods:
            method_name = method.get("name", "")
            if not self._is_method_tested(method_name, test_code):
                uncovered_methods.append(method_name)
        
        return {
            "success": True,
            "output": {
                "uncovered_methods": uncovered_methods,
                "coverage_percentage": (len(methods) - len(uncovered_methods)) / max(len(methods), 1),
                "recommendations": [f"Add tests for method: {m}" for m in uncovered_methods]
            }
        }
    
    async def _identify_smells(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """识别测试代码坏味道"""
        test_code = payload.get("test_code", "")
        
        smells = []
        
        if "Thread.sleep" in test_code:
            smells.append({
                "type": "sleep_in_test",
                "severity": "medium",
                "description": "使用Thread.sleep可能导致测试不稳定",
                "suggestion": "考虑使用Mockito的timeout或Awaitility"
            })
        
        if test_code.count("@Test") > 20:
            smells.append({
                "type": "too_many_tests",
                "severity": "low",
                "description": "测试类中测试方法过多",
                "suggestion": "考虑拆分测试类"
            })
        
        if "System.out" in test_code or "System.err" in test_code:
            smells.append({
                "type": "console_output",
                "severity": "low",
                "description": "测试中有控制台输出",
                "suggestion": "移除或使用日志框架"
            })
        
        lines = test_code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('// TODO') or line.strip().startswith('// FIXME'):
                smells.append({
                    "type": "todo_in_test",
                    "severity": "medium",
                    "description": f"测试中有未完成的TODO: {line.strip()}",
                    "line": i + 1,
                    "suggestion": "完成或移除TODO注释"
                })
        
        return {
            "success": True,
            "output": {
                "smells": smells,
                "smells_count": len(smells)
            }
        }
    
    def _check_naming_conventions(self, test_code: str, class_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查命名规范"""
        issues = []
        
        class_name = class_info.get("name", "")
        test_class_name = f"{class_name}Test"
        
        if test_class_name not in test_code:
            issues.append({
                "type": "naming",
                "severity": "high",
                "description": f"测试类名不符合规范，应为: {test_class_name}",
                "category": "class_naming"
            })
        
        import re
        test_methods = re.findall(r'void\s+(\w+)\s*\(', test_code)
        
        for method in test_methods:
            if not (method.startswith('test') or method.startswith('should') or method.startswith('verify')):
                issues.append({
                    "type": "naming",
                    "severity": "medium",
                    "description": f"测试方法名不符合规范: {method}",
                    "category": "test_method_prefix"
                })
        
        return issues
    
    def _check_structure(self, test_code: str) -> List[Dict[str, Any]]:
        """检查测试结构"""
        issues = []
        
        if "@BeforeEach" not in test_code and "@BeforeAll" not in test_code:
            pass
        
        import re
        test_methods = re.findall(r'void\s+(\w+)\s*\([^)]*\)\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', test_code, re.DOTALL)
        
        for method_name, body in test_methods:
            assertions = body.count("assert") + body.count("verify")
            if assertions == 0:
                issues.append({
                    "type": "structure",
                    "severity": "high",
                    "description": f"测试方法 {method_name} 没有断言",
                    "category": "missing_assertions"
                })
            
            if body.count("assert") > 5:
                issues.append({
                    "type": "structure",
                    "severity": "low",
                    "description": f"测试方法 {method_name} 断言过多，考虑拆分",
                    "category": "too_many_assertions"
                })
        
        return issues
    
    def _check_assertions(self, test_code: str) -> List[Dict[str, Any]]:
        """检查断言有效性"""
        issues = []
        
        if "assertTrue(true)" in test_code:
            issues.append({
                "type": "assertion",
                "severity": "medium",
                "description": "发现无意义的断言: assertTrue(true)",
                "category": "meaningless_assertion"
            })
        
        import re
        empty_assertions = re.findall(r'assert\w+\(\s*\)', test_code)
        for assertion in empty_assertions:
            issues.append({
                "type": "assertion",
                "severity": "high",
                "description": f"发现空断言: {assertion}",
                "category": "empty_assertion"
            })
        
        return issues
    
    def _check_coverage_completeness(
        self,
        test_code: str,
        test_design: Dict[str, Any],
        class_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """检查覆盖完整性"""
        issues = []
        
        scenarios = test_design.get("test_scenarios", [])
        test_types = set(s.get("type", "positive") for s in scenarios)
        
        if "positive" not in test_types:
            issues.append({
                "type": "coverage",
                "severity": "high",
                "description": "缺少正向测试用例",
                "category": "happy_path"
            })
        
        if "negative" not in test_types:
            issues.append({
                "type": "coverage",
                "severity": "medium",
                "description": "缺少负向测试用例",
                "category": "error_path"
            })
        
        if "boundary" not in test_types:
            issues.append({
                "type": "coverage",
                "severity": "low",
                "description": "缺少边界测试用例",
                "category": "edge_cases"
            })
        
        return issues
    
    def _generate_suggestion(self, issue: Dict[str, Any]) -> str:
        """生成改进建议"""
        suggestions_map = {
            "class_naming": "将测试类重命名为符合规范的名称",
            "test_method_prefix": "使用test/should/verify前缀命名测试方法",
            "missing_assertions": "添加有意义的断言来验证测试结果",
            "too_many_assertions": "将测试拆分为多个独立的测试方法",
            "meaningless_assertion": "移除无意义的断言，添加实际验证",
            "empty_assertion": "补充断言条件",
            "happy_path": "添加验证正常执行路径的测试用例",
            "error_path": "添加验证异常处理路径的测试用例",
            "edge_cases": "添加边界值测试用例",
        }
        
        category = issue.get("category", "")
        return suggestions_map.get(category, f"修复问题: {issue.get('description', '')}")
    
    def _calculate_quality_score(self, issues: List[Dict[str, Any]]) -> float:
        """计算质量分数"""
        if not issues:
            return 1.0
        
        severity_weights = {
            "high": 0.3,
            "medium": 0.15,
            "low": 0.05
        }
        
        total_deduction = sum(
            severity_weights.get(issue.get("severity", "low"), 0.05)
            for issue in issues
        )
        
        return max(0.0, 1.0 - total_deduction)
    
    def _generate_review_summary(self, issues: List[Dict[str, Any]], quality_score: float) -> str:
        """生成审查摘要"""
        high_issues = sum(1 for i in issues if i.get("severity") == "high")
        medium_issues = sum(1 for i in issues if i.get("severity") == "medium")
        low_issues = sum(1 for i in issues if i.get("severity") == "low")
        
        status = "通过" if quality_score >= 0.7 else "需要改进"
        
        return (
            f"测试审查结果: {status} (质量分数: {quality_score:.2f})\n"
            f"发现问题: 高={high_issues}, 中={medium_issues}, 低={low_issues}"
        )
    
    def _is_method_tested(self, method_name: str, test_code: str) -> bool:
        """检查方法是否被测试"""
        import re
        test_methods = re.findall(r'void\s+(\w+)\s*\(', test_code)
        
        for test_method in test_methods:
            if method_name.lower() in test_method.lower():
                return True
        
        return method_name.lower() in test_code.lower()
