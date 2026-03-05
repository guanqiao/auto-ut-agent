"""Test Fix Agent - 测试修复专家

负责修复失败的测试和编译错误。
"""

import asyncio
import logging
import re
from typing import Dict, Any, Set, Optional, List

from ..multi_agent.specialized_agent import (
    SpecializedAgent,
    AgentCapability,
    AgentTask,
)
from ..multi_agent.message_bus import MessageBus
from ..multi_agent.shared_knowledge import SharedKnowledgeBase, ExperienceReplay

logger = logging.getLogger(__name__)


class TestFixAgent(SpecializedAgent):
    """测试修复专家
    
    职责：
    - 分析测试失败原因
    - 修复编译错误
    - 修复测试断言失败
    - 修复Mock配置问题
    - 学习常见错误模式
    """
    
    COMMON_FIX_PATTERNS = {
        "null_pointer": {
            "symptoms": ["NullPointerException", "null"],
            "fixes": ["添加null检查", "初始化对象", "配置Mock返回值"]
        },
        "assertion_failed": {
            "symptoms": ["AssertionFailedError", "expected:", "but was:"],
            "fixes": ["修正预期值", "修正实际值", "更新断言逻辑"]
        },
        "mock_exception": {
            "symptoms": ["Unfinished stubbing", "WrongTypeOfReturnValue"],
            "fixes": ["完善Mock配置", "修正返回类型", "添加when().thenReturn()"]
        },
        "compilation_error": {
            "symptoms": ["cannot find symbol", "incompatible types", "method cannot be applied"],
            "fixes": ["添加导入语句", "修正类型转换", "修正方法签名"]
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
                AgentCapability.TEST_FIX,
                AgentCapability.ERROR_ANALYSIS,
            },
            message_bus=message_bus,
            knowledge_base=knowledge_base,
            experience_replay=experience_replay
        )
        self.llm_client = llm_client
        self.fix_history: list = []
        self.learned_patterns: Dict[str, List[Dict[str, Any]]] = {}
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """执行测试修复任务"""
        task_type = task.task_type
        payload = task.payload
        
        if task_type == "fix_test":
            return await self._fix_test(payload)
        elif task_type == "fix_compilation":
            return await self._fix_compilation(payload)
        elif task_type == "analyze_failure":
            return await self._analyze_failure(payload)
        elif task_type == "apply_pattern":
            return await self._apply_learned_pattern(payload)
        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }
    
    async def _fix_test(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """修复测试
        
        Args:
            payload: 包含 test_code, failure_info, class_info 等
            
        Returns:
            修复后的测试代码
        """
        test_code = payload.get("test_code", "")
        failure_info = payload.get("failure_info", {})
        class_info = payload.get("class_info", {})
        
        logger.info(f"[TestFixAgent] Fixing test for: {class_info.get('name', 'Unknown')}")
        
        error_type = self._classify_error(failure_info)
        
        fix_strategy = self._determine_fix_strategy(error_type, failure_info)
        
        fixed_code = await self._apply_fix(test_code, fix_strategy, failure_info)
        
        fix_result = {
            "original_error": failure_info.get("message", ""),
            "error_type": error_type,
            "fix_strategy": fix_strategy,
            "fixed_code": fixed_code,
            "changes_made": self._extract_changes(test_code, fixed_code),
        }
        
        self.fix_history.append(fix_result)
        
        self._learn_from_fix(error_type, failure_info, fix_strategy)
        
        self.share_knowledge(
            item_type="test_fix",
            content=fix_result,
            confidence=0.8,
            tags=["test_fix", error_type, class_info.get("name", "unknown")]
        )
        
        return {
            "success": True,
            "output": fix_result,
            "metadata": {
                "error_type": error_type,
                "fixes_applied": len(fix_result.get("changes_made", []))
            }
        }
    
    async def _fix_compilation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """修复编译错误"""
        test_code = payload.get("test_code", "")
        compilation_errors = payload.get("compilation_errors", [])
        class_info = payload.get("class_info", {})
        
        logger.info(f"[TestFixAgent] Fixing compilation errors for: {class_info.get('name', 'Unknown')}")
        
        fixed_code = test_code
        fixes_applied = []
        
        for error in compilation_errors:
            error_message = error.get("message", "")
            line_number = error.get("line", 0)
            
            if "cannot find symbol" in error_message:
                symbol = self._extract_symbol(error_message)
                import_fix = self._find_import_for_symbol(symbol, class_info)
                if import_fix:
                    fixed_code = self._add_import(fixed_code, import_fix)
                    fixes_applied.append({
                        "type": "add_import",
                        "symbol": symbol,
                        "import": import_fix
                    })
            
            elif "incompatible types" in error_message:
                type_fix = self._fix_type_mismatch(fixed_code, line_number, error_message)
                if type_fix:
                    fixed_code = type_fix["code"]
                    fixes_applied.append({
                        "type": "type_conversion",
                        "line": line_number,
                        "details": type_fix["details"]
                    })
            
            elif "method cannot be applied" in error_message:
                method_fix = self._fix_method_signature(fixed_code, line_number, error_message, class_info)
                if method_fix:
                    fixed_code = method_fix["code"]
                    fixes_applied.append({
                        "type": "method_signature",
                        "line": line_number,
                        "details": method_fix["details"]
                    })
        
        return {
            "success": True,
            "output": {
                "fixed_code": fixed_code,
                "fixes_applied": fixes_applied,
                "errors_fixed": len(fixes_applied)
            }
        }
    
    async def _analyze_failure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """分析失败原因"""
        failure_info = payload.get("failure_info", {})
        test_code = payload.get("test_code", "")
        
        analysis = {
            "error_type": self._classify_error(failure_info),
            "root_cause": self._identify_root_cause(failure_info, test_code),
            "affected_areas": self._identify_affected_areas(failure_info, test_code),
            "suggested_fixes": [],
            "confidence": 0.0
        }
        
        error_type = analysis["error_type"]
        if error_type in self.COMMON_FIX_PATTERNS:
            analysis["suggested_fixes"] = self.COMMON_FIX_PATTERNS[error_type]["fixes"]
            analysis["confidence"] = 0.8
        
        if error_type in self.learned_patterns:
            learned_fixes = [p["fix"] for p in self.learned_patterns[error_type]]
            analysis["suggested_fixes"].extend(learned_fixes)
            analysis["confidence"] = max(analysis["confidence"], 0.9)
        
        return {
            "success": True,
            "output": analysis
        }
    
    async def _apply_learned_pattern(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """应用学习到的模式"""
        pattern_id = payload.get("pattern_id", "")
        test_code = payload.get("test_code", "")
        
        for error_type, patterns in self.learned_patterns.items():
            for pattern in patterns:
                if pattern.get("id") == pattern_id:
                    fixed_code = self._apply_pattern_to_code(test_code, pattern)
                    return {
                        "success": True,
                        "output": {
                            "fixed_code": fixed_code,
                            "pattern_applied": pattern
                        }
                    }
        
        return {
            "success": False,
            "error": f"Pattern not found: {pattern_id}"
        }
    
    def _classify_error(self, failure_info: Dict[str, Any]) -> str:
        """分类错误类型"""
        message = failure_info.get("message", "").lower()
        exception_type = failure_info.get("exception_type", "").lower()
        
        if "nullpointer" in exception_type or "null" in message:
            return "null_pointer"
        elif "assertion" in exception_type or "expected" in message:
            return "assertion_failed"
        elif "mock" in message or "stubbing" in message:
            return "mock_exception"
        elif "compilation" in message or "cannot find symbol" in message:
            return "compilation_error"
        else:
            return "unknown"
    
    def _determine_fix_strategy(self, error_type: str, failure_info: Dict[str, Any]) -> Dict[str, Any]:
        """确定修复策略"""
        strategy = {
            "type": error_type,
            "actions": [],
            "priority": "medium"
        }
        
        if error_type == "null_pointer":
            strategy["actions"] = [
                "检查对象初始化",
                "添加null检查",
                "配置Mock返回非null值"
            ]
            strategy["priority"] = "high"
        
        elif error_type == "assertion_failed":
            expected = failure_info.get("expected")
            actual = failure_info.get("actual")
            strategy["actions"] = [
                f"检查预期值: {expected}",
                f"检查实际值: {actual}",
                "修正断言或修正实现"
            ]
            strategy["priority"] = "medium"
        
        elif error_type == "mock_exception":
            strategy["actions"] = [
                "检查Mock配置完整性",
                "确保when().thenReturn()成对出现",
                "验证返回类型匹配"
            ]
            strategy["priority"] = "high"
        
        elif error_type == "compilation_error":
            strategy["actions"] = [
                "检查导入语句",
                "验证类型匹配",
                "检查方法签名"
            ]
            strategy["priority"] = "high"
        
        return strategy
    
    async def _apply_fix(
        self,
        test_code: str,
        fix_strategy: Dict[str, Any],
        failure_info: Dict[str, Any]
    ) -> str:
        """应用修复"""
        error_type = fix_strategy.get("type", "")
        
        if error_type == "null_pointer":
            return self._fix_null_pointer(test_code, failure_info)
        elif error_type == "assertion_failed":
            return self._fix_assertion(test_code, failure_info)
        elif error_type == "mock_exception":
            return self._fix_mock(test_code, failure_info)
        else:
            return test_code
    
    def _fix_null_pointer(self, test_code: str, failure_info: Dict[str, Any]) -> str:
        """修复空指针异常"""
        lines = test_code.split('\n')
        stack_trace = failure_info.get("stack_trace", "")
        
        for i, line in enumerate(lines):
            if "@Mock" in line and i + 1 < len(lines):
                next_line = lines[i + 1]
                if "private" in next_line and ";" in next_line:
                    var_match = re.search(r'private\s+\w+\s+(\w+);', next_line)
                    if var_match:
                        var_name = var_match.group(1)
                        if var_name in stack_trace:
                            lines.insert(i + 2, f"        // Ensure {var_name} is properly mocked")
        
        return '\n'.join(lines)
    
    def _fix_assertion(self, test_code: str, failure_info: Dict[str, Any]) -> str:
        """修复断言失败"""
        expected = failure_info.get("expected", "")
        actual = failure_info.get("actual", "")
        
        if expected and actual:
            pattern = rf'assertEquals\([^,]+,\s*[^)]+\)'
            
            def replace_assertion(match):
                return f'assertEquals({expected}, {actual})'
            
            test_code = re.sub(pattern, replace_assertion, test_code, count=1)
        
        return test_code
    
    def _fix_mock(self, test_code: str, failure_info: Dict[str, Any]) -> str:
        """修复Mock问题"""
        message = failure_info.get("message", "")
        
        if "Unfinished stubbing" in message:
            lines = test_code.split('\n')
            for i, line in enumerate(lines):
                if "when(" in line and "thenReturn" not in line:
                    if i + 1 < len(lines):
                        lines[i] = line.rstrip() + ").thenReturn(/* value */);"
            test_code = '\n'.join(lines)
        
        return test_code
    
    def _extract_changes(self, original: str, fixed: str) -> List[Dict[str, Any]]:
        """提取变更"""
        changes = []
        original_lines = original.split('\n')
        fixed_lines = fixed.split('\n')
        
        max_lines = max(len(original_lines), len(fixed_lines))
        
        for i in range(max_lines):
            orig_line = original_lines[i] if i < len(original_lines) else ""
            fixed_line = fixed_lines[i] if i < len(fixed_lines) else ""
            
            if orig_line != fixed_line:
                changes.append({
                    "line": i + 1,
                    "original": orig_line,
                    "fixed": fixed_line
                })
        
        return changes
    
    def _learn_from_fix(self, error_type: str, failure_info: Dict[str, Any], fix_strategy: Dict[str, Any]):
        """从修复中学习"""
        if error_type not in self.learned_patterns:
            self.learned_patterns[error_type] = []
        
        pattern = {
            "id": f"{error_type}_{len(self.learned_patterns[error_type])}",
            "symptoms": failure_info.get("message", "")[:100],
            "fix": fix_strategy.get("actions", []),
            "success_rate": 1.0
        }
        
        self.learned_patterns[error_type].append(pattern)
        
        logger.info(f"[TestFixAgent] Learned new pattern for {error_type}")
    
    def _extract_symbol(self, error_message: str) -> str:
        """从错误消息中提取符号"""
        match = re.search(r'symbol:\s*(?:class|method|variable)\s+(\w+)', error_message)
        return match.group(1) if match else ""
    
    def _find_import_for_symbol(self, symbol: str, class_info: Dict[str, Any]) -> Optional[str]:
        """为符号查找导入"""
        common_imports = {
            "List": "java.util.List",
            "ArrayList": "java.util.ArrayList",
            "Map": "java.util.Map",
            "HashMap": "java.util.HashMap",
            "Set": "java.util.Set",
            "Optional": "java.util.Optional",
            "Assert": "org.junit.Assert",
            "Mockito": "org.mockito.Mockito",
        }
        
        if symbol in common_imports:
            return common_imports[symbol]
        
        dependencies = class_info.get("dependencies", [])
        for dep in dependencies:
            if symbol in dep.get("type", ""):
                return dep.get("type", "")
        
        return None
    
    def _add_import(self, code: str, import_statement: str) -> str:
        """添加导入语句"""
        if import_statement in code:
            return code
        
        lines = code.split('\n')
        insert_pos = 0
        
        for i, line in enumerate(lines):
            if line.startswith('import '):
                insert_pos = i + 1
            elif line.startswith('package '):
                insert_pos = i + 1
        
        lines.insert(insert_pos, f"import {import_statement};")
        return '\n'.join(lines)
    
    def _fix_type_mismatch(self, code: str, line_number: int, error_message: str) -> Optional[Dict[str, Any]]:
        """修复类型不匹配"""
        return None
    
    def _fix_method_signature(self, code: str, line_number: int, error_message: str, class_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """修复方法签名"""
        return None
    
    def _identify_root_cause(self, failure_info: Dict[str, Any], test_code: str) -> str:
        """识别根本原因"""
        error_type = self._classify_error(failure_info)
        message = failure_info.get("message", "")
        
        root_causes = {
            "null_pointer": "对象未正确初始化或Mock配置不完整",
            "assertion_failed": "预期结果与实际结果不匹配",
            "mock_exception": "Mock配置存在问题",
            "compilation_error": "代码存在语法或类型错误",
        }
        
        return root_causes.get(error_type, f"未知错误: {message[:100]}")
    
    def _identify_affected_areas(self, failure_info: Dict[str, Any], test_code: str) -> List[str]:
        """识别受影响的区域"""
        areas = []
        stack_trace = failure_info.get("stack_trace", "")
        
        if "setUp" in stack_trace:
            areas.append("测试初始化")
        if "tearDown" in stack_trace:
            areas.append("测试清理")
        
        test_methods = re.findall(r'void\s+(\w+)\s*\(', test_code)
        for method in test_methods:
            if method in stack_trace:
                areas.append(f"测试方法: {method}")
        
        return areas if areas else ["未知区域"]
    
    def _apply_pattern_to_code(self, code: str, pattern: Dict[str, Any]) -> str:
        """将模式应用到代码"""
        return code
