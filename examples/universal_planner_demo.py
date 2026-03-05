"""Universal Task Planner 演示脚本

演示如何使用通用任务规划器处理不同类型的编程任务。
"""

import asyncio
import json
from typing import Dict, Any
from unittest.mock import Mock

from pyutagent.agent.universal_planner import (
    UniversalTaskPlanner,
    TaskType,
    TaskUnderstanding,
    Subtask,
    SubtaskResult,
    TaskHandler
)


class MockLLMClient:
    """模拟 LLM 客户端"""
    
    async def generate(self, prompt: str) -> str:
        """根据提示生成响应"""
        # 模拟 LLM 响应
        if "分析以下用户编程请求" in prompt:
            return json.dumps({
                "task_type": "test_generation",
                "description": "为 UserService 生成单元测试",
                "target_files": ["UserService.java"],
                "constraints": ["使用 JUnit 5", "Mockito"],
                "success_criteria": ["测试编译通过", "覆盖率 > 80%"],
                "estimated_complexity": 3,
                "context_requirements": ["UserService 源码", "依赖分析"]
            })
        elif "子任务执行失败" in prompt:
            return json.dumps({
                "action": "add_subtask",
                "details": {
                    "id": "fix_imports",
                    "description": "修复导入语句",
                    "task_type": "