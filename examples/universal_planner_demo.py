"""Universal Task Planner 演示示例

演示如何使用 UniversalTaskPlanner 处理不同类型的编程任务。
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
        """根据 prompt 生成响应"""
        # 模拟不同任务的响应
        if "test_generation" in prompt or "测试" in prompt:
            return json.dumps({
                "task_type": "test_generation",
                "description": "为 UserService 生成单元测试",
                "target_files": ["UserService.java"],
                "constraints": ["使用 JUnit 5", "Mockito"],
                "success_criteria": ["测试覆盖率 > 80%", "所有测试通过"],
                "estimated_complexity": 3,
                "context_requirements": ["UserService 源码", "依赖类列表"]
            })
        elif "refactor" in prompt or "重构" in prompt:
            return json.dumps({
                "task_type": "code_refactoring",
                "description": "重构 OrderService 提取接口",
                "target_files": ["OrderService.java"],
                "constraints": ["保持向后兼容"],
                "success_criteria": ["所有测试通过", "代码更简洁"],
                "estimated_complexity": 4,
                "context_requirements": []
            })
        elif "bug_fix" in prompt or "修复" in prompt:
            return json.dumps({
                "task_type": "bug_fix",
                "description": "修复空指针异常",
                "target_files": ["PaymentProcessor.java"],
                "constraints": [],
                "success_criteria": ["Bug 不再复现", "回归测试通过"],
                "estimated_complexity": 2,
                "context_requirements": []
            })
        else:
            return json.dumps({
                "task_type": "query",
                "description": "查询任务",
                "target_files": [],
                "constraints": [],
                "success_criteria": ["完成任务"],
                "estimated_complexity": 1,
                "context_requirements": []
            })


class MockTaskHandler(TaskHandler):
    """模拟任务处理器"""
    
    def __init__(self, name: str, should_succeed: bool = True):
        self.name = name
        self.should_succeed = should_succeed
        self.executed_tasks = []
    
    async def handle(self, subtask: Subtask, context: Dict[str, Any]) -> SubtaskResult:
        """处理子任务"""
        print(f"  [{self.name}] 执行: {subtask.description}")
        self.executed_tasks.append(subtask.id)
        
        await asyncio.sleep(0.1)  # 模拟执行时间
        
        return SubtaskResult(
            subtask_id=subtask.id,
            success=self.should_succeed,
            data={"handler": self.name, "task": subtask.id},
            error=None if self.should_succeed else "模拟错误"
        )
    
    def can_handle(self, task_type: TaskType) -> bool:
        return True


async def demo_task_understanding():
    """演示任务理解功能"""
    print("=" * 60)
    print("演示 1: 任务理解 (Task Understanding)")
    print("=" * 60)
    
    llm = MockLLMClient()
    analyzer = Mock()
    tool_registry = Mock()
    
    planner = UniversalTaskPlanner(llm, analyzer, tool_registry)
    
    project_context = {
        "language": "java",
        "build_tool": "maven",
        "structure": {"src": "src/main/java"}
    }
    
    # 测试不同任务的识别
    requests = [
        "为 UserService 生成单元测试",
        "重构 OrderService 提取接口",
        "修复 PaymentProcessor 的空指针异常"
    ]
    
    for request in requests:
        print(f"\n用户请求: {request}")
        understanding = await planner.understand_task(request, project_context)
        print(f"  任务类型: {understanding.task_type.value}")
        print(f"  描述: {understanding.description}")
        print(f"  目标文件: {understanding.target_files}")
        print(f"  复杂度: {understanding.estimated_complexity}/5")


async def demo_task_decomposition():
    """演示任务分解功能"""
    print("\n" + "=" * 60)
    print("演示 2: 任务分解 (Task Decomposition)")
    print("=" * 60)
    
    llm = MockLLMClient()
    analyzer = Mock()
    tool_registry = Mock()
    
    planner = UniversalTaskPlanner(llm, analyzer, tool_registry)
    
    # 测试生成任务分解
    understanding = TaskUnderstanding(
        task_type=TaskType.TEST_GENERATION,
        description="为 UserService 生成单元测试",
        target_files=["UserService.java"],
        estimated_complexity=3
    )
    
    print(f"\n任务: {understanding.description}")
    plan = await planner.decompose_task(understanding, {})
    
    print(f"\n生成的执行计划 (共 {len(plan.subtasks)} 个子任务):")
    for i, subtask in enumerate(plan.subtasks, 1):
        deps = f" (依赖: {', '.join(subtask.dependencies)})" if subtask.dependencies else ""
        print(f"  {