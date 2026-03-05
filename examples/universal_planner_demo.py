"""Universal Task Planner 演示脚本

演示如何使用通用任务规划器处理不同类型的编程任务。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                    "task_type": "bug_fix",
                    "dependencies": [],
                    "insert_after": "compile_test"
                }
            })
        return json.dumps({"result": "success"})


class TestGenerationHandler(TaskHandler):
    """测试生成任务处理器"""
    
    async def handle(self, subtask: Subtask, context: Dict[str, Any]) -> SubtaskResult:
        print(f"  📝 执行: {subtask.description}")
        return SubtaskResult(
            subtask_id=subtask.id,
            success=True,
            data={"generated_tests": 5, "coverage": 0.85}
        )
    
    def can_handle(self, task_type: TaskType) -> bool:
        return task_type == TaskType.TEST_GENERATION


class QueryHandler(TaskHandler):
    """查询任务处理器"""
    
    async def handle(self, subtask: Subtask, context: Dict[str, Any]) -> SubtaskResult:
        print(f"  🔍 执行: {subtask.description}")
        return SubtaskResult(
            subtask_id=subtask.id,
            success=True,
            data={"found": True}
        )
    
    def can_handle(self, task_type: TaskType) -> bool:
        return task_type == TaskType.QUERY


class BugFixHandler(TaskHandler):
    """Bug 修复任务处理器"""
    
    async def handle(self, subtask: Subtask, context: Dict[str, Any]) -> SubtaskResult:
        print(f"  🐛 执行: {subtask.description}")
        return SubtaskResult(
            subtask_id=subtask.id,
            success=True,
            data={"fixes_applied": 2}
        )
    
    def can_handle(self, task_type: TaskType) -> bool:
        return task_type == TaskType.BUG_FIX


async def demo_task_understanding():
    """演示任务理解"""
    print("\n" + "="*60)
    print("演示 1: 任务理解")
    print("="*60)
    
    llm = MockLLMClient()
    analyzer = Mock()
    tools = Mock()
    
    planner = UniversalTaskPlanner(llm, analyzer, tools)
    
    user_request = "为 UserService 生成单元测试"
    project_context = {
        "language": "java",
        "build_tool": "maven",
        "structure": {"src": "src/main/java"}
    }
    
    understanding = await planner.understand_task(user_request, project_context)
    
    print(f"\n用户请求: {user_request}")
    print(f"任务类型: {understanding.task_type.value}")
    print(f"描述: {understanding.description}")
    print(f"目标文件: {understanding.target_files}")
    print(f"约束条件: {understanding.constraints}")
    print(f"成功标准: {understanding.success_criteria}")
    print(f"复杂度: {understanding.estimated_complexity}/5")


async def demo_task_decomposition():
    """演示任务分解"""
    print("\n" + "="*60)
    print("演示 2: 任务分解")
    print("="*60)
    
    llm = MockLLMClient()
    analyzer = Mock()
    tools = Mock()
    
    planner = UniversalTaskPlanner(llm, analyzer, tools)
    
    # 测试生成任务
    understanding = TaskUnderstanding(
        task_type=TaskType.TEST_GENERATION,
        description="为 UserService 生成单元测试",
        target_files=["UserService.java"]
    )
    
    plan = await planner.decompose_task(understanding, {})
    
    print(f"\n任务: {plan.original_request}")
    print(f"任务 ID: {plan.task_id}")
    print(f"\n子任务列表:")
    for i, subtask in enumerate(plan.subtasks, 1):
        deps = f" (依赖: {', '.join(subtask.dependencies)})" if subtask.dependencies else ""
        print(f"  {i}. [{subtask.task_type.value}] {subtask.description}{deps}")
    
    print(f"\n执行顺序: {' -> '.join(plan.execution_order)}")


async def demo_execution():
    """演示任务执行"""
    print("\n" + "="*60)
    print("演示 3: 任务执行")
    print("="*60)
    
    llm = MockLLMClient()
    analyzer = Mock()
    tools = Mock()
    
    planner = UniversalTaskPlanner(llm, analyzer, tools)
    
    # 注册任务处理器
    planner.register_task_handler(TaskType.TEST_GENERATION, TestGenerationHandler())
    planner.register_task_handler(TaskType.QUERY, QueryHandler())
    planner.register_task_handler(TaskType.BUG_FIX, BugFixHandler())
    
    # 创建测试生成计划
    understanding = TaskUnderstanding(
        task_type=TaskType.TEST_GENERATION,
        description="为 UserService 生成单元测试",
        target_files=["UserService.java"]
    )
    
    plan = await planner.decompose_task(understanding, {})
    
    print(f"\n执行任务: {plan.original_request}")
    print("-" * 40)
    
    # 执行计划
    result = await planner.execute_with_feedback(
        plan, 
        {},
        progress_callback=lambda subtask, result: print(f"    ✓ 完成: {subtask.id}")
    )
    
    print("-" * 40)
    print(f"执行结果: {'✅ 成功' if result.success else '❌ 失败'}")
    print(f"完成子任务: {len(result.completed_subtasks)}/{len(plan.subtasks)}")
    print(f"执行时间: {result.execution_time:.2f}秒")
    
    # 显示统计
    stats = planner.get_statistics()
    print(f"\n统计信息:")
    print(f"  总执行次数: {stats['total_executions']}")
    print(f"  成功率: {stats['success_rate']*100:.1f}%")


async def demo_different_task_types():
    """演示不同任务类型的分解"""
    print("\n" + "="*60)
    print("演示 4: 不同任务类型的分解策略")
    print("="*60)
    
    llm = MockLLMClient()
    analyzer = Mock()
    tools = Mock()
    
    planner = UniversalTaskPlanner(llm, analyzer, tools)
    
    task_types = [
        (TaskType.CODE_REFACTORING, "重构 OrderService"),
        (TaskType.BUG_FIX, "修复空指针异常"),
        (TaskType.FEATURE_ADD, "添加支付功能"),
        (TaskType.CODE_REVIEW, "审查 UserService"),
    ]
    
    for task_type, description in task_types:
        understanding = TaskUnderstanding(
            task_type=task_type,
            description=description
        )
        
        plan = await planner.decompose_task(understanding, {})
        
        print(f"\n【{task_type.value}】{description}")
        print(f"  子任务数: {len(plan.subtasks)}")
        print(f"  关键步骤: {', '.join([s.id for s in plan.subtasks[:3]])}")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("Universal Task Planner 演示")
    print("通用任务规划器 - 从 UT 生成到通用 Coding Agent")
    print("="*60)
    
    await demo_task_understanding()
    await demo_task_decomposition()
    await demo_execution()
    await demo_different_task_types()
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
