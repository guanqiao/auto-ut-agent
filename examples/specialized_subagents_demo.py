"""专业化 Subagent 演示脚本

演示 BashSubagent、PlanSubagent、ExploreSubagent 和 TestGenSubagent 的使用。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from typing import Dict, Any
from unittest.mock import Mock

from pyutagent.agent.subagents import (
    SpecializedSubagent,
    SubagentResult,
    BashSubagent,
    PlanSubagent,
    ExploreSubagent,
    TestGenSubagent,
    SubagentRouter,
    create_default_router,
)


class MockLLMClient:
    """模拟 LLM 客户端"""
    
    async def generate(self, prompt: str) -> str:
        """根据提示生成响应"""
        import json
        
        if "shell 命令" in prompt or "bash" in prompt.lower():
            return json.dumps({
                "commands": ["mvn clean test"],
                "description": "清理并运行测试",
                "expected_output": "Tests run successfully",
                "working_directory": ".",
                "timeout_seconds": 300,
                "risk_level": "low"
            })
        elif "实现方案" in prompt or "plan" in prompt.lower():
            return json.dumps({
                "overview": "重构 UserService 以支持手机号",
                "steps": [
                    {
                        "order": 1,
                        "title": "修改 User 实体",
                        "description": "添加 phone 字段",
                        "estimated_hours": 1,
                        "dependencies": []
                    },
                    {
                        "order": 2,
                        "title": "更新 UserService",
                        "description": "修改 createUser 方法",
                        "estimated_hours": 2,
                        "dependencies": [1]
                    }
                ],
                "files_involved": ["User.java", "UserService.java"],
                "modules_involved": ["model", "service"],
                "risks": [
                    {
                        "description": "可能影响现有调用",
                        "mitigation": "检查所有调用点",
                        "severity": "medium"
                    }
                ],
                "validation_methods": ["运行单元测试", "集成测试"],
                "alternatives": []
            })
        elif "探索" in prompt or "explore" in prompt.lower():
            return json.dumps({
                "summary": "这是一个 Maven 项目",
                "key_findings": ["标准 Maven 结构", "使用 Spring Boot"],
                "file_structure": {
                    "total_files": 50,
                    "main_directories": ["src/main/java", "src/test/java"]
                },
                "important_files": [
                    {
                        "path": "src/main/java/UserService.java",
                        "relevance": "high",
                        "description": "用户服务核心类"
                    }
                ],
                "dependencies": ["Spring Boot", "JUnit"],
                "patterns_identified": ["Service 模式", "Repository 模式"],
                "recommendations": ["添加更多单元测试"]
            })
        elif "单元测试" in prompt or "test" in prompt.lower():
            return json.dumps({
                "test_class_name": "UserServiceTest",
                "package": "com.example.service",
                "imports": [
                    "org.junit.jupiter.api.Test",
                    "org.mockito.InjectMocks",
                    "org.mockito.Mock"
                ],
                "test_methods": [
                    {
                        "name": "testCreateUser_Success",
                        "description": "测试成功创建用户",
                        "body": "void testCreateUser_Success() { ... }",
                        "test_type": "positive"
                    },
                    {
                        "name": "testCreateUser_EmptyEmail",
                        "description": "测试空邮箱异常",
                        "body": "void testCreateUser_EmptyEmail() { ... }",
                        "test_type": "negative"
                    }
                ],
                "mocks_needed": ["UserRepository"],
                "test_data": ["有效用户信息", "无效用户信息"],
                "coverage_analysis": {
                    "expected_line_coverage": 0.85,
                    "expected_branch_coverage": 0.75,
                    "scenarios_covered": ["正常流程", "异常流程"]
                }
            })
        
        return json.dumps({"result": "success"})


def demo_bash_subagent():
    """演示 BashSubagent"""
    print("\n" + "="*60)
    print("演示 1: BashSubagent - 命令行任务")
    print("="*60)
    
    llm = MockLLMClient()
    tools = Mock()
    
    bash_agent = BashSubagent(llm, tools)
    
    # 测试任务匹配
    tasks = [
        "运行 Maven 测试",
        "构建项目",
        "部署到生产环境",
        "查看代码",
        "分析项目结构"
    ]
    
    print(f"\n任务匹配测试:")
    for task in tasks:
        confidence = bash_agent.can_handle(task)
        status = "✅ 匹配" if confidence > 0.3 else "❌ 不匹配"
        print(f"  {status} [{confidence:.2f}] {task}")
    
    # 显示统计
    stats = bash_agent.get_stats()
    print(f"\nBashSubagent 统计:")
    print(f"  名称: {stats['name']}")
    print(f"  描述: {stats['description']}")
    print(f"  执行次数: {stats['execution_count']}")
    print(f"  成功率: {stats['success_rate']:.1%}")


def demo_plan_subagent():
    """演示 PlanSubagent"""
    print("\n" + "="*60)
    print("演示 2: PlanSubagent - 方案设计")
    print("="*60)
    
    llm = MockLLMClient()
    tools = Mock()
    
    plan_agent = PlanSubagent(llm, tools)
    
    # 测试任务匹配
    tasks = [
        "设计用户模块重构方案",
        "如何优化数据库查询",
        "实现订单处理流程",
        "运行测试",
        "查找文件"
    ]
    
    print(f"\n任务匹配测试:")
    for task in tasks:
        confidence = plan_agent.can_handle(task)
        status = "✅ 匹配" if confidence > 0.3 else "❌ 不匹配"
        print(f"  {status} [{confidence:.2f}] {task}")
    
    # 显示统计
    stats = plan_agent.get_stats()
    print(f"\nPlanSubagent 统计:")
    print(f"  名称: {stats['name']}")
    print(f"  描述: {stats['description']}")


def demo_explore_subagent():
    """演示 ExploreSubagent"""
    print("\n" + "="*60)
    print("演示 3: ExploreSubagent - 代码库探索")
    print("="*60)
    
    llm = MockLLMClient()
    tools = Mock()
    
    explore_agent = ExploreSubagent(llm, tools)
    
    # 测试任务匹配
    tasks = [
        "查找所有 UserService 的引用",
        "分析项目结构",
        "了解订单模块的依赖关系",
        "构建项目",
        "设计 API"
    ]
    
    print(f"\n任务匹配测试:")
    for task in tasks:
        confidence = explore_agent.can_handle(task)
        status = "✅ 匹配" if confidence > 0.3 else "❌ 不匹配"
        print(f"  {status} [{confidence:.2f}] {task}")
    
    # 显示统计
    stats = explore_agent.get_stats()
    print(f"\nExploreSubagent 统计:")
    print(f"  名称: {stats['name']}")
    print(f"  描述: {stats['description']}")


def demo_testgen_subagent():
    """演示 TestGenSubagent"""
    print("\n" + "="*60)
    print("演示 4: TestGenSubagent - 测试生成")
    print("="*60)
    
    llm = MockLLMClient()
    tools = Mock()
    
    testgen_agent = TestGenSubagent(llm, tools)
    
    # 测试任务匹配
    tasks = [
        "为 UserService 生成单元测试",
        "提高代码覆盖率",
        "创建测试用例",
        "重构代码",
        "分析依赖"
    ]
    
    print(f"\n任务匹配测试:")
    for task in tasks:
        confidence = testgen_agent.can_handle(task)
        status = "✅ 匹配" if confidence > 0.3 else "❌ 不匹配"
        print(f"  {status} [{confidence:.2f}] {task}")
    
    # 显示统计
    stats = testgen_agent.get_stats()
    print(f"\nTestGenSubagent 统计:")
    print(f"  名称: {stats['name']}")
    print(f"  描述: {stats['description']}")


async def demo_subagent_router():
    """演示 SubagentRouter"""
    print("\n" + "="*60)
    print("演示 5: SubagentRouter - 子代理路由器")
    print("="*60)
    
    llm = MockLLMClient()
    tools = Mock()
    
    # 创建路由器
    router = SubagentRouter()
    
    # 注册子代理
    router.register(BashSubagent(llm, tools))
    router.register(PlanSubagent(llm, tools))
    router.register(ExploreSubagent(llm, tools))
    router.register(TestGenSubagent(llm, tools))
    
    print(f"\n已注册 {len(router.subagents)} 个子代理:")
    capabilities = router.get_capabilities()
    for name, description in capabilities.items():
        print(f"  - {name}: {description}")
    
    # 测试路由
    tasks = [
        "运行 Maven 测试",
        "设计用户模块重构方案",
        "查找所有 UserService 的引用",
        "为 OrderService 生成单元测试",
    ]
    
    print(f"\n路由测试:")
    for task in tasks:
        # 评估每个子代理的匹配度
        scores = [
            (subagent.name, subagent.can_handle(task))
            for subagent in router.subagents
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_agent = scores[0][0]
        best_score = scores[0][1]
        
        print(f"\n  任务: {task}")
        print(f"  最佳匹配: {best_agent} ({best_score:.2f})")
        print(f"  所有匹配度:")
        for name, score in scores:
            bar = "█" * int(score * 20)
            print(f"    {name:15} [{bar:<20}] {score:.2f}")


def demo_keywords_matching():
    """演示关键词匹配"""
    print("\n" + "="*60)
    print("演示 6: 关键词匹配机制")
    print("="*60)
    
    print(f"\n各子代理的关键词:")
    
    print(f"\n  BashSubagent ({len(BashSubagent.BASH_KEYWORDS)} 个关键词):")
    print(f"    {', '.join(BashSubagent.BASH_KEYWORDS[:10])}, ...")
    
    print(f"\n  PlanSubagent ({len(PlanSubagent.PLAN_KEYWORDS)} 个关键词):")
    print(f"    {', '.join(PlanSubagent.PLAN_KEYWORDS[:10])}, ...")
    
    print(f"\n  ExploreSubagent ({len(ExploreSubagent.EXPLORE_KEYWORDS)} 个关键词):")
    print(f"    {', '.join(ExploreSubagent.EXPLORE_KEYWORDS[:10])}, ...")
    
    print(f"\n  TestGenSubagent ({len(TestGenSubagent.TEST_KEYWORDS)} 个关键词):")
    print(f"    {', '.join(TestGenSubagent.TEST_KEYWORDS[:10])}, ...")


def demo_create_default_router():
    """演示创建默认路由器"""
    print("\n" + "="*60)
    print("演示 7: 创建默认路由器")
    print("="*60)
    
    llm = MockLLMClient()
    tools = Mock()
    
    router = create_default_router(llm, tools)
    
    print(f"\n默认路由器已创建，包含 {len(router.subagents)} 个子代理:")
    for agent in router.subagents:
        print(f"  ✓ {agent.name}")
    
    # 显示统计
    stats = router.get_stats()
    print(f"\n路由器统计:")
    print(f"  注册子代理数: {stats['registered_agents']}")
    print(f"  路由历史记录: {stats['routing_history_count']}")


def demo_integration_with_universal_planner():
    """演示与 UniversalTaskPlanner 集成"""
    print("\n" + "="*60)
    print("演示 8: 与 UniversalTaskPlanner 集成")
    print("="*60)
    
    print("""
与 UniversalTaskPlanner 集成示例:

```python
from pyutagent.agent import UniversalTaskPlanner, TaskType
from pyutagent.agent.subagents import create_default_router

# 创建子代理路由器
router = create_default_router(llm_client, tool_registry)

# 注册到任务规划器
planner = UniversalTaskPlanner(llm_client, analyzer, tool_registry)

# 为每种任务类型注册处理器
async def handle_bash_task(subtask):
    result = await router.route(subtask.description, context)
    return result

async def handle_plan_task(subtask):
    result = await router.route(subtask.description, context)
    return result

planner.register_task_handler(TaskType.EXECUTE, handle_bash_task)
planner.register_task_handler(TaskType.PLAN, handle_plan_task)

# 执行任务
result = await planner.execute_with_feedback(plan, context)
```

优势:
1. 自动任务路由 - 根据任务内容自动选择最合适的子代理
2. 专业化处理 - 每个子代理专注特定领域
3. 统计和监控 - 跟踪每个子代理的性能
4. 可扩展 - 轻松添加新的子代理
""")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("专业化 Subagent 演示")
    print("参考 Claude Code 的 Subagents 设计")
    print("="*60)
    
    demo_bash_subagent()
    demo_plan_subagent()
    demo_explore_subagent()
    demo_testgen_subagent()
    await demo_subagent_router()
    demo_keywords_matching()
    demo_create_default_router()
    demo_integration_with_universal_planner()
    
    print("\n" + "="*60)
    print("专业化 Subagent 演示完成!")
    print("="*60)
    print("""
核心组件:
1. SpecializedSubagent - 子代理基类
2. BashSubagent - 命令行任务 (mvn, gradle, git, docker)
3. PlanSubagent - 方案设计 (架构、重构、实现)
4. ExploreSubagent - 代码库探索 (查找、分析、理解)
5. TestGenSubagent - 测试生成 (单元测试、覆盖率)
6. SubagentRouter - 子代理路由器

特点:
- 关键词匹配机制
- 置信度评分 (0-1)
- 自动任务路由
- 执行统计和监控
- 易于扩展
""")


if __name__ == "__main__":
    asyncio.run(main())
