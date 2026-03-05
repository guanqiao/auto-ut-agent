"""
Claude Code Inspired Features Demo
===================================

展示如何使用参考 Claude Code 设计的新功能：
1. 通用任务规划器 (UniversalTaskPlanner)
2. Hooks 生命周期系统
3. 项目配置系统 (PYUT.md)
4. 专业化 Subagents
5. 智能上下文压缩
"""

import asyncio
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyutagent.agent.universal_planner import (
    UniversalTaskPlanner,
    TaskType,
    TaskUnderstanding,
    ExecutionPlan
)
from pyutagent.core.hooks import (
    HookManager,
    HookType,
    HookContext,
    HookResult
)
from pyutagent.core.project_config import (
    ProjectConfigManager,
    ProjectContext,
    BuildTool
)
from pyutagent.agent.subagents import (
    SubagentRouter,
    BashSubagent,
    PlanSubagent,
    ExploreSubagent,
    TestGenSubagent,
    create_default_router
)
from pyutagent.core.context_compactor import (
    ContextCompactor,
    AutoCompactManager,
    CompactionStrategy
)


# Mock LLM Client for demo
class MockLLMClient:
    """模拟 LLM 客户端"""
    
    async def generate(self, prompt: str) -> str:
        """模拟生成响应"""
        # 根据 prompt 内容返回不同的模拟响应
        if "分析以下用户编程请求" in prompt:
            return '''{
                "task_type": "test_generation",
                "description": "为 UserService 生成单元测试",
                "target_files": ["UserService.java"],
                "constraints": ["使用 JUnit 5", "Mockito"],
                "success_criteria": ["测试覆盖率 > 80%"],
                "estimated_complexity": 3,
                "context_requirements": ["类结构", "依赖关系"]
            }'''
        
        elif "将以下任务转换为可执行的 shell 命令" in prompt:
            return '''{
                "commands": ["mvn test -Dtest=UserServiceTest"],
                "description": "运行 UserService 的单元测试",
                "expected_output": "Tests run: X, Failures: 0",
                "risk_level": "low"
            }'''
        
        elif "为以下任务设计详细的实现方案" in prompt:
            return '''{
                "overview": "为 UserService 添加缓存功能",
                "steps": [
                    {"order": 1, "title": "分析现有代码", "description": "了解 UserService 结构", "estimated_hours": 1},
                    {"order": 2, "title": "设计缓存方案", "description": "选择合适的缓存策略", "estimated_hours": 2}
                ],
                "files_involved": ["UserService.java", "CacheConfig.java"],
                "risks": [{"description": "缓存一致性问题", "mitigation": "使用 TTL", "severity": "medium"}]
            }'''
        
        elif "分析以下代码库探索结果" in prompt:
            return '''{
                "summary": "标准 Maven 项目结构",
                "key_findings": ["使用 Spring Boot", "JUnit 5 测试"],
                "file_structure": {"total_files": 50, "main_directories": ["src", "test"]},
                "important_files": [{"path": "pom.xml", "relevance": "high", "description": "Maven 配置"}]
            }'''
        
        elif "为以下目标生成单元测试" in prompt:
            return '''{
                "test_class_name": "UserServiceTest",
                "package": "com.example.service",
                "imports": ["org.junit.jupiter.api.Test", "org.mockito.Mockito"],
                "test_methods": [
                    {"name": "testGetUserById", "description": "测试根据 ID 获取用户", "test_type": "positive"}
                ],
                "coverage_analysis": {"expected_line_coverage": 0.85}
            }'''
        
        elif "请将以下对话历史压缩为结构化摘要" in prompt:
            return '''{
                "summary": "正在为用户服务生成测试",
                "completed_tasks": ["分析 UserService 结构"],
                "current_focus": "生成测试代码",
                "pending_tasks": ["编译测试", "运行测试"],
                "key_decisions": [{"decision": "使用 Mockito", "rationale": "标准 Mock 框架"}],
                "active_files": ["UserService.java", "UserServiceTest.java"]
            }'''
        
        return '{"result": "success"}'


# Mock Tool Registry for demo
class MockToolRegistry:
    """模拟工具注册表"""
    
    def get(self, tool_name: str):
        return None


async def demo_universal_planner():
    """演示通用任务规划器"""
    print("=" * 60)
    print("Demo 1: Universal Task Planner (通用任务规划器)")
    print("=" * 60)
    
    llm = MockLLMClient()
    analyzer = None
    tools = MockToolRegistry()
    
    planner = UniversalTaskPlanner(llm, analyzer, tools)
    
    # 1. 理解任务
    user_request = "Generate unit tests for UserService"
    project_context = {
        'language': 'java',
        'build_tool': 'maven',
        'structure': {'src': 'src/main/java'}
    }
    
    print(f"\n用户请求: {user_request}")
    understanding = await planner.understand_task(user_request, project_context)
    
    print(f"任务类型: {understanding.task_type.value}")
    print(f"任务描述: {understanding.description}")
    print(f"目标文件: {understanding.target_files}")
    print(f"复杂度: {understanding.estimated_complexity}/5")
    
    # 2. 分解任务
    plan = await planner.decompose_task(understanding, project_context)
    
    print(f"\n执行计划 ID: {plan.task_id}")
    print(f"子任务数量: {len(plan.subtasks)}")
    print("\n子任务列表:")
    for i, subtask in enumerate(plan.subtasks, 1):
        deps = f" (依赖: {', '.join(subtask.dependencies)})" if subtask.dependencies else ""
        print(f"  {i}. [{subtask.task_type.value}] {subtask.description}{deps}")
    
    print("\n执行顺序:")
    print(f"  {' -> '.join(plan.execution_order)}")
    
    return plan


async def demo_hooks_system_async():
    """演示 Hooks 系统（异步版本）"""
    print("\n" + "=" * 60)
    print("Demo 2: Hooks System (生命周期钩子系统)")
    print("=" * 60)
    
    hook_manager = HookManager()
    
    # 注册内置钩子
    hook_manager.register_builtin_hooks()
    print("\n已注册内置钩子:")
    stats = hook_manager.get_stats()
    print(f"  内置钩子已注册: {stats['builtin_hooks_registered']}")
    
    # 注册自定义钩子
    def my_custom_handler(context: HookContext) -> HookResult:
        print(f"  自定义钩子触发: {context.hook_type.name}")
        return HookResult(success=True, data={'custom': 'data'})
    
    hook_manager.register_hook(
        name="my_custom_hook",
        hook_type=HookType.POST_TOOL_USE,
        handler=my_custom_handler,
        priority=5
    )
    print("  已注册自定义钩子: my_custom_hook")
    
    # 触发钩子
    print("\n触发 POST_TOOL_USE 钩子:")
    result = await hook_manager.trigger(
        HookType.POST_TOOL_USE,
        data={'tool_name': 'file_write', 'file_path': 'Test.java'}
    )
    print(f"  钩子执行结果: success={result.success}")
    
    return hook_manager

def demo_hooks_system():
    """演示 Hooks 系统（同步包装）"""
    # 在已有事件循环中运行
    try:
        loop = asyncio.get_running_loop()
        # 如果在事件循环中，使用 ensure_future
        return asyncio.ensure_future(demo_hooks_system_async())
    except RuntimeError:
        # 没有事件循环，创建新的
        return asyncio.run(demo_hooks_system_async())


def demo_project_config():
    """演示项目配置系统"""
    print("\n" + "=" * 60)
    print("Demo 3: Project Configuration (项目配置系统)")
    print("=" * 60)
    
    # 使用当前项目作为示例
    project_root = Path(__file__).parent.parent
    config_manager = ProjectConfigManager(project_root)
    
    # 检查配置是否存在
    if config_manager.config_exists():
        print(f"\n配置文件已存在: {config_manager.get_config_path()}")
    else:
        print("\n配置文件不存在，将自动分析项目结构...")
    
    # 加载项目上下文
    context = config_manager.load_context()
    
    print(f"\n项目信息:")
    print(f"  名称: {context.name}")
    print(f"  语言: {context.language}")
    print(f"  构建工具: {context.build_tool.value}")
    print(f"  Java 版本: {context.java_version}")
    print(f"  测试框架: {context.test_preferences.test_framework.value}")
    print(f"  Mock 框架: {context.test_preferences.mock_framework.value}")
    
    print(f"\n构建命令:")
    print(f"  构建: {context.build_commands.build}")
    print(f"  测试: {context.build_commands.test}")
    print(f"  覆盖率: {context.build_commands.coverage}")
    
    if context.key_modules:
        print(f"\n关键模块:")
        for module in context.key_modules[:5]:
            print(f"  - {module}")
    
    # 获取 Prompt 上下文
    prompt_context = config_manager.get_prompt_context()
    print(f"\nPrompt 上下文预览 (前 200 字符):")
    print(f"  {prompt_context[:200]}...")
    
    return config_manager


async def demo_subagents():
    """演示专业化 Subagents"""
    print("\n" + "=" * 60)
    print("Demo 4: Specialized Subagents (专业化子代理)")
    print("=" * 60)
    
    llm = MockLLMClient()
    tools = MockToolRegistry()
    
    # 创建默认路由器
    router = create_default_router(llm, tools)
    
    print("\n已注册子代理:")
    capabilities = router.get_capabilities()
    for name, desc in capabilities.items():
        print(f"  - {name}: {desc}")
    
    # 测试任务路由
    test_tasks = [
        "Run maven tests",
        "Design a caching solution",
        "Explore the project structure",
        "Generate tests for UserService"
    ]
    
    print("\n任务路由测试:")
    for task in test_tasks:
        # 评估每个代理的匹配度
        scores = [
            (agent.name, agent.can_handle(task))
            for agent in router.subagents
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_agent = scores[0]
        print(f"  任务: '{task[:40]}...' -> {best_agent[0]} (置信度: {best_agent[1]:.2f})")
    
    # 执行一个任务
    print("\n执行测试生成任务:")
    result = await router.route(
        "Generate tests for UserService",
        context={
            'target_class': 'UserService',
            'test_framework': 'junit5',
            'mock_framework': 'mockito'
        }
    )
    print(f"  结果: success={result.success}")
    print(f"  摘要: {result.summary}")
    
    return router


async def demo_context_compaction():
    """演示智能上下文压缩"""
    print("\n" + "=" * 60)
    print("Demo 5: Context Compaction (智能上下文压缩)")
    print("=" * 60)
    
    llm = MockLLMClient()
    
    # 创建自动压缩管理器
    manager = AutoCompactManager(
        llm_client=llm,
        max_tokens=128000,
        threshold=0.85,
        enable_auto_compact=True
    )
    
    print("\n自动压缩管理器配置:")
    print(f"  最大 Token 数: {manager.max_tokens}")
    print(f"  压缩阈值: {manager.compactor.threshold}")
    print(f"  自动压缩: {'启用' if manager.enable_auto_compact else '禁用'}")
    
    # 模拟对话历史
    conversation_history = [
        {'role': 'user', 'content': 'Generate tests for UserService'},
        {'role': 'assistant', 'content': 'Analyzing UserService structure...'},
        {'role': 'assistant', 'content': 'Found 5 methods to test'},
        {'role': 'user', 'content': 'Make sure to cover edge cases'},
        {'role': 'assistant', 'content': 'Generated 10 test cases'},
        {'role': 'assistant', 'content': 'Tests compiled successfully'},
    ]
    
    # 检查是否需要压缩
    current_tokens = 5000  # 模拟当前 token 数
    should_compact, reason = manager.compactor.should_compact(
        current_tokens, manager.max_tokens
    )
    
    print(f"\n当前 Token 数: {current_tokens}")
    print(f"使用率: {current_tokens / manager.max_tokens:.1%}")
    print(f"是否需要压缩: {should_compact} ({reason})")
    
    # 模拟高 token 使用情况
    high_tokens = 110000  # 超过 85% 阈值
    should_compact, reason = manager.compactor.should_compact(
        high_tokens, manager.max_tokens
    )
    
    print(f"\n模拟高负载情况:")
    print(f"  Token 数: {high_tokens}")
    print(f"  使用率: {high_tokens / manager.max_tokens:.1%}")
    print(f"  是否需要压缩: {should_compact}")
    
    if should_compact:
        compacted = await manager.check_and_compact(
            conversation_history,
            current_tokens=high_tokens,
            current_task="Generate tests for UserService"
        )
        
        if compacted:
            print(f"\n压缩结果:")
            print(f"  原始 Token: {compacted.original_token_count}")
            print(f"  压缩后 Token: {compacted.compacted_token_count}")
            print(f"  压缩率: {compacted.get_compression_ratio():.1%}")
            print(f"  策略: {compacted.strategy_used.value}")
            
            # 格式化输出
            formatted = manager.compactor.format_for_prompt(compacted)
            print(f"\n格式化上下文预览 (前 300 字符):")
            print(f"  {formatted[:300]}...")
    
    return manager


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("Claude Code Inspired Features Demo")
    print("参考 Claude Code 核心能力的改进实现")
    print("=" * 60)
    
    # 运行所有演示
    await demo_universal_planner()
    await demo_hooks_system_async()
    demo_project_config()
    await demo_subagents()
    await demo_context_compaction()
    
    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
    print("\n已实现的功能:")
    print("  ✓ 通用任务规划器 (UniversalTaskPlanner)")
    print("  ✓ Hooks 生命周期系统")
    print("  ✓ 项目配置系统 (PYUT.md)")
    print("  ✓ 专业化 Subagents (Bash/Plan/Explore/TestGen)")
    print("  ✓ 智能上下文压缩 (Auto Compact)")


if __name__ == "__main__":
    asyncio.run(main())
