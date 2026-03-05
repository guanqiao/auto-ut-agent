"""
Claude Code Agent Demo - 集成 Agent 演示
==========================================

展示如何使用 ClaudeCodeAgent 处理各种编程任务
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyutagent.agent.claude_code_agent import (
    ClaudeCodeAgent,
    AgentConfig,
    create_agent
)


class MockLLMClient:
    """模拟 LLM 客户端"""
    
    async def generate(self, prompt: str) -> str:
        """根据 prompt 返回模拟响应"""
        
        # 任务理解
        if "分析以下用户编程请求" in prompt:
            return '''{
                "task_type": "test_generation",
                "description": "为 UserService 生成单元测试",
                "target_files": ["UserService.java"],
                "constraints": ["使用 JUnit 5", "Mockito"],
                "success_criteria": ["测试覆盖率 > 80%"],
                "estimated_complexity": 3,
                "context_requirements": []
            }'''
        
        # Bash 命令生成
        elif "将以下任务转换为可执行的 shell 命令" in prompt:
            return '''{
                "commands": ["mvn test -Dtest=UserServiceTest"],
                "description": "运行测试",
                "expected_output": "Tests run successfully",
                "risk_level": "low"
            }'''
        
        # 方案设计
        elif "为以下任务设计详细的实现方案" in prompt:
            return '''{
                "overview": "实现用户服务测试",
                "steps": [
                    {"order": 1, "title": "分析代码", "description": "分析 UserService", "estimated_hours": 1},
                    {"order": 2, "title": "生成测试", "description": "编写单元测试", "estimated_hours": 2}
                ],
                "files_involved": ["UserService.java", "UserServiceTest.java"],
                "risks": []
            }'''
        
        # 代码库探索
        elif "分析以下代码库探索结果" in prompt:
            return '''{
                "summary": "标准 Maven 项目",
                "key_findings": ["使用 Spring Boot"],
                "file_structure": {"total_files": 50},
                "important_files": [{"path": "pom.xml", "relevance": "high"}]
            }'''
        
        # 测试生成
        elif "为以下目标生成单元测试" in prompt:
            return '''{
                "test_class_name": "UserServiceTest",
                "package": "com.example.service",
                "imports": ["org.junit.jupiter.api.Test"],
                "test_methods": [
                    {"name": "testGetUser", "description": "测试获取用户", "test_type": "positive"}
                ],
                "coverage_analysis": {"expected_line_coverage": 0.85}
            }'''
        
        # 上下文压缩
        elif "请将以下对话历史压缩为结构化摘要" in prompt:
            return '''{
                "summary": "正在生成测试",
                "completed_tasks": ["分析代码"],
                "current_focus": "生成测试",
                "pending_tasks": ["运行测试"],
                "key_decisions": [{"decision": "使用 JUnit 5"}],
                "active_files": ["UserService.java"]
            }'''
        
        return '{"result": "success"}'


class MockToolRegistry:
    """模拟工具注册表"""
    
    def get(self, tool_name: str):
        return None


async def demo_basic_usage():
    """演示基本用法"""
    print("=" * 70)
    print("Demo 1: 基本用法 - 创建 Agent 并处理请求")
    print("=" * 70)
    
    # 创建 Agent
    llm = MockLLMClient()
    tools = MockToolRegistry()
    
    agent = ClaudeCodeAgent(
        llm_client=llm,
        tool_registry=tools,
        config=AgentConfig(
            enable_auto_compact=True,
            compact_threshold=0.85
        ),
        project_root=Path.cwd()
    )
    
    # 开始会话
    await agent.start_session()
    print(f"\n✓ 会话已启动: {agent.state.session_id}")
    
    # 获取能力说明
    capabilities = agent.get_capabilities()
    print(f"\nAgent 能力:")
    print(f"  - 支持任务类型: {len(capabilities['task_types'])} 种")
    print(f"  - 子代理: {list(capabilities['subagents'].keys())}")
    print(f"  - 自动压缩: {'启用' if capabilities['auto_compact_enabled'] else '禁用'}")
    
    # 处理请求
    print("\n处理请求: 'Generate tests for UserService'")
    result = await agent.process_request("Generate tests for UserService")
    
    print(f"\n结果:")
    print(f"  - 成功: {result['success']}")
    print(f"  - 执行时间: {result['execution_time']:.2f} 秒")
    print(f"  - 会话 ID: {result['session_id']}")
    
    # 获取会话统计
    stats = agent.get_session_stats()
    print(f"\n会话统计:")
    print(f"  - 对话次数: {stats['conversation_count']}")
    print(f"  - 压缩次数: {stats['compaction_count']}")
    
    # 停止会话
    await agent.stop()
    print("\n✓ 会话已停止")
    
    return agent


async def demo_task_types():
    """演示不同任务类型的处理"""
    print("\n" + "=" * 70)
    print("Demo 2: 不同任务类型的处理")
    print("=" * 70)
    
    agent = await create_agent(
        project_root=Path.cwd(),
        llm_client=MockLLMClient(),
        tool_registry=MockToolRegistry()
    )
    
    # 不同任务类型的请求
    tasks = [
        ("test_generation", "Generate tests for OrderService"),
        ("code_refactoring", "Refactor UserService to use dependency injection"),
        ("bug_fix", "Fix the null pointer exception in PaymentProcessor"),
        ("feature_add", "Add caching to UserService"),
        ("code_review", "Review the OrderService implementation"),
    ]
    
    print("\n处理不同类型的任务:")
    for task_type, request in tasks:
        print(f"\n  [{task_type}] {request[:50]}...")
        result = await agent.process_request(request)
        status = "✓" if result['success'] else "✗"
        print(f"    {status} 完成 (耗时: {result['execution_time']:.2f}s)")
    
    await agent.stop()
    return agent


async def demo_project_config():
    """演示项目配置初始化"""
    print("\n" + "=" * 70)
    print("Demo 3: 项目配置初始化")
    print("=" * 70)
    
    agent = await create_agent(
        project_root=Path.cwd(),
        llm_client=MockLLMClient(),
        tool_registry=MockToolRegistry()
    )
    
    print("\n项目配置状态:")
    if agent.config_manager.config_exists():
        print("  ✓ 配置文件已存在")
    else:
        print("  ℹ 配置文件不存在")
    
    # 加载项目上下文
    context = agent.config_manager.load_context()
    print(f"\n项目信息:")
    print(f"  - 名称: {context.name}")
    print(f"  - 语言: {context.language}")
    print(f"  - 构建工具: {context.build_tool.value}")
    print(f"  - Java 版本: {context.java_version}")
    print(f"  - 测试框架: {context.test_preferences.test_framework.value}")
    print(f"  - Mock 框架: {context.test_preferences.mock_framework.value}")
    
    await agent.stop()
    return agent


async def demo_hooks_and_callbacks():
    """演示 Hooks 和回调"""
    print("\n" + "=" * 70)
    print("Demo 4: Hooks 和回调")
    print("=" * 70)
    
    agent = await create_agent(
        project_root=Path.cwd(),
        llm_client=MockLLMClient(),
        tool_registry=MockToolRegistry()
    )
    
    # 注册自定义钩子
    hook_triggered = []
    
    def custom_hook(context):
        hook_triggered.append(context.hook_type.name)
        return {'custom': 'data'}
    
    agent.hook_manager.register_hook(
        name="demo_hook",
        hook_type=agent.hook_manager.registry._hooks.keys().__iter__().__class__(  # 获取任意 hook type
            agent.hook_manager.registry._hooks.keys()
        ),
        handler=custom_hook
    )
    
    print("\n已注册自定义钩子: demo_hook")
    
    # 处理请求（会触发钩子）
    print("\n处理请求（触发钩子）:")
    result = await agent.process_request("Generate tests for UserService")
    
    print(f"\n触发的钩子:")
    for hook_name in set(hook_triggered):
        print(f"  - {hook_name}")
    
    await agent.stop()
    return agent


async def demo_context_compaction():
    """演示上下文压缩"""
    print("\n" + "=" * 70)
    print("Demo 5: 上下文压缩")
    print("=" * 70)
    
    agent = await create_agent(
        project_root=Path.cwd(),
        llm_client=MockLLMClient(),
        tool_registry=MockToolRegistry(),
        config=AgentConfig(
            enable_auto_compact=True,
            compact_threshold=0.85,
            max_tokens=100000
        )
    )
    
    print("\n上下文压缩配置:")
    print(f"  - 最大 Token: {agent.config.max_tokens}")
    print(f"  - 压缩阈值: {agent.config.compact_threshold}")
    print(f"  - 自动压缩: {'启用' if agent.config.enable_auto_compact else '禁用'}")
    
    # 模拟多次对话
    print("\n模拟多次对话:")
    for i in range(3):
        request = f"Task {i+1}: Generate tests for Service{i+1}"
        result = await agent.process_request(request)
        print(f"  ✓ 请求 {i+1} 完成")
    
    # 检查压缩统计
    stats = agent.get_session_stats()
    print(f"\n压缩统计:")
    print(f"  - 压缩次数: {stats['compaction_count']}")
    print(f"  - 对话次数: {stats['conversation_count']}")
    
    await agent.stop()
    return agent


async def demo_subagent_routing():
    """演示 Subagent 路由"""
    print("\n" + "=" * 70)
    print("Demo 6: Subagent 路由")
    print("=" * 70)
    
    agent = await create_agent(
        project_root=Path.cwd(),
        llm_client=MockLLMClient(),
        tool_registry=MockToolRegistry()
    )
    
    print("\n已注册的 Subagents:")
    for name, desc in agent.subagent_router.get_capabilities().items():
        print(f"  - {name}: {desc}")
    
    # 测试路由
    test_tasks = [
        "Run maven tests",
        "Design a caching solution",
        "Explore the project structure",
        "Generate tests for UserService",
        "Fix the bug in OrderService"
    ]
    
    print("\n任务路由测试:")
    for task in test_tasks:
        # 评估每个 subagent 的匹配度
        scores = [
            (agent.name, agent.can_handle(task))
            for agent in agent.subagent_router.subagents
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best = scores[0]
        print(f"  '{task[:30]}...' -> {best[0]} (置信度: {best[1]:.2f})")
    
    await agent.stop()
    return agent


async def demo_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 70)
    print("Demo 7: 错误处理")
    print("=" * 70)
    
    # 创建一个会失败的 LLM 客户端
    failing_llm = MockLLMClient()
    failing_llm.generate = AsyncMock(side_effect=Exception("LLM API Error"))
    
    agent = await create_agent(
        project_root=Path.cwd(),
        llm_client=failing_llm,
        tool_registry=MockToolRegistry()
    )
    
    print("\n模拟错误情况:")
    result = await agent.process_request("Generate tests for UserService")
    
    print(f"  成功: {result['success']}")
    print(f"  错误: {result.get('error', 'None')}")
    print(f"  执行时间: {result['execution_time']:.2f}s")
    
    # Agent 仍然可以继续使用
    print("\n  ✓ Agent 仍然可用，可以处理后续请求")
    
    await agent.stop()
    return agent


async def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("Claude Code Agent 集成演示")
    print("展示所有新功能的集成使用")
    print("=" * 70)
    
    try:
        # 运行所有演示
        await demo_basic_usage()
        await demo_task_types()
        await demo_project_config()
        await demo_hooks_and_callbacks()
        await demo_context_compaction()
        await demo_subagent_routing()
        await demo_error_handling()
        
        print("\n" + "=" * 70)
        print("所有演示完成!")
        print("=" * 70)
        print("\n已实现的功能:")
        print("  ✓ 通用任务规划器 (UniversalTaskPlanner)")
        print("  ✓ Hooks 生命周期系统")
        print("  ✓ 项目配置系统 (PYUT.md)")
        print("  ✓ 专业化 Subagents (Bash/Plan/Explore/TestGen)")
        print("  ✓ 智能上下文压缩 (Auto Compact)")
        print("  ✓ 集成 Agent (ClaudeCodeAgent)")
        print("\n这些改进使 PyUT Agent 从专用工具进化为通用 Coding Agent")
        
    except Exception as e:
        print(f"\n演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
