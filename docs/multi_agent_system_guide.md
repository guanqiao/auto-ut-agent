# 多智能体系统使用指南

## 概述

PyUT Agent 的多智能体协作系统是一个分布式测试生成框架，通过多个专业化智能体协同工作，提高测试生成的效率和质量。

## 核心概念

### 智能体角色

系统包含以下专业化智能体：

| 智能体 | 角色 | 职责 |
|--------|------|------|
| **CodeAnalysisAgent** | ANALYZER | 分析 Java 代码结构，提取方法、依赖、复杂度等信息 |
| **TestGenerationAgent** | IMPLEMENTER | 生成 JUnit 测试代码，包括测试模板、Mock 配置 |
| **TestFixAgent** | FIXER | 修复编译错误和测试失败，支持自动修复建议 |

### 任务类型

每个智能体支持特定的任务类型：

**CodeAnalysisAgent:**
- `analyze_code` - 分析 Java 源代码
- `extract_methods` - 提取方法信息
- `analyze_dependencies` - 分析代码依赖
- `identify_test_targets` - 识别高价值测试目标

**TestGenerationAgent:**
- `generate_tests` - 为类生成完整测试
- `generate_test_for_method` - 为特定方法生成测试
- `generate_mocks` - 生成 Mock 对象配置
- `create_test_fixture` - 创建测试 Fixture

**TestFixAgent:**
- `fix_compilation_error` - 修复编译错误
- `fix_test_failure` - 修复测试失败
- `fix_import_error` - 修复导入错误
- `fix_mock_error` - 修复 Mock 配置错误
- `analyze_error` - 分析错误类型

## 快速开始

### 启用多智能体模式

多智能体模式默认已启用。在 `EnhancedAgent` 配置中：

```python
from pyutagent.agent import EnhancedAgent

agent = EnhancedAgent(
    llm_client=llm_client,
    config=AgentConfig(
        enable_multi_agent=True,  # 默认启用
        multi_agent_workers=3
    )
)
```

### 在批量生成中使用

```python
from pyutagent.services.batch_generator import BatchGenerator, BatchConfig

config = BatchConfig(
    enable_multi_agent=True,  # 启用多智能体模式
    multi_agent_workers=3,    # 智能体工作线程数
    parallel_workers=4        # 并行工作线程数
)

generator = BatchGenerator(
    project_path="/path/to/project",
    llm_client=llm_client,
    config=config
)

results = await generator.generate_all()
```

## 高级配置

### 任务分配策略

支持多种任务分配策略：

```python
from pyutagent.agent.multi_agent import TaskAllocation

coordinator = AgentCoordinator(
    allocation_strategy=TaskAllocation.CAPABILITY_MATCH  # 默认
)
```

可用策略：
- `ROUND_ROBIN` - 轮询分配
- `CAPABILITY_MATCH` - 能力匹配（默认）
- `LOAD_BALANCED` - 负载均衡
- `PRIORITY_BASED` - 基于优先级

### 性能调优

#### 批处理大小

调整批处理大小以优化吞吐量：

```python
# 在 AgentCoordinator 中设置
self._batch_size = 10  # 默认 10
```

#### 缓存配置

**代码分析缓存：**

```python
# CodeAnalysisAgent 缓存配置
self._cache_ttl_seconds = 300      # 5 分钟 TTL
self._max_cache_size = 1000        # 最大 1000 条缓存
```

**依赖检查缓存：**

```python
# AgentCoordinator 依赖缓存
self._dependency_cache_ttl = 5.0   # 5 秒 TTL
```

查看缓存统计：

```python
stats = code_analysis_agent.get_cache_stats()
print(f"缓存命中率: {stats['hit_rate']:.2%}")
print(f"缓存大小: {stats['cache_size']}")
```

## 架构说明

### 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (CLI / GUI / API)                          │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                BatchGenerator                            │
│         (批量生成协调器)                                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              AgentCoordinator                            │
│    (任务调度、分配、监控)                                  │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│CodeAnalysis  │ │TestGen   │ │TestFix     │
│Agent         │ │Agent     │ │Agent       │
└──────────────┘ └──────────┘ └────────────┘
```

### 消息流

1. **任务提交** → AgentCoordinator
2. **任务分配** → 专业化智能体
3. **任务执行** → 智能体处理
4. **结果返回** → AgentCoordinator
5. **知识共享** → SharedKnowledgeBase

### 数据流

```
Java Source Code
       ↓
CodeAnalysisAgent (解析 AST)
       ↓
TestGenerationAgent (生成测试)
       ↓
TestFixAgent (修复错误)
       ↓
Generated Test Files
```

## 故障排查

### 常见问题

#### 1. 多智能体模式未生效

**症状：** 批量生成时仍使用标准模式

**检查：**
```python
# 确认配置
print(f"enable_multi_agent: {config.enable_multi_agent}")
print(f"_multi_agent_initialized: {generator._multi_agent_initialized}")
```

**解决：**
- 确保 `BatchConfig.enable_multi_agent = True`
- 检查初始化日志中是否有错误

#### 2. 任务执行超时

**症状：** 任务长时间无响应

**解决：**
```python
# 增加超时时间
config = BatchConfig(
    timeout_per_file=600,  # 增加到 10 分钟
)
```

#### 3. 缓存命中率低

**症状：** 相同文件被重复分析

**检查：**
```python
stats = agent.get_cache_stats()
print(f"命中率: {stats['hit_rate']}")
print(f"命中: {stats['cache_hits']}, 未命中: {stats['cache_misses']}")
```

**解决：**
- 增加 TTL：`agent._cache_ttl_seconds = 600`
- 检查文件是否频繁修改

#### 4. 内存占用过高

**症状：** 系统内存不足

**解决：**
```python
# 减小缓存大小
agent._max_cache_size = 500

# 减小批处理大小
coordinator._batch_size = 5
```

### 调试技巧

#### 启用详细日志

```python
import logging

logging.getLogger("pyutagent.agent.multi_agent").setLevel(logging.DEBUG)
```

#### 查看智能体状态

```python
# 获取智能体状态
status = coordinator.get_agent_status()
for agent_id, info in status.items():
    print(f"{agent_id}: {info['status']} "
          f"(completed: {info['tasks_completed']})" )
```

#### 查看任务状态

```python
# 获取任务状态
task_status = coordinator.get_task_status(task_id)
print(f"状态: {task_status['status']}")
print(f"分配智能体: {task_status['assigned_agent']}")
```

## 最佳实践

### 1. 合理配置批处理大小

- **小项目**（< 50 个文件）：`batch_size = 5`
- **中项目**（50-200 个文件）：`batch_size = 10`（默认）
- **大项目**（> 200 个文件）：`batch_size = 20`

### 2. 调整缓存 TTL

- **开发阶段**（频繁修改）：`ttl = 60` 秒
- **稳定阶段**（很少修改）：`ttl = 600` 秒

### 3. 选择合适的分配策略

- **通用场景**：`CAPABILITY_MATCH`（默认）
- **负载不均**：`LOAD_BALANCED`
- **简单场景**：`ROUND_ROBIN`

### 4. 监控性能指标

```python
# 获取协调器统计
stats = coordinator.get_stats()
print(f"任务创建: {stats['tasks_created']}")
print(f"任务完成: {stats['tasks_completed']}")
print(f"任务失败: {stats['tasks_failed']}")
print(f"批处理任务: {stats.get('tasks_batch_processed', 0)}")
```

## API 参考

### AgentCoordinator

#### 方法

- `register_agent(agent_id, capabilities, role)` - 注册智能体
- `submit_task(task_type, payload, priority, dependencies)` - 提交任务
- `get_task_status(task_id)` - 获取任务状态
- `get_agent_status(agent_id)` - 获取智能体状态
- `get_stats()` - 获取统计信息

### CodeAnalysisAgent

#### 方法

- `execute_task(task)` - 执行任务
- `get_cache_stats()` - 获取缓存统计

### TestGenerationAgent

#### 方法

- `execute_task(task)` - 执行任务

### TestFixAgent

#### 方法

- `execute_task(task)` - 执行任务

## 示例代码

### 完整示例

```python
import asyncio
from pyutagent.agent.multi_agent import (
    AgentCoordinator,
    CodeAnalysisAgent,
    TestGenerationAgent,
    TestFixAgent,
    MessageBus,
    SharedKnowledgeBase,
    ExperienceReplay,
    AgentRole
)

async def main():
    # 创建基础设施
    message_bus = MessageBus()
    knowledge_base = SharedKnowledgeBase()
    experience_replay = ExperienceReplay()
    
    # 创建协调器
    coordinator = AgentCoordinator(
        message_bus=message_bus,
        knowledge_base=knowledge_base,
        experience_replay=experience_replay
    )
    
    # 创建智能体
    code_analyzer = CodeAnalysisAgent(
        agent_id="analyzer_1",
        message_bus=message_bus,
        knowledge_base=knowledge_base,
        experience_replay=experience_replay
    )
    
    test_generator = TestGenerationAgent(
        agent_id="generator_1",
        message_bus=message_bus,
        knowledge_base=knowledge_base,
        experience_replay=experience_replay,
        llm_client=llm_client
    )
    
    # 注册智能体
    coordinator.register_agent(
        code_analyzer.agent_id,
        code_analyzer.capabilities,
        AgentRole.ANALYZER
    )
    
    coordinator.register_agent(
        test_generator.agent_id,
        test_generator.capabilities,
        AgentRole.IMPLEMENTER
    )
    
    # 启动协调器
    await coordinator.start()
    
    # 提交代码分析任务
    task_id = await coordinator.submit_task(
        task_type="analyze_code",
        payload={
            "file_path": "UserService.java",
            "source_code": "public class UserService { ... }"
        },
        priority=1
    )
    
    # 等待任务完成
    await coordinator.wait_for_task(task_id, timeout=30)
    
    # 获取结果
    status = coordinator.get_task_status(task_id)
    print(f"任务状态: {status}")
    
    # 停止协调器
    await coordinator.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 相关文档

- [项目 README](../../README.md)
- [架构设计文档](./architecture.md)
- [API 参考文档](./api_reference.md)

## 更新日志

### v1.0.0

- 初始版本发布
- 支持 3 个专业化智能体
- 实现任务批处理和缓存机制
- 41 个单元测试通过
