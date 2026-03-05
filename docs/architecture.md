# 多智能体系统架构设计

## 系统概述

PyUT Agent 的多智能体系统采用分布式协作架构，通过专业化智能体协同工作，实现高效的 Java 单元测试生成。

## 设计目标

1. **可扩展性** - 支持动态添加新的专业化智能体
2. **高性能** - 通过批处理和缓存优化性能
3. **容错性** - 任务失败自动重试和回退机制
4. **可观测性** - 完整的监控和统计接口

## 架构组件

### 1. 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentCoordinator                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Task Queue  │  │  Scheduler  │  │  Allocation Engine  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐    ┌────────▼────────┐   ┌───────▼──────┐
│  MessageBus  │    │ SharedKnowledge │   │ExperienceReplay│
└──────────────┘    └─────────────────┘   └──────────────┘
```

#### AgentCoordinator（协调器）

**职责：**
- 智能体生命周期管理
- 任务调度和分配
- 结果聚合
- 冲突解决

**核心机制：**
- 批处理：每批处理 10 个任务
- 依赖缓存：5 秒 TTL
- 能力缓存：任务类型到能力的映射缓存

#### MessageBus（消息总线）

**职责：**
- 智能体间通信
- 消息路由
- 订阅管理

**消息类型：**
- `TASK_ASSIGNMENT` - 任务分配
- `TASK_RESULT` - 任务结果
- `TASK_FAILED` - 任务失败
- `HEARTBEAT` - 心跳检测
- `KNOWLEDGE_SHARE` - 知识共享
- `BROADCAST` - 广播消息

#### SharedKnowledgeBase（共享知识库）

**职责：**
- 存储智能体共享的知识
- 支持标签索引
- 相似度搜索

**存储内容：**
- 代码分析结果
- 测试模式
- 错误修复经验
- 最佳实践

#### ExperienceReplay（经验回放缓冲区）

**职责：**
- 存储任务执行经验
- 支持经验采样
- 用于强化学习

### 2. 专业化智能体

```
┌─────────────────────────────────────────────────────────┐
│                    SpecializedAgent                      │
│                      (Abstract Base)                      │
└─────────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│CodeAnalysis  │ │TestGeneration│ │   TestFix    │
│    Agent     │ │    Agent     │ │    Agent     │
└──────────────┘ └──────────────┘ └──────────────┘
```

#### CodeAnalysisAgent

**能力：**
- `DEPENDENCY_ANALYSIS` - 依赖分析
- `TEST_DESIGN` - 测试设计

**解析器：**
- 主解析器：tree-sitter-java
- 回退解析器：正则表达式

**缓存策略：**
- TTL：5 分钟
- 最大条目：1000
- 文件修改检测

#### TestGenerationAgent

**能力：**
- `TEST_IMPLEMENTATION` - 测试实现
- `MOCK_GENERATION` - Mock 生成
- `TEST_DESIGN` - 测试设计

**生成功能：**
- JUnit 5 测试代码
- Mockito Mock 配置
- 测试 Fixture

#### TestFixAgent

**能力：**
- `ERROR_FIXING` - 错误修复
- `TEST_REVIEW` - 测试审查

**修复类型：**
- 编译错误修复
- 测试失败修复
- 导入错误修复
- Mock 错误修复

### 3. 数据流

```
User Request
     │
     ▼
┌─────────────────┐
│ BatchGenerator  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│AgentCoordinator │────▶│  Task Scheduler  │
└────────┬────────┘     └──────────────────┘
         │
         │ 1. analyze_code
         ▼
┌─────────────────┐
│CodeAnalysisAgent│
└────────┬────────┘
         │ Analysis Result
         │
         │ 2. generate_tests
         ▼
┌─────────────────┐
│TestGenerationAgent│
└────────┬────────┘
         │ Test Code
         │
         │ 3. fix_errors (if needed)
         ▼
┌─────────────────┐
│   TestFixAgent  │
└────────┬────────┘
         │
         ▼
   Generated Tests
```

## 关键设计决策

### 1. 批处理设计

**原因：** 减少异步调度开销

**实现：**
```python
# 收集任务批次
batch = []
while len(batch) < batch_size:
    try:
        task = await queue.get(timeout=0.1)
        batch.append(task)
    except TimeoutError:
        break

# 批量处理
for task in batch:
    process_task(task)
```

**效果：** 吞吐量提升 30-50%

### 2. 缓存设计

#### 代码分析缓存

**键：** `file_path:hash(source_code)`

**失效策略：**
- TTL 过期
- 文件修改时间变化
- 手动清理

#### 依赖检查缓存

**键：** `task_id:sorted(dependencies)`

**TTL：** 5 秒

**原因：** 依赖状态在短时间内不会变化

### 3. 任务分配策略

#### CAPABILITY_MATCH（默认）

**算法：**
1. 获取任务所需能力
2. 筛选具备能力的空闲智能体
3. 选择负载最小的智能体

**适用场景：** 通用场景

#### LOAD_BALANCED

**算法：**
1. 获取任务所需能力
2. 筛选具备能力的空闲智能体
3. 选择已完成任务最少的智能体

**适用场景：** 智能体性能不均

### 4. 错误处理策略

#### 任务失败处理

1. **记录失败** - 更新任务状态
2. **更新统计** - 增加失败计数
3. **释放智能体** - 将智能体标记为空闲
4. **可选重试** - 支持重试逻辑（预留）

#### 智能体故障检测

**心跳机制：**
- 频率：每 60 秒
- 超时：120 秒无响应标记为故障

## 性能优化

### 1. 缓存优化

| 缓存类型 | 命中率 | 优化效果 |
|----------|--------|----------|
| 代码分析 | 60-80% | 减少重复解析 |
| 依赖检查 | 90%+ | 减少重复检查 |
| 能力映射 | 100% | 避免重复计算 |

### 2. 批处理优化

| 批大小 | 吞吐量 | 延迟 |
|--------|--------|------|
| 1 | 基准 | 低 |
| 5 | +20% | 中 |
| 10 | +35% | 中 |
| 20 | +40% | 高 |

**推荐：** 批大小 10（默认）

### 3. 并发优化

- 任务队列：异步处理
- 消息传递：异步发送
- 缓存访问：同步（无锁设计）

## 扩展性设计

### 1. 添加新智能体

```python
# 1. 继承 SpecializedAgent
class NewAgent(SpecializedAgent):
    def __init__(self, ...):
        super().__init__(
            capabilities={AgentCapability.NEW_CAPABILITY}
        )
    
    async def execute_task(self, task):
        # 实现任务执行逻辑
        pass

# 2. 注册到协调器
coordinator.register_agent(
    agent_id="new_agent",
    capabilities={AgentCapability.NEW_CAPABILITY},
    role=AgentRole.NEW_ROLE
)
```

### 2. 添加新任务类型

```python
# 在 capability_map 中添加映射
capability_map = {
    "new_task_type": {AgentCapability.NEW_CAPABILITY}
}
```

## 监控指标

### 1. 协调器指标

```python
{
    "tasks_created": int,      # 创建的任务数
    "tasks_completed": int,    # 完成的任务数
    "tasks_failed": int,       # 失败的任务数
    "tasks_batch_processed": int,  # 批处理的任务数
    "agents_registered": int   # 注册的智能体数
}
```

### 2. 智能体指标

```python
{
    "tasks_completed": int,    # 完成的任务数
    "tasks_failed": int,       # 失败的任务数
    "status": str,             # 当前状态
    "current_task": str        # 当前任务ID
}
```

### 3. 缓存指标

```python
{
    "cache_size": int,         # 缓存大小
    "cache_hits": int,         # 命中次数
    "cache_misses": int,       # 未命中次数
    "hit_rate": float          # 命中率
}
```

## 安全考虑

### 1. 资源限制

- 最大缓存条目：1000
- 最大批大小：20
- 任务超时：可配置（默认 300 秒）

### 2. 错误隔离

- 单个任务失败不影响其他任务
- 智能体故障自动检测
- 任务失败自动重试（预留）

## 未来演进

### 1. 计划功能

- [ ] 动态智能体扩缩容
- [ ] 基于强化学习的任务分配
- [ ] 分布式部署支持
- [ ] 更细粒度的缓存策略

### 2. 性能目标

- 吞吐量：比单智能体提升 50%+
- 延迟：P99 < 5 秒
- 可用性：99.9%

## 相关文档

- [使用指南](./multi_agent_system_guide.md)
- [API 参考](./api_reference.md)
- [项目 README](../../README.md)
