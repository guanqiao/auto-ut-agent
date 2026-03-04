# PyUTAgent 核心架构文档

## 概述

PyUTAgent 是一个智能单元测试生成代理系统，采用模块化、组件化架构设计。

## 架构层次

```
┌─────────────────────────────────────┐
│         UI Layer (Optional)         │
│      main_window.py, dialogs/       │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│         Agent Layer                 │
│  react_agent.py, incremental_fixer  │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│         Core Layer                  │
│  event_bus, state_store, metrics    │
└─────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────┐
│         LLM Layer                   │
│  client.py, prompt_cache.py         │
└─────────────────────────────────────┘
```

## 核心模块

### 1. 事件总线 (Event Bus)

**文件**: `pyutagent/core/event_bus.py`

**功能**:
- 发布/订阅模式
- 同步和异步事件处理
- 组件解耦

**使用示例**:
```python
bus = EventBus()
bus.subscribe('event', handler)
bus.publish('event', data)
```

### 2. 状态存储 (State Store)

**文件**: `pyutagent/core/state_store.py`

**功能**:
- Redux 风格的状态管理
- Action 模式更新状态
- 状态历史追踪

**核心概念**:
- `AgentState`: 状态数据
- `Action`: 状态变更
- `LifecycleState`: 生命周期状态

### 3. 消息总线 (Message Bus)

**文件**: `pyutagent/core/message_bus.py`

**功能**:
- 消息队列
- 优先级处理
- 发布/订阅

### 4. 组件注册表 (Component Registry)

**文件**: `pyutagent/core/component_registry.py`

**功能**:
- 组件注册和发现
- 依赖管理
- 生命周期管理

**使用示例**:
```python
@component('my_component')
class MyComponent(SimpleComponent):
    pass

registry.register('my_component', MyComponent)
```

### 5. 指标收集 (Metrics)

**文件**: `pyutagent/core/metrics.py`

**功能**:
- 计数器、仪表盘、直方图
- 性能追踪
- 全局指标收集

**指标类型**:
- `COUNTER`: 累加值
- `GAUGE`: 瞬时值
- `HISTOGRAM`: 分布统计

### 6. 错误处理 (Error Handling)

**文件**: `pyutagent/core/error_handling.py`

**功能**:
- 统一错误类型
- 错误传播链
- 恢复策略

**错误分类**:
- `TEST_FAILURE`: 测试失败
- `COMPILE_ERROR`: 编译错误
- `TIMEOUT`: 超时
- `LLM_ERROR`: LLM 错误

### 7. Action 系统 (Actions)

**文件**: `pyutagent/core/actions.py`

**功能**:
- `BatchAction`: 批量操作
- `TransactionalAction`: 事务操作
- `ConditionalAction`: 条件操作
- `ActionSequence`: 操作序列

## LLM 模块

### 1. LLM 客户端

**文件**: `pyutagent/llm/client.py`

**功能**:
- 统一的 LLM 接口
- 异步调用
- 错误处理

### 2. Prompt 缓存

**文件**: `pyutagent/llm/prompt_cache.py`

**功能**:
- LRU 缓存
- TTL 过期
- 缓存统计

### 3. 多级缓存

**文件**: `pyutagent/llm/multi_level_cache.py`

**功能**:
- L1 内存缓存
- L2 磁盘缓存
- 压缩支持
- 缓存预热

**性能提升**: 5-10 倍

## Agent 模块

### 1. 增量修复器 (Incremental Fixer)

**文件**: `pyutagent/agent/incremental_fixer.py`

**功能**:
- 测试失败聚类
- 批量修复
- 减少 LLM 调用

**优化效果**: 减少 60-80% LLM 调用

### 2. 智能聚类 (Smart Clusterer)

**文件**: `pyutagent/agent/smart_clusterer.py`

**功能**:
- 词向量语义分析
- 余弦相似度计算
- 自动根因提取

**算法**:
1. 失败文本分词
2. 词向量嵌入
3. 余弦相似度计算
4. 基于阈值聚类

## 数据流

### 典型工作流程

```
1. 用户输入代码
   ↓
2. 分析代码结构
   ↓
3. 生成测试用例 (LLM)
   ↓
4. 运行测试
   ↓
5. 如果有失败:
   - 聚类失败
   - 批量修复
   - 回到步骤 4
6. 输出测试代码
```

## 扩展点

### 1. 自定义组件

```python
from pyutagent.core.component_registry import SimpleComponent, component

@component('custom_component')
class CustomComponent(SimpleComponent):
    def initialize(self):
        # 初始化逻辑
        return True
```

### 2. 自定义 Action

```python
from pyutagent.core.state_store import Action, AgentState

class CustomAction(Action):
    def reduce(self, state: AgentState) -> AgentState:
        # 状态变更逻辑
        return new_state
```

### 3. 自定义错误恢复

```python
from pyutagent.core.error_handling import RecoveryStrategy

strategy = RecoveryStrategy(
    category=ErrorCategory.CUSTOM,
    action=RecoveryAction.RETRY,
    max_retries=3
)
```

## 性能优化建议

1. **使用缓存**: 对 LLM 调用结果进行缓存
2. **批量操作**: 减少 LLM 调用次数
3. **异步处理**: 提高并发性能
4. **智能聚类**: 相似失败批量修复

## 监控和调试

### 1. 启用日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. 查看指标

```python
from pyutagent.core.metrics import get_metrics
metrics = get_metrics()
print(metrics)
```

### 3. 性能分析

```python
from pyutagent.core.metrics import PerformanceTracker

tracker = PerformanceTracker()
tracker.start_timer('operation')
# ... 操作 ...
elapsed = tracker.stop_timer('operation')
```

## 测试策略

### 单元测试

```bash
pytest tests/unit/ -v
```

### 集成测试

```bash
pytest tests/integration/ -v
```

### 性能测试

```bash
pytest tests/benchmarks/ -v
```

## 依赖关系

```
pyutagent/
├── core/           # 核心模块
│   ├── event_bus.py
│   ├── state_store.py
│   ├── message_bus.py
│   ├── component_registry.py
│   ├── metrics.py
│   ├── error_handling.py
│   └── actions.py
├── llm/            # LLM 模块
│   ├── client.py
│   ├── prompt_cache.py
│   └── multi_level_cache.py
├── agent/          # Agent 模块
│   ├── incremental_fixer.py
│   └── smart_clusterer.py
└── ui/             # UI 模块 (可选)
    ├── main_window.py
    └── dialogs/
```

## 版本信息

- **当前版本**: 0.1.0
- **Python 版本**: 3.12+
- **主要依赖**: pytest, asyncio

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 编写测试
4. 实现功能 (TDD)
5. 运行测试
6. 提交 PR

## 许可证

MIT License
