# 持续重构改进计划

## 一、背景分析

### 1.1 当前架构状态

**Agent类概况**（16个）:
- BaseAgent (基类)
- ReActAgent, EnhancedAgent, UniversalCodingAgent
- ToolEnabledReActAgent, ToolUseAgent, TestGeneratorAgent
- SubAgent, SpecializedSubAgent, DelegatingSubAgent
- SpecializedAgent, AgentCoordinator
- 等

**问题**:
1. 多个Agent类功能重叠
2. 继承关系复杂
3. 接口不统一

### 1.2 重构目标

| 目标 | 说明 |
|------|------|
| 统一接口 | 所有Agent实现统一协议 |
| 减少重复 | 提取公共逻辑 |
| 提升性能 | 缓存、异步优化 |
| 代码质量 | 类型注解、文档 |

---

## 二、重构计划

### Phase 1: Agent架构统一

#### 1.1 定义Agent协议

**文件**: `pyutagent/agent/protocols.py` (扩展)

```python
from typing import Protocol, Any, Dict, List
from dataclasses import dataclass

@dataclass
class AgentCapability:
    """Agent能力定义"""
    name: str
    description: str
    parameters: Dict[str, Any]

class AgentProtocol(Protocol):
    """Agent统一协议"""

    @property
    def capabilities(self) -> List[AgentCapability]:
        """获取Agent能力列表"""
        ...

    async def execute(self, task: str, context: Dict[str, Any]) -> Any:
        """执行任务"""
        ...

    async def plan(self, goal: str) -> List[Dict[str, Any]]:
        """制定计划"""
        ...
```

#### 1.2 创建Agent基类

**文件**: `pyutagent/agent/unified_agent.py`

```python
class UnifiedAgent:
    """统一Agent基类"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools = ToolRegistry()
        self.memory = MemorySystem()
        self.llm = LLMClient()

    async def execute(self, task: str) -> AgentResult:
        """统一执行入口"""
        pass

    async def think(self, context: Dict) -> ThoughtResult:
        """统一思考"""
        pass
```

---

### Phase 2: 代码质量提升

#### 2.1 完善类型注解

**目标**:
- 所有公共方法添加类型注解
- 添加运行时类型检查
- 修复mypy警告

**步骤**:
1. 运行mypy检查
2. 逐文件修复类型错误
3. 添加pydoc文档

#### 2.2 提取公共逻辑

**可提取的公共逻辑**:
- 错误处理 → ErrorHandler
- 重试逻辑 → RetryMixin
- 状态管理 → StateMixin
- 工具执行 → ToolExecutorMixin

---

### Phase 3: 性能优化

#### 3.1 缓存优化

**改进**:
- LLM响应缓存
- 工具结果缓存
- 项目结构缓存

#### 3.2 异步优化

**改进**:
- 批量工具执行
- 并发LLM调用
- 非阻塞IO

---

### Phase 4: 测试增强

#### 4.1 测试覆盖率目标

| 模块 | 目标覆盖率 |
|------|-----------|
| agent/ | 80%+ |
| core/ | 85%+ |
| tools/ | 80%+ |
| memory/ | 75%+ |

#### 4.2 测试类型

- 单元测试
- 集成测试
- 性能测试
- 模糊测试

---

## 三、实施步骤

### Step 1: Agent协议定义 (2小时)

- [ ] 1.1 扩展AgentProtocol
- [ ] 1.2 定义AgentCapability
- [ ] 1.3 创建统一Agent基类

### Step 2: 统一工具接口 (2小时)

- [ ] 2.1 定义ToolProtocol
- [ ] 2.2 统一ToolResult格式
- [ ] 2.3 统一ToolError处理

### Step 3: 错误处理标准化 (1小时)

- [ ] 3.1 定义统一异常
- [ ] 3.2 创建ErrorHandler
- [ ] 3.3 统一错误日志

### Step 4: 类型注解完善 (3小时)

- [ ] 4.1 运行mypy检查
- [ ] 4.2 修复Agent类型
- [ ] 4.3 修复Core类型
- [ ] 4.4 修复Tools类型

### Step 5: 性能优化 (2小时)

- [ ] 5.1 实现响应缓存
- [ ] 5.2 优化工具执行
- [ ] 5.3 添加性能监控

### Step 6: 测试增强 (2小时)

- [ ] 6.1 补充单元测试
- [ ] 6.2 添加集成测试
- [ ] 6.3 性能基准测试

---

## 四、验收标准

### 代码质量
- [ ] 所有Agent类实现统一协议
- [ ] 类型注解覆盖90%+
- [ ] mypy检查通过
- [ ] ruff检查通过

### 性能
- [ ] 工具执行延迟<100ms
- [ ] LLM调用缓存命中率>50%
- [ ] 内存使用稳定

### 测试
- [ ] 测试覆盖率>80%
- [ ] 单元测试通过率100%
- [ ] 集成测试通过率100%

---

**计划制定日期**: 2026-03-04
