# 并行任务执行引擎 - 技术规格说明书

**文档版本**: v1.0  
**创建日期**: 2026-03-05  
**状态**: 评审中  
**优先级**: P0 - 高优先级

---

## 1. 概述

### 1.1 项目背景

PyUT Agent 当前已具备基础的并行执行能力（工具级并行），但相比顶级 Coding Agent（Cursor Agent、Claude Code）的任务级并行能力存在显著差距。为提升任务执行效率 3-5 倍，需要打造业界领先的并行任务执行引擎。

### 1.2 目标

构建生产级并行任务执行引擎，实现：
- **任务级并行**: 支持最大 16 任务并发执行
- **智能调度**: 动态优先级 + 抢占式调度
- **负载均衡**: 避免热点和资源饥饿
- **并行恢复**: 多路径并行错误恢复
- **进度可视化**: 实时进度追踪 + ETA 预测

### 1.3 范围

**包含**:
- ✅ 任务编排层（路由、优先级、调度）
- ✅ 并行执行层（执行引擎、资源管理、负载均衡）
- ✅ 恢复层（并行恢复、结果选择）
- ✅ 性能监控与可视化
- ✅ 完整的测试套件

**不包含**:
- ❌ 分布式执行（Phase 4 预研）
- ❌ ML 预测模型（Phase 4 高级特性）
- ❌ UI 重构（现有 UI 集成）

### 1.4 定义与缩略语

| 术语 | 定义 |
|------|------|
| Task | 任务，执行的基本单元 |
| SubTask | 子任务，任务分解后的单元 |
| Worker | 工作线程/协程，执行任务的实体 |
| ETA | Estimated Time of Arrival，预计到达时间 |
| P99 | 99 百分位，性能指标 |
| TDD | Test-Driven Development，测试驱动开发 |

---

## 2. 架构设计

### 2.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ REST API    │  │ Event Bus    │  │ CLI Interface    │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Task Orchestration Layer                    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ TaskRouter  │  │ Priority     │  │ Dynamic          │   │
│  │ (任务路由)  │  │ Manager      │  │ Scheduler        │   │
│  │             │  │ (优先级)     │  │ (动态调度)       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Parallel Execution Layer                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Parallel    │  │ Resource     │  │ Load             │   │
│  │ Executor    │  │ Optimizer    │  │ Balancer         │   │
│  │ (执行引擎)  │  │ (资源优化)   │  │ (负载均衡)       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Dependency  │  │ Progress     │  │ Fault            │   │
│  │ Tracker     │  │ Tracker      │  │ Isolator         │   │
│  │ (依赖追踪)  │  │ (进度追踪)   │  │ (故障隔离)       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Recovery Layer                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Parallel    │  │ Result       │  │ Best             │   │
│  │ Recovery    │  │ Aggregator   │  │ Selector         │   │
│  │ (并行恢复)  │  │ (结果聚合)   │  │ (最优选择)       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Tool Execution Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Tool        │  │ Tool         │  │ Result           │   │
│  │ Registry    │  │ Executor     │  │ Cache            │   │
│  │ (工具注册)  │  │ (工具执行)   │  │ (结果缓存)       │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 组件关系图

```
User Request
     │
     ▼
┌─────────────────┐
│   TaskRouter    │ ──► 任务分类 (CPU/IO/LLM)
│                 │ ──► 优先级计算
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PriorityManager │ ──► 动态优先级调整
│                 │ ──► 优先级队列管理
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│DynamicScheduler │ ──► 调度决策 (立即/延迟/并行)
│                 │ ──► 任务抢占
└────────┬────────┘
         │
         ├──────────┬──────────┬──────────┐
         ▼          ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │ Worker │ │ Worker │ │ Worker │ │ Worker │
    │   1    │ │   2    │ │   3    │ │   4    │
    └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
        │          │          │          │
        └──────────┴──────────┴──────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ ResultAggregator│ ──► 结果聚合
         │                 │ ──► 最优选择
         └────────┬────────┘
                  │
                  ▼
            Return to User
```

### 2.3 数据模型

#### 2.3.1 PriorityTask

```python
@dataclass
class PriorityTask:
    """支持优先级的任务封装"""
    id: str                              # 唯一标识
    description: str                     # 任务描述
    priority: float                      # 优先级 (0.0-1.0)
    created_at: datetime                 # 创建时间
    deadline: Optional[datetime] = None  # 截止时间
    dependencies: Set[str] = field(default_factory=set)  # 依赖任务 ID
    resource_requirements: Dict[str, float] = field(default_factory=dict)  # 资源需求
    estimated_duration: float = 0.0      # 预估执行时间 (秒)
    actual_duration: Optional[float] = None  # 实际执行时间
    status: TaskStatus = TaskStatus.PENDING  # 任务状态
    result: Optional[Any] = None         # 执行结果
    error: Optional[str] = None          # 错误信息
    retry_count: int = 0                 # 重试次数
    max_retries: int = 2                 # 最大重试次数
    preemptible: bool = True             # 是否可被抢占
    paused: bool = False                 # 是否被暂停
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    def __lt__(self, other: 'PriorityTask') -> bool:
        """用于堆排序 (优先级高的在前)"""
        return self.priority > other.priority
```

#### 2.3.2 TaskStatus

```python
class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 等待中
    QUEUED = "queued"            # 已入队
    RUNNING = "running"          # 运行中
    PAUSED = "paused"            # 已暂停
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消
```

#### 2.3.3 ResourcePool

```python
@dataclass
class ResourcePool:
    """增强的资源池"""
    resource_type: ResourceType              # 资源类型
    max_capacity: int                        # 最大容量
    current_usage: int = 0                   # 当前使用量
    reserved: int = 0                        # 预留量
    waiting_tasks: List[PriorityTask] = field(default_factory=list)  # 等待任务
    usage_history: List[Tuple[datetime, int]] = field(default_factory=list)  # 使用历史
    
    @property
    def available(self) -> int:
        """可用资源量"""
        return self.max_capacity - self.current_usage - self.reserved
    
    @property
    def utilization(self) -> float:
        """利用率"""
        if self.max_capacity == 0:
            return 0.0
        return self.current_usage / self.max_capacity
```

### 2.4 接口定义

#### 2.4.1 TaskRouter 接口

```python
class ITaskRouter(Protocol):
    """任务路由器接口"""
    
    def classify_task(self, task: PriorityTask) -> TaskType:
        """分类任务类型"""
        ...
    
    def calculate_priority(self, task: PriorityTask) -> float:
        """计算任务优先级"""
        ...
    
    def route(self, task: PriorityTask) -> RoutingDecision:
        """路由决策"""
        ...
```

#### 2.4.2 ParallelExecutionEngine 接口

```python
class IParallelExecutionEngine(Protocol):
    """并行执行引擎接口"""
    
    async def execute(
        self,
        tasks: List[PriorityTask],
        executor: Callable[[PriorityTask], Any],
        config: Optional[ExecutionConfig] = None
    ) -> ExecutionResult:
        """执行任务列表"""
        ...
    
    async def execute_with_priority(
        self,
        tasks: List[PriorityTask],
        executor: Callable[[PriorityTask], Any]
    ) -> ExecutionResult:
        """按优先级执行"""
        ...
    
    async def execute_with_dependencies(
        self,
        tasks: List[PriorityTask],
        executor: Callable[[PriorityTask], Any],
        dependency_graph: DependencyGraph
    ) -> ExecutionResult:
        """考虑依赖关系执行"""
        ...
    
    async def preempt(self, task_id: str) -> bool:
        """抢占任务"""
        ...
    
    async def pause(self, task_id: str) -> bool:
        """暂停任务"""
        ...
    
    async def resume(self, task_id: str) -> bool:
        """恢复任务"""
        ...
    
    def get_stats(self) -> ExecutionStats:
        """获取执行统计"""
        ...
```

---

## 3. 功能规格

### 3.1 任务编排 (Task Orchestration)

#### 3.1.1 任务分类 (TaskRouter)

**功能描述**: 根据任务特征自动分类，优化资源分配

**输入**:
- 任务描述
- 任务参数
- 历史执行数据

**处理逻辑**:
1. 提取任务关键词
2. 匹配资源类型模式
3. 识别任务密集型特征
4. 返回任务类型标签

**输出**:
- TaskType 枚举值 (CPU_BOUND/IO_BOUND/LLM_BOUND/MIXED)

**算法**:
```python
def classify_task(self, task: PriorityTask) -> TaskType:
    """基于关键词和历史的任务分类"""
    keywords = {
        "cpu": ["compute", "calculate", "process", "transform"],
        "io": ["read", "write", "file", "save", "load"],
        "llm": ["generate", "analyze", "summarize", "explain"],
    }
    
    text = f"{task.description} {task.metadata.get('context', '')}".lower()
    
    scores = {}
    for resource_type, words in keywords.items():
        scores[resource_type] = sum(1 for word in words if word in text)
    
    max_score = max(scores.values())
    if max_score == 0:
        return TaskType.MIXED
    
    dominant_type = max(scores, key=scores.get)
    return TaskType[dominant_type.upper() + "_BOUND"]
```

#### 3.1.2 优先级计算 (PriorityManager)

**功能描述**: 动态计算和调整任务优先级

**优先级公式**:
```
priority = base_priority * 0.4 + 
           deadline_factor * 0.3 + 
           dependency_factor * 0.2 + 
           type_factor * 0.1
```

**其中**:
- `base_priority`: 用户指定基础优先级 (0.0-1.0)
- `deadline_factor`: 截止时间紧迫性 (0.0-1.0)
- `dependency_factor`: 依赖因子 = 1.0 / (1.0 + 依赖数)
- `type_factor`: 任务类型权重

**动态调整规则**:
1. 等待时间超过阈值：优先级 +0.1
2. 被抢占任务：优先级 +0.05
3. 用户手动提升：直接设置
4. 优先级老化：每分钟 -0.01（防止饥饿）

#### 3.1.3 路由决策 (DynamicScheduler)

**功能描述**: 决定任务执行策略

**决策矩阵**:

| 优先级 | 依赖数 | 资源可用性 | 决策 |
|--------|--------|------------|------|
| >0.8 | 0 | 充足 | 立即执行 |
| >0.8 | >0 | 充足 | 等待依赖后执行 |
| 0.5-0.8 | 0 | 充足 | 并行执行 |
| 0.5-0.8 | 任意 | 不足 | 排队等待 |
| <0.5 | 任意 | 任意 | 延迟执行 |

### 3.2 并行执行 (Parallel Execution)

#### 3.2.1 执行引擎核心逻辑

**状态机**:
```
PENDING → QUEUED → RUNNING → COMPLETED
                      ↘
                       FAILED → RETRY → QUEUED
                              ↘
                               CANCELLED
```

**执行流程**:
1. 从优先级队列取出任务
2. 检查资源可用性
3. 获取资源锁
4. 执行任务
5. 释放资源
6. 记录结果
7. 通知依赖任务

#### 3.2.2 资源管理策略

**资源类型**:
- CPU: 计算密集型任务
- IO: 文件/网络 IO
- LLM: API 调用配额
- MEMORY: 内存使用
- CUSTOM: 自定义资源

**分配策略**:
1. 预留机制：关键任务预留资源
2. 预测机制：基于历史预测需求
3. 公平机制：防止资源饥饿

#### 3.2.3 负载均衡算法

**加权最小连接数**:
```python
def select_worker(workers: List[Worker], task: PriorityTask) -> Worker:
    """选择最优 Worker"""
    min_score = float('inf')
    selected = None
    
    for worker in workers:
        # 当前负载
        load = worker.current_tasks / worker.max_capacity
        
        # 历史性能
        avg_time = worker.get_avg_execution_time()
        success_rate = worker.get_success_rate()
        
        # 资源匹配度
        resource_match = worker.match_resources(task.resource_requirements)
        
        # 综合评分
        score = (
            load * 0.4 +
            avg_time * 0.3 +
            (1.0 - success_rate) * 0.2 +
            (1.0 - resource_match) * 0.1
        )
        
        if score < min_score:
            min_score = score
            selected = worker
    
    return selected
```

### 3.3 并行恢复 (Parallel Recovery)

#### 3.3.1 恢复触发条件

**自动触发**:
1. 任务执行失败
2. 错误类型匹配恢复策略
3. 重试次数未超限

**手动触发**:
1. 用户主动请求
2. 命令行指令
3. API 调用

#### 3.3.2 多策略并行恢复

**流程**:
1. 收集所有可用恢复策略
2. 选择 Top-N 策略（默认 3 个）
3. 并行执行策略
4. 选择最优结果
5. 取消剩余策略
6. 应用最佳结果

**策略选择优先级**:
```python
STRATEGY_PRIORITY = {
    ErrorCategory.COMPILATION_ERROR: [
        RecoveryStrategy.ANALYZE_AND_FIX,
        RecoveryStrategy.FALLBACK_ALTERNATIVE,
        RecoveryStrategy.RESET_AND_REGENERATE,
    ],
    ErrorCategory.LLM_API_ERROR: [
        RecoveryStrategy.RETRY_WITH_BACKOFF,
        RecoveryStrategy.RETRY_IMMEDIATE,
        RecoveryStrategy.FALLBACK_ALTERNATIVE,
    ],
    # ... 其他错误类型
}
```

### 3.4 进度追踪 (Progress Tracking)

#### 3.4.1 进度计算

**整体进度**:
```
progress = (completed_tasks / total_tasks) * 100
```

**加权进度** (考虑任务复杂度):
```
weighted_progress = (
    sum(completed_task_weights) / 
    sum(total_task_weights)
) * 100
```

#### 3.4.2 ETA 预测

**简单预测**:
```
eta = (remaining_tasks / completed_tasks) * elapsed_time
```

**加权预测** (考虑复杂度):
```
avg_speed = completed_complexity / elapsed_time
eta = remaining_complexity / avg_speed
```

**置信区间**:
```
confidence = min(1.0, completed_tasks / 10)
eta_range = eta * (1.0 - confidence * 0.8)
```

---

## 4. 非功能规格

### 4.1 性能要求

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| 吞吐量 | >3 倍于串行 | 基准测试对比 |
| P99 延迟 | <5 秒 | 监控统计 |
| 资源利用率 | >70% | 系统监控 |
| 并发任务数 | 最大 16 | 压力测试 |
| 优先级队列响应 | <10ms | 基准测试 |
| ETA 预测误差 | <20% | 实际对比 |

### 4.2 可靠性要求

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 任务成功率 | >95% | 正常场景 |
| 恢复成功率 | >85% | 错误场景 |
| 无死锁 | 100% | 设计保证 |
| 无资源泄漏 | 100% | 设计保证 |
| 故障隔离 | 100% | 单任务失败不影响其他 |

### 4.3 可扩展性要求

- **水平扩展**: 支持增加 Worker 数量
- **垂直扩展**: 支持增加资源池容量
- **线性扩展**: 8 并发内接近线性加速

### 4.4 可维护性要求

- **测试覆盖**: >90%
- **文档完整**: 100% API 文档
- **代码审查**: 所有代码需审查
- **日志完整**: 关键操作 100% 日志

---

## 5. 实施计划

### 5.1 Phase 分解

| Phase | 周期 | 目标 | 交付物 |
|-------|------|------|--------|
| Phase 1 | Week 1-2 | 核心引擎增强 | ParallelExecutionEngine v2 |
| Phase 2 | Week 3-4 | 负载均衡优化 | LoadBalancer, ProgressTracker |
| Phase 3 | Week 5 | 并行恢复集成 | RecoveryOrchestrator |
| Phase 4 | Week 6-7 | 高级特性 | PredictiveScheduler |
| Phase 5 | Week 8 | 生产就绪 | 性能报告、文档 |

### 5.2 关键里程碑

- **M1** (Week 2): 核心引擎通过验收测试
- **M2** (Week 4): 负载均衡效果达标
- **M3** (Week 5): 恢复集成完成
- **M4** (Week 7): 所有功能开发完成
- **M5** (Week 8): 生产发布

---

## 6. 测试策略

### 6.1 测试层次

```
┌─────────────────────┐
│   E2E 测试 (10%)    │ ──► 完整流程
├─────────────────────┤
│   集成测试 (20%)    │ ──► 组件协作
├─────────────────────┤
│   单元测试 (70%)    │ ──► 单个组件
└─────────────────────┘
```

### 6.2 测试覆盖要求

| 组件类型 | 覆盖率要求 | 测试重点 |
|----------|------------|----------|
| 核心引擎 | >95% | 所有分支、边界条件 |
| 调度算法 | >90% | 各种调度场景 |
| 工具类 | >85% | 正常流程、异常处理 |
| UI 组件 | >70% | 用户交互流程 |

### 6.3 性能基准测试

**基准场景**:
1. 单任务执行（基线）
2. 10 任务并行
3. 50 任务并发
4. 混合负载（不同优先级）
5. 压力测试（资源耗尽）

**通过标准**:
- 并行加速比 > 3.0
- 资源利用率 > 70%
- 无内存泄漏
- P99 延迟 < 5 秒

---

## 7. 部署与运维

### 7.1 部署架构

```
┌─────────────────────────────────────┐
│          Load Balancer              │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Worker │ │Worker │ │Worker │
│  1    │ │  2    │ │  N    │
└───────┘ └───────┘ └───────┘
```

### 7.2 监控指标

**系统指标**:
- CPU 使用率
- 内存使用率
- 磁盘 IO
- 网络 IO

**业务指标**:
- 并发任务数
- 任务成功率
- 平均执行时间
- 资源利用率

### 7.3 日志规范

**日志级别**:
- ERROR: 错误、异常
- WARN: 警告、降级
- INFO: 关键操作、状态变更
- DEBUG: 详细调试信息

**日志格式**:
```
[时间戳] [级别] [组件] [任务 ID] 消息
```

---

## 8. 风险与缓解

### 8.1 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 死锁 | 高 | 中 | 严格锁顺序、超时机制 |
| 竞态条件 | 高 | 中 | 充分测试、形式化验证 |
| 性能退化 | 高 | 低 | 基准测试、持续监控 |
| 资源饥饿 | 中 | 中 | 公平调度、优先级老化 |

### 8.2 进度风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 复杂度低估 | 高 | 中 | 分阶段实施、早期原型 |
| 技术难点 | 中 | 中 | 充分调研、专家咨询 |
| 测试不足 | 高 | 低 | TDD、自动化测试 |

---

## 9. 附录

### 9.1 参考文档

- [Python asyncio 最佳实践](https://docs.python.org/3/library/asyncio.html)
- [并发设计模式](https://www.patternsforconcurrency.com/)
- [调度算法理论](https://en.wikipedia.org/wiki/Scheduling_(computing))

### 9.2 竞品分析

- Cursor Agent 并行执行架构
- Claude Code 终端代理模式
- Windsurf Flow 流式协作

### 9.3 版本历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| v1.0 | 2026-03-05 | AI Agent | 初始版本 |

---

**审批**:
- [ ] 技术负责人审批
- [ ] 产品经理审批
- [ ] 架构师审批

**生效日期**: 审批通过后生效
