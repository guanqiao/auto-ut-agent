# Sub Agent 能力增强与协调机制实现计划

## 一、背景分析

### 1.1 现有实现评估

| 模块 | 文件 | 现状 | 差距 |
|------|------|------|------|
| SubAgent机制 | `subagents.py` | 基础框架完整 | 缺少动态创建、与主Agent集成 |
| 多Agent协调 | `multi_agent/` | 消息总线+协调器 | 缺少层级分解、冲突解决 |
| Skills机制 | `skills.py` | 完整的Skill框架 | 未与SubAgent集成 |
| 自主循环 | `enhanced_autonomous_loop.py` | LLM驱动决策 | 缺少SubAgent委派能力 |

### 1.2 核心差距

1. **SubAgent与主Agent解耦** - 现有SubAgent独立运行，无法被主Agent动态调用
2. **缺少任务委派机制** - 主Agent无法将子任务委派给专用SubAgent
3. **协调策略单一** - 仅有能力匹配，缺少层级分解和动态路由
4. **Skill与SubAgent分离** - 两者独立存在，未形成协同效应

## 二、实现目标

### 2.1 核心目标

1. **增强SubAgent能力** - 支持动态创建、自主决策、Skill绑定
2. **完善协调机制** - 层级任务分解、智能路由、冲突解决
3. **主Agent集成** - 任务委派、结果聚合、上下文共享
4. **端到端工作流** - 从任务理解到SubAgent执行的完整链路

### 2.2 技术指标

- SubAgent创建延迟 < 100ms
- 任务委派成功率 > 95%
- 协调开销 < 10%
- 支持最多 10 个并发SubAgent

## 三、详细实现步骤

### Phase 1: SubAgent能力增强 (预计3个任务)

#### 任务 1.1: 创建 DelegatingSubAgent 类

**文件**: `pyutagent/agent/delegating_subagent.py`

**功能**:
- 继承现有 SubAgent 基类
- 支持绑定 Skill 执行
- 支持独立的 LLM 决策
- 支持结果回调和进度报告

**关键接口**:
```python
class DelegatingSubAgent(SubAgent):
    async def delegate(self, task: Task, skill: Optional[Skill] = None) -> TaskResult
    async def execute_with_skill(self, task: Task, skill: Skill) -> Any
    async def execute_autonomously(self, task: Task) -> Any
    def bind_skill(self, skill: Skill) -> None
    def set_progress_callback(self, callback: Callable) -> None
```

#### 任务 1.2: 创建 SubAgentFactory 工厂类

**文件**: `pyutagent/agent/subagent_factory.py`

**功能**:
- 动态创建不同类型的 SubAgent
- 管理 SubAgent 生命周期
- 支持从配置创建
- 支持模板化创建

**关键接口**:
```python
class SubAgentFactory:
    def create_agent(self, agent_type: str, config: Dict) -> DelegatingSubAgent
    def create_from_skill(self, skill: Skill) -> DelegatingSubAgent
    def create_specialized(self, capability: AgentCapability) -> DelegatingSubAgent
    def destroy_agent(self, agent_id: str) -> bool
    def get_agent_pool(self, agent_type: str) -> List[DelegatingSubAgent]
```

#### 任务 1.3: 增强 SubAgentManager

**文件**: 修改 `pyutagent/agent/subagents.py`

**功能**:
- 集成 SubAgentFactory
- 支持按需创建/销毁
- 支持任务委派
- 支持结果聚合

**新增接口**:
```python
class SubAgentManager:
    async def delegate_task(self, task: Task, agent_type: Optional[str] = None) -> TaskResult
    async def delegate_to_skill(self, task: Task, skill_name: str) -> TaskResult
    async def broadcast_task(self, task: Task) -> List[TaskResult]
    def get_or_create_agent(self, agent_type: str) -> DelegatingSubAgent
    async def aggregate_results(self, task_ids: List[str]) -> Dict
```

### Phase 2: 协调机制增强 (预计3个任务)

#### 任务 2.1: 创建 HierarchicalTaskPlanner

**文件**: `pyutagent/agent/hierarchical_planner.py`

**功能**:
- 层级任务分解
- 依赖关系分析
- 并行任务识别
- 执行顺序优化

**关键接口**:
```python
class HierarchicalTaskPlanner:
    async def decompose(self, task: str, context: Dict) -> TaskTree
    def analyze_dependencies(self, subtasks: List[Subtask]) -> DependencyGraph
    def identify_parallel_tasks(self, graph: DependencyGraph) -> List[List[str]]
    def optimize_execution_order(self, tree: TaskTree) -> ExecutionPlan
```

#### 任务 2.2: 创建 IntelligentTaskRouter

**文件**: `pyutagent/agent/task_router.py`

**功能**:
- 基于任务特征选择最优SubAgent
- 考虑Agent负载和能力
- 支持学习历史路由决策
- 支持回退策略

**关键接口**:
```python
class IntelligentTaskRouter:
    def route(self, task: Task, agents: List[SubAgent]) -> SubAgent
    def calculate_affinity(self, task: Task, agent: SubAgent) -> float
    def record_routing_decision(self, task: Task, agent: SubAgent, success: bool)
    def get_routing_stats(self) -> Dict
```

#### 任务 2.3: 创建 ConflictResolver

**文件**: `pyutagent/agent/conflict_resolver.py`

**功能**:
- 检测资源冲突
- 检测逻辑冲突
- 解决策略（优先级、投票、人工干预）
- 冲突日志和追溯

**关键接口**:
```python
class ConflictResolver:
    def detect_conflicts(self, tasks: List[Task]) -> List[Conflict]
    async def resolve(self, conflict: Conflict) -> Resolution
    def register_strategy(self, conflict_type: str, strategy: ResolutionStrategy)
    def get_conflict_history(self) -> List[ConflictRecord]
```

### Phase 3: 主Agent集成 (预计3个任务)

#### 任务 3.1: 创建 AgentDelegationMixin

**文件**: `pyutagent/agent/delegation_mixin.py`

**功能**:
- 为主Agent提供委派能力
- 管理SubAgent生命周期
- 处理委派结果
- 支持同步/异步委派

**关键接口**:
```python
class AgentDelegationMixin:
    async def delegate_subtask(self, subtask: Subtask) -> TaskResult
    async def delegate_parallel(self, subtasks: List[Subtask]) -> List[TaskResult]
    async def delegate_with_retry(self, subtask: Subtask, max_retries: int) -> TaskResult
    def create_subagent_for_task(self, task: Task) -> DelegatingSubAgent
```

#### 任务 3.2: 创建 SharedContextManager

**文件**: `pyutagent/agent/shared_context.py`

**功能**:
- 管理主Agent与SubAgent间的上下文共享
- 支持增量更新
- 支持上下文隔离
- 支持快照和恢复

**关键接口**:
```python
class SharedContextManager:
    def create_context(self, parent_id: str, child_id: str) -> AgentContext
    def update_context(self, agent_id: str, updates: Dict) -> None
    def get_context(self, agent_id: str) -> AgentContext
    def create_snapshot(self, agent_id: str) -> ContextSnapshot
    def restore_snapshot(self, snapshot: ContextSnapshot) -> None
```

#### 任务 3.3: 创建 ResultAggregator

**文件**: `pyutagent/agent/result_aggregator.py`

**功能**:
- 聚合多个SubAgent结果
- 冲突检测和处理
- 结果验证
- 生成汇总报告

**关键接口**:
```python
class ResultAggregator:
    async def aggregate(self, results: List[TaskResult]) -> AggregatedResult
    def detect_inconsistencies(self, results: List[TaskResult]) -> List[Inconsistency]
    def validate_results(self, results: List[TaskResult]) -> ValidationResult
    def generate_summary(self, results: List[TaskResult]) -> SummaryReport
```

### Phase 4: 端到端集成 (预计2个任务)

#### 任务 4.1: 创建 DelegatingAutonomousLoop

**文件**: `pyutagent/agent/delegating_autonomous_loop.py`

**功能**:
- 扩展 EnhancedAutonomousLoop
- 在Think阶段识别可委派任务
- 在Act阶段执行委派
- 支持并行委派

**关键接口**:
```python
class DelegatingAutonomousLoop(EnhancedAutonomousLoop):
    async def _identify_delegable_subtasks(self, thought: Thought) -> List[Subtask]
    async def _delegate_subtasks(self, subtasks: List[Subtask]) -> List[TaskResult]
    async def _integrate_delegation_results(self, results: List[TaskResult]) -> Dict
```

#### 任务 4.2: 创建 SubAgentOrchestrator

**文件**: `pyutagent/agent/subagent_orchestrator.py`

**功能**:
- 统一管理SubAgent编排
- 协调多个SubAgent协作
- 监控执行状态
- 处理异常和恢复

**关键接口**:
```python
class SubAgentOrchestrator:
    async def orchestrate(self, plan: ExecutionPlan) -> OrchestrationResult
    async def execute_parallel(self, tasks: List[Task]) -> List[TaskResult]
    async def execute_sequential(self, tasks: List[Task]) -> List[TaskResult]
    def get_execution_status(self) -> OrchestrationStatus
```

### Phase 5: 测试与验证 (预计2个任务)

#### 任务 5.1: 单元测试

**文件**: `tests/unit/agent/test_subagent_enhancements.py`

**测试范围**:
- DelegatingSubAgent 功能测试
- SubAgentFactory 创建测试
- HierarchicalTaskPlanner 分解测试
- IntelligentTaskRouter 路由测试
- ConflictResolver 冲突解决测试

#### 任务 5.2: 集成测试

**文件**: `tests/integration/test_subagent_coordination.py`

**测试场景**:
- 端到端任务委派流程
- 多SubAgent并行执行
- 冲突检测和解决
- 异常恢复

## 四、文件结构

```
pyutagent/agent/
├── delegating_subagent.py      # 新增: 可委派SubAgent
├── subagent_factory.py         # 新增: SubAgent工厂
├── hierarchical_planner.py     # 新增: 层级任务规划器
├── task_router.py              # 新增: 智能任务路由
├── conflict_resolver.py        # 新增: 冲突解决器
├── delegation_mixin.py         # 新增: 委派混入类
├── shared_context.py           # 新增: 共享上下文管理
├── result_aggregator.py        # 新增: 结果聚合器
├── delegating_autonomous_loop.py # 新增: 支持委派的自主循环
├── subagent_orchestrator.py    # 新增: SubAgent编排器
├── subagents.py                # 修改: 增强SubAgentManager
└── multi_agent/
    └── agent_coordinator.py    # 修改: 集成新组件

tests/
├── unit/agent/
│   └── test_subagent_enhancements.py  # 新增
└── integration/
    └── test_subagent_coordination.py  # 新增
```

## 五、依赖关系

```
Phase 1 (SubAgent增强)
    ↓
Phase 2 (协调机制) ← 依赖 Phase 1
    ↓
Phase 3 (主Agent集成) ← 依赖 Phase 1, 2
    ↓
Phase 4 (端到端集成) ← 依赖 Phase 1, 2, 3
    ↓
Phase 5 (测试验证) ← 依赖 Phase 1-4
```

## 六、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SubAgent创建开销大 | 性能下降 | 实现Agent池化和复用 |
| 任务分解不准确 | 执行失败 | 增加LLM验证环节 |
| 并发冲突 | 数据不一致 | 实现乐观锁和冲突检测 |
| SubAgent失控 | 资源泄漏 | 实现超时和强制终止机制 |

## 七、验收标准

1. **功能验收**
   - [ ] 主Agent可以成功委派任务给SubAgent
   - [ ] SubAgent可以执行绑定的Skill
   - [ ] 多SubAgent可以并行执行并正确聚合结果
   - [ ] 冲突可以被检测和解决

2. **性能验收**
   - [ ] SubAgent创建延迟 < 100ms
   - [ ] 任务委派成功率 > 95%
   - [ ] 支持 10 个并发SubAgent

3. **代码质量**
   - [ ] 单元测试覆盖率 > 80%
   - [ ] 所有测试通过
   - [ ] 代码符合项目规范

## 八、时间估算

| Phase | 任务数 | 预计时间 |
|-------|--------|----------|
| Phase 1 | 3 | 1-2天 |
| Phase 2 | 3 | 1-2天 |
| Phase 3 | 3 | 1-2天 |
| Phase 4 | 2 | 1天 |
| Phase 5 | 2 | 1天 |
| **总计** | **13** | **5-8天** |
