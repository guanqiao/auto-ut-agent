# PyUT Agent 主流程优化任务清单

## Phase 1: 架构重构 (优先级: 高)

### 1.1 EnhancedAgent 拆分

- [ ] **T1.1.1** 创建 Capability 基类和注册机制
  - 创建 `pyutagent/agent/capabilities/base.py`
  - 定义 `Capability` 抽象基类
  - 实现 `CapabilityRegistry` 注册表
  - 编写单元测试

- [ ] **T1.1.2** 拆分 P0 能力模块
  - 创建 `pyutagent/agent/capabilities/p0/` 目录
  - 迁移 ContextManager 相关代码
  - 迁移 GenerationEvaluator 相关代码
  - 迁移 PartialSuccessHandler 相关代码
  - 编写单元测试

- [ ] **T1.1.3** 拆分 P1 能力模块
  - 创建 `pyutagent/agent/capabilities/p1/` 目录
  - 迁移 PromptOptimizer 相关代码
  - 迁移 ErrorLearner 相关代码
  - 迁移 BuildToolManager 相关代码
  - 编写单元测试

- [ ] **T1.1.4** 拆分 P2 能力模块
  - 创建 `pyutagent/agent/capabilities/p2/` 目录
  - 迁移 MultiAgent 相关代码
  - 迁移 MessageBus 相关代码
  - 迁移 SharedKnowledge 相关代码
  - 编写单元测试

- [ ] **T1.1.5** 拆分 P3 能力模块
  - 创建 `pyutagent/agent/capabilities/p3/` 目录
  - 迁移 ErrorPredictor 相关代码
  - 迁移 AdaptiveStrategy 相关代码
  - 迁移 SandboxExecutor 相关代码
  - 编写单元测试

- [ ] **T1.1.6** 拆分 P4 能力模块
  - 创建 `pyutagent/agent/capabilities/p4/` 目录
  - 迁移 SelfReflection 相关代码
  - 迁移 KnowledgeGraph 相关代码
  - 迁移 PatternLibrary 相关代码
  - 编写单元测试

- [ ] **T1.1.7** 重构 EnhancedAgent
  - 精简 EnhancedAgent 类
  - 使用 CapabilityRegistry 加载能力
  - 保持 API 兼容性
  - 编写集成测试

### 1.2 统一依赖注入

- [ ] **T1.2.1** 扩展 Container 功能
  - 添加生命周期管理（singleton/transient）
  - 添加依赖解析功能
  - 添加循环依赖检测
  - 编写单元测试

- [ ] **T1.2.2** 迁移组件注册
  - 创建组件注册配置文件
  - 将所有组件注册到 Container
  - 更新组件获取方式
  - 编写单元测试

- [ ] **T1.2.3** 更新 Agent 初始化
  - 使用 Container 获取组件
  - 移除直接实例化代码
  - 保持向后兼容
  - 编写集成测试

### 1.3 统一重试机制

- [ ] **T1.3.1** 创建重试装饰器
  - 创建 `pyutagent/agent/execution/retry.py`
  - 实现 `@with_retry` 装饰器
  - 支持多种退避策略
  - 编写单元测试

- [ ] **T1.3.2** 创建 SmartRetryPolicy
  - 实现基于错误类型的重试策略
  - 支持自定义重试条件
  - 编写单元测试

- [ ] **T1.3.3** 重构现有重试代码
  - 更新 `execute_with_recovery`
  - 更新 `compile_with_recovery`
  - 更新 `run_tests_with_recovery`
  - 编写集成测试

---

## Phase 2: 性能优化 (优先级: 中)

### 2.1 增量编译

- [ ] **T2.1.1** 创建 IncrementalCompiler
  - 创建 `pyutagent/agent/execution/compiler.py`
  - 实现文件哈希计算
  - 实现缓存编译结果
  - 编写单元测试

- [ ] **T2.1.2** 集成到执行流程
  - 更新 StepExecutor 使用增量编译
  - 添加强制重新编译选项
  - 编写集成测试

- [ ] **T2.1.3** 性能基准测试
  - 创建编译性能测试
  - 对比优化前后性能
  - 记录结果

### 2.2 LLM 调用优化

- [ ] **T2.2.1** 创建 OptimizedLLMClient
  - 创建 `pyutagent/llm/optimized_client.py`
  - 集成 MultiLevelCache
  - 集成 SmartClusterer
  - 编写单元测试

- [ ] **T2.2.2** 实现相似问题复用
  - 实现语义相似度计算
  - 实现响应适配逻辑
  - 编写单元测试

- [ ] **T2.2.3** 集成到 Agent
  - 更新 Agent 使用 OptimizedLLMClient
  - 编写集成测试

- [ ] **T2.2.4** 性能基准测试
  - 创建 LLM 调用性能测试
  - 对比优化前后性能
  - 记录结果

### 2.3 流程优化

- [ ] **T2.3.1** 优化覆盖率分析
  - 实现增量覆盖率检查
  - 跳过已覆盖代码区域
  - 编写单元测试

- [ ] **T2.3.2** 优化测试执行
  - 实现增量测试执行
  - 只运行新添加的测试
  - 编写单元测试

---

## Phase 3: 代码质量 (优先级: 低)

### 3.1 结构化日志

- [ ] **T3.1.1** 创建结构化日志配置
  - 定义日志格式规范
  - 创建日志配置文件
  - 编写单元测试

- [ ] **T3.1.2** 更新现有日志
  - 更新 StepExecutor 日志
  - 更新 FeedbackLoopExecutor 日志
  - 更新 AgentRecoveryManager 日志
  - 编写集成测试

### 3.2 指标收集

- [ ] **T3.2.1** 扩展 MetricsCollector
  - 添加更多指标类型
  - 支持 Prometheus 格式导出
  - 编写单元测试

- [ ] **T3.2.2** 添加指标装饰器
  - 实现 `@metrics.time` 装饰器
  - 实现 `@metrics.count` 装饰器
  - 实现 `@metrics.track` 装饰器
  - 编写单元测试

- [ ] **T3.2.3** 应用指标装饰器
  - 更新关键方法添加指标
  - 创建性能分析报告
  - 编写集成测试

### 3.3 文档更新

- [ ] **T3.3.1** 更新架构文档
  - 更新 ARCHITECTURE.md
  - 添加新组件说明
  - 更新依赖关系图

- [ ] **T3.3.2** 更新 API 文档
  - 更新 Sphinx 文档
  - 添加新接口说明
  - 添加使用示例

---

## Phase 4: 测试和验收 (持续)

### 4.1 单元测试

- [ ] **T4.1.1** 确保测试覆盖率 > 85%
- [ ] **T4.1.2** 所有新代码有对应测试
- [ ] **T4.1.3** 运行完整测试套件

### 4.2 集成测试

- [ ] **T4.2.1** 端到端测试
- [ ] **T4.2.2** 性能回归测试
- [ ] **T4.2.3** 兼容性测试

### 4.3 验收

- [ ] **T4.3.1** 功能验收
- [ ] **T4.3.2** 性能验收
- [ ] **T4.3.3** 质量验收

---

## 任务依赖关系

```
T1.1.1 ──┬── T1.1.2 ──┬── T1.1.7
         │            │
         ├── T1.1.3 ──┤
         │            │
         ├── T1.1.4 ──┤
         │            │
         ├── T1.1.5 ──┤
         │            │
         └── T1.1.6 ──┘

T1.2.1 ─── T1.2.2 ─── T1.2.3

T1.3.1 ─── T1.3.2 ─── T1.3.3

T2.1.1 ─── T2.1.2 ─── T2.1.3

T2.2.1 ─── T2.2.2 ─── T2.2.3 ─── T2.2.4

T3.1.1 ─── T3.1.2

T3.2.1 ─── T3.2.2 ─── T3.2.3
```

---

## 时间估算

| Phase | 任务数 | 预估时间 |
|-------|--------|----------|
| Phase 1 | 17 | 5-7 天 |
| Phase 2 | 11 | 3-4 天 |
| Phase 3 | 8 | 2-3 天 |
| Phase 4 | 9 | 2-3 天 |
| **总计** | **45** | **12-17 天** |

---

## 里程碑

1. **M1** - Phase 1 完成：架构重构完成，EnhancedAgent 拆分
2. **M2** - Phase 2 完成：性能优化完成，性能指标达标
3. **M3** - Phase 3 完成：代码质量改进完成
4. **M4** - Phase 4 完成：所有验收通过，可发布
