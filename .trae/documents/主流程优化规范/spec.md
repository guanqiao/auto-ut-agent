# PyUT Agent 主流程优化规范

## 一、项目背景

PyUT Agent 是一个 AI 驱动的 Java 单元测试生成器，经过分析发现以下问题需要优化：

### 1.1 当前问题

| 问题类型 | 具体问题 | 影响 |
|---------|---------|------|
| 架构问题 | EnhancedAgent 类过大（1362行），职责过多 | 难以维护和测试 |
| 架构问题 | 组件依赖复杂，初始化顺序不清晰 | 容易出现隐式依赖问题 |
| 代码质量 | 重试逻辑分散在多处 | 代码重复，行为不一致 |
| 性能问题 | 缺乏增量编译检查 | 浪费编译时间 |
| 性能问题 | LLM 调用未充分优化 | 成本高，响应慢 |

### 1.2 优化目标

1. **降低复杂度**: 将 EnhancedAgent 拆分为更小的组件
2. **统一依赖管理**: 使用依赖注入容器统一管理组件生命周期
3. **减少重复代码**: 提取通用的重试和恢复逻辑
4. **提升性能**: 优化编译、测试、LLM 调用流程
5. **提高可测试性**: 改进架构以支持更好的单元测试

---

## 二、功能规范

### 2.1 架构重构

#### 2.1.1 EnhancedAgent 拆分

**现状：**
```python
class EnhancedAgent(ReActAgent):
    # P0-P4 所有组件都在一个类中初始化
    def _init_p3_components(self): ...
    def _init_p4_components(self): ...
    # 1362 行代码
```

**目标：**
```python
class Agent:
    """精简的 Agent 核心"""
    def __init__(self, config: AgentConfig, container: Container):
        self.capabilities = CapabilityRegistry(container)
        self.capabilities.load_enabled(config)
```

**验收标准：**
- [ ] EnhancedAgent 代码量减少 50% 以上
- [ ] 每个组件独立可测试
- [ ] 组件可按需加载（配置开关生效）

#### 2.1.2 统一依赖注入

**现状：**
```python
# 分散在各处的组件创建
self.error_predictor = ErrorPredictor()
self.strategy_manager = AdaptiveStrategyManager()
# ...
```

**目标：**
```python
# 统一在 Container 中注册
container.register_singleton(ErrorPredictor)
container.register_singleton(AdaptiveStrategyManager)
# 自动解析依赖
predictor = container.resolve(ErrorPredictor)
```

**验收标准：**
- [ ] 所有组件通过 Container 获取
- [ ] 依赖关系在注册时声明
- [ ] 支持组件生命周期管理（singleton/transient）

### 2.2 重试机制统一

#### 2.2.1 统一重试装饰器

**现状：**
```python
# execute_with_recovery 有重试逻辑
# compile_with_recovery 有独立重试逻辑
# run_tests_with_recovery 有独立重试逻辑
# 代码重复约 200 行
```

**目标：**
```python
@with_retry(
    max_attempts=3,
    backoff=BackoffStrategy.EXPONENTIAL,
    on_retry=on_retry_callback
)
async def compile_tests(self) -> StepResult:
    ...
```

**验收标准：**
- [ ] 重试逻辑代码减少 70% 以上
- [ ] 所有重试使用统一配置
- [ ] 支持自定义重试策略

#### 2.2.2 智能重试策略

**目标：**
```python
class SmartRetryPolicy:
    """根据错误类型选择重试策略"""
    def should_retry(self, error: Exception, attempt: int) -> bool:
        if self._is_network_error(error):
            return attempt < 5  # 网络错误多试几次
        if self._is_code_error(error):
            return attempt < 2  # 代码错误少试
        return False
```

**验收标准：**
- [ ] 网络错误自动重试
- [ ] 代码错误使用 LLM 分析后重试
- [ ] 重复失败自动跳过

### 2.3 性能优化

#### 2.3.1 增量编译检查

**现状：**
```python
# 每次迭代都完整编译
async def _iteration_compile(self):
    compile_success = await self.step_executor.compile_with_recovery()
```

**目标：**
```python
class IncrementalCompiler:
    """增量编译器"""
    def __init__(self):
        self._file_hashes: Dict[str, str] = {}
    
    async def compile_if_changed(self, test_file: str) -> CompileResult:
        current_hash = self._compute_hash(test_file)
        if self._file_hashes.get(test_file) == current_hash:
            return CompileResult(cached=True)  # 跳过编译
        result = await self._compile(test_file)
        self._file_hashes[test_file] = current_hash
        return result
```

**验收标准：**
- [ ] 相同代码不重复编译
- [ ] 编译时间减少 30% 以上
- [ ] 支持强制重新编译选项

#### 2.3.2 LLM 调用优化

**现状：**
```python
# 每次生成/修复都直接调用 LLM
response = await self.agent_core.llm_client.agenerate(prompt)
```

**目标：**
```python
class OptimizedLLMClient:
    """优化的 LLM 客户端"""
    def __init__(self, base_client: LLMClient):
        self.base_client = base_client
        self.cache = MultiLevelCache()
        self.clusterer = SmartClusterer()
    
    async def generate(self, prompt: str, context: Dict) -> str:
        # 1. 检查缓存
        cached = self.cache.get(prompt)
        if cached:
            return cached
        
        # 2. 检查相似问题
        similar = self.clusterer.find_similar(prompt)
        if similar:
            return self._adapt_response(similar.response, context)
        
        # 3. 调用 LLM
        response = await self.base_client.generate(prompt)
        self.cache.set(prompt, response)
        return response
```

**验收标准：**
- [ ] LLM 调用次数减少 20% 以上
- [ ] 相似问题复用率 > 30%
- [ ] 平均响应时间减少 15%

### 2.4 代码质量改进

#### 2.4.1 结构化日志

**现状：**
```python
logger.info(f"[StepExecutor] Test generation complete - File: {file}, Coverage: {coverage}")
```

**目标：**
```python
logger.info(
    "test_generation_complete",
    extra={
        "file": file,
        "coverage": coverage,
        "iterations": iterations,
        "time_elapsed": elapsed,
        "llm_calls": llm_calls
    }
)
```

**验收标准：**
- [ ] 所有日志使用结构化格式
- [ ] 支持日志聚合和分析
- [ ] 关键指标可追踪

#### 2.4.2 指标收集完善

**目标：**
```python
@metrics.time("generate_tests")
@metrics.count("llm.calls")
@metrics.track("coverage.improvement")
async def generate_tests(self, target_file: str) -> AgentResult:
    ...
```

**验收标准：**
- [ ] 所有关键操作有指标
- [ ] 支持导出 Prometheus 格式
- [ ] 提供性能分析报告

---

## 三、非功能规范

### 3.1 性能要求

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| 单文件生成时间 | ~60s | ~45s |
| 编译时间占比 | ~30% | ~20% |
| LLM 调用次数/文件 | ~5次 | ~4次 |
| 内存占用峰值 | ~500MB | ~400MB |

### 3.2 可维护性要求

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| EnhancedAgent 行数 | 1362 | <700 |
| 代码重复率 | ~15% | <10% |
| 测试覆盖率 | ~80% | >85% |
| 循环复杂度 | 高 | 中 |

### 3.3 兼容性要求

- 保持现有 API 接口不变
- 配置文件格式向后兼容
- 支持渐进式迁移

---

## 四、技术方案

### 4.1 架构重构方案

```
pyutagent/
├── agent/
│   ├── core/
│   │   ├── agent.py              # 精简的 Agent 核心
│   │   ├── state.py              # 状态管理
│   │   └── context.py            # 上下文管理
│   ├── capabilities/             # 能力模块（新增）
│   │   ├── __init__.py
│   │   ├── base.py               # Capability 基类
│   │   ├── p0/                   # P0 能力
│   │   ├── p1/                   # P1 能力
│   │   ├── p2/                   # P2 能力
│   │   ├── p3/                   # P3 能力
│   │   └── p4/                   # P4 能力
│   ├── execution/
│   │   ├── retry.py              # 统一重试机制
│   │   ├── compiler.py           # 增量编译
│   │   └── executor.py           # 步骤执行器
│   └── ...
```

### 4.2 依赖关系

```
Agent
  ├── Container (依赖注入)
  │     ├── ErrorClassifier
  │     ├── ErrorLearner
  │     ├── StrategyOptimizer
  │     └── ...
  ├── CapabilityRegistry
  │     ├── P0Capabilities
  │     ├── P1Capabilities
  │     ├── P2Capabilities
  │     ├── P3Capabilities
  │     └── P4Capabilities
  └── ExecutionEngine
        ├── RetryPolicy
        ├── IncrementalCompiler
        └── OptimizedLLMClient
```

---

## 五、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 重构引入新 bug | 高 | 高 | TDD 开发，保持测试覆盖 |
| API 不兼容 | 中 | 高 | 保持接口不变，渐进迁移 |
| 性能优化效果不达预期 | 中 | 中 | 先做基准测试，逐步优化 |
| 组件依赖循环 | 低 | 高 | 使用依赖分析工具检查 |

---

## 六、验收标准总览

### 6.1 功能验收

- [ ] EnhancedAgent 拆分完成，代码量减少 50%
- [ ] 所有组件通过 Container 获取
- [ ] 重试逻辑统一，代码减少 70%
- [ ] 增量编译实现，编译时间减少 30%
- [ ] LLM 调用优化，调用次数减少 20%

### 6.2 质量验收

- [ ] 测试覆盖率 > 85%
- [ ] 所有新代码有单元测试
- [ ] 无循环依赖
- [ ] 代码复杂度降低

### 6.3 性能验收

- [ ] 单文件生成时间 < 45s
- [ ] 内存占用 < 400MB
- [ ] LLM 调用次数/文件 < 4次

---

## 七、参考资料

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - 现有架构文档
- [README.md](../../README.md) - 项目说明
- [pyutagent/agent/react_agent.py](../../pyutagent/agent/react_agent.py) - 核心实现
- [pyutagent/agent/enhanced_agent.py](../../pyutagent/agent/enhanced_agent.py) - 增强实现
