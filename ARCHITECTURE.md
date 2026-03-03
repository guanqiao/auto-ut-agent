# PyUT Agent 架构文档

本文档详细描述 PyUT Agent 的系统架构，包括核心组件、数据流、模块关系和扩展点。

## 目录

- [架构概览](#架构概览)
- [核心层 (P0)](#核心层-p0)
- [增强层 (P1)](#增强层-p1)
- [协作层 (P2)](#协作层-p2)
- [高级层 (P3)](#高级层-p3)
- [数据流](#数据流)
- [扩展指南](#扩展指南)

---

## 架构概览

PyUT Agent 采用分层架构设计，从核心到高级分为四个层次：

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 (UI)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ MainWindow  │  │ChatWidget   │  │ StreamingMessageWidget  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      协作层 (P2)                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              IntegrationManager                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │    │
│  │  │ AgentCoordinator │ │ MessageBus  │  │ SharedKnowledge │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │    │
│  │  │SpecializedAgent│ │ExperienceReplay│ │ MetricsCollector │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      增强层 (P1)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │PromptOptimizer│ │ErrorKnowledge │ │   BuildToolManager      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │StaticAnalysis │ │ContextCompressor│ │   ProjectAnalyzer    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │MCPIntegration│ │               │  │                        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      核心层 (P0)                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    EnhancedAgent                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │    │
│  │  │  ReActAgent  │  │ContextManager │ │GenerationEvaluator│ │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │    │
│  │  │PartialSuccess │ │StreamingTest │ │  SmartCodeEditor │  │    │
│  │  │   Handler    │  │  Generator   │  │                 │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      基础设施层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐  │
│  │ LLMClient   │  │ WorkingMemory│ │ VectorStore │  │ Container│  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐  │
│  │JavaParser   │  │ MavenTools   │  │  ErrorRecovery │ │ RetryManager│  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心层 (P0)

核心层提供 Agent 的基础能力，对标 Cursor/Devin 的核心功能。

### 1. ReAct Agent

**文件**: `pyutagent/agent/react_agent.py`

**职责**:
- 实现 ReAct (Reasoning + Acting) 循环
- 管理 Agent 状态机
- 协调工具调用
- 处理错误恢复

**关键方法**:
```python
async def generate_tests(self, target_file: str) -> AgentResult
async def _generate_initial_tests(self, use_streaming: bool = True) -> StepResult
async def _compile_with_recovery(self) -> bool
async def _run_tests_with_recovery(self) -> bool
async def _try_recover(self, error: Exception, context: Dict) -> Dict
```

### 2. Context Manager

**文件**: `pyutagent/agent/context_manager.py`

**职责**:
- 处理大文件的上下文压缩
- 提取关键代码片段
- 生成分层摘要

**核心类**:
```python
class ContextManager:
    def compress_context(self, context: str) -> str
    def extract_key_snippets(self, code: str, query: str) -> List[CodeSnippet]
    def generate_summary(self, code: str, level: int = 1) -> str
```

### 3. Generation Evaluator

**文件**: `pyutagent/agent/generation_evaluator.py`

**职责**:
- 预编译代码质量评估
- 6维度评分系统
- 提供改进建议

**评估维度**:
1. 语法正确性 (Syntax Correctness)
2. 语义完整性 (Semantic Completeness)
3. 测试覆盖率 (Test Coverage)
4. 代码风格 (Code Style)
5. 最佳实践 (Best Practices)
6. 可维护性 (Maintainability)

### 4. Partial Success Handler

**文件**: `pyutagent/agent/partial_success_handler.py`

**职责**:
- 处理部分成功的测试
- 增量修复失败的测试方法
- 管理测试片段

### 5. Streaming Test Generator

**文件**: `pyutagent/agent/streaming.py`

**职责**:
- 流式代码生成
- 实时预览
- 支持用户中断

### 6. Smart Code Editor

**文件**: `pyutagent/tools/smart_editor.py`

**职责**:
- Search/Replace 精确编辑
- Unified diff 格式支持
- 模糊匹配和增量修复

---

## 增强层 (P1)

增强层提供对标顶级 Agent 的高级功能。

### 1. Prompt Optimizer

**文件**: `pyutagent/agent/prompt_optimizer.py`

**职责**:
- 模型特定的提示词优化
- A/B 测试框架
- Few-shot 示例选择

**支持的模型**:
- GPT-4 / GPT-4o
- Claude 3 (Opus/Sonnet/Haiku)
- DeepSeek
- Ollama 本地模型

### 2. Error Knowledge Base

**文件**: `pyutagent/core/error_knowledge_base.py`

**职责**:
- 错误模式持久化存储 (SQLite)
- 相似度匹配
- 解决方案推荐

### 3. Build Tool Manager

**文件**: `pyutagent/tools/build_tool_manager.py`

**职责**:
- 自动检测构建工具
- 支持 Maven / Gradle / Bazel
- 统一命令接口

### 4. Static Analysis Manager

**文件**: `pyutagent/tools/static_analysis_manager.py`

**职责**:
- SpotBugs 集成
- PMD 集成
- 静态分析报告解析

### 5. MCP Integration

**文件**: `pyutagent/tools/mcp_integration.py`

**职责**:
- Model Context Protocol 客户端
- 工具适配器
- 外部工具扩展

### 6. Context Compressor

**文件**: `pyutagent/memory/context_compressor.py`

**职责**:
- 相关性评分
- 智能上下文压缩
- 大项目支持

### 7. Project Analyzer

**文件**: `pyutagent/tools/project_analyzer.py`

**职责**:
- 项目结构分析
- 依赖关系分析
- 多文件协调

---

## 协作层 (P2)

协作层实现多智能体协作系统。

### 1. Integration Manager

**文件**: `pyutagent/agent/integration_manager.py`

**职责**:
- 组件生命周期管理
- 依赖解析
- 健康监控
- 事件路由

**初始化顺序**:
1. MessageBus
2. SharedKnowledgeBase
3. ExperienceReplay
4. MetricsCollector
5. AgentCoordinator
6. EnhancedAgent

### 2. Agent Coordinator

**文件**: `pyutagent/agent/multi_agent/agent_coordinator.py`

**职责**:
- 智能体注册和管理
- 任务分配策略
- 任务调度
- 结果聚合

**任务分配策略**:
- Round Robin (轮询)
- Capability Match (能力匹配)
- Load Balanced (负载均衡)
- Priority Based (优先级)

### 3. Specialized Agent

**文件**: `pyutagent/agent/multi_agent/specialized_agent.py`

**职责**:
- 专业化智能体基类
- 能力系统
- 任务执行框架
- 心跳机制

**智能体类型**:
- TestDesignerAgent (测试设计)
- TestImplementerAgent (测试实现)
- TestReviewerAgent (测试审查)
- ErrorFixerAgent (错误修复)

### 4. Message Bus

**文件**: `pyutagent/agent/multi_agent/message_bus.py`

**职责**:
- 异步消息总线
- 点对点通信
- 广播消息
- 消息订阅

### 5. Shared Knowledge Base

**文件**: `pyutagent/agent/multi_agent/shared_knowledge.py`

**职责**:
- 知识持久化
- 标签系统
- 相似度搜索
- 使用统计

### 6. Experience Replay

**文件**: `pyutagent/agent/multi_agent/shared_knowledge.py`

**职责**:
- 经验存储
- 采样学习
- 成功/失败记录

### 7. Metrics Collector

**文件**: `pyutagent/core/metrics.py`

**职责**:
- 操作计时
- LLM 调用统计
- 错误追踪
- 报告生成

---

## 高级层 (P3)

高级层提供企业级能力（全部实现）。

### 1. Error Predictor

**文件**: `pyutagent/core/error_predictor.py`

**职责**:
- 编译前错误预测
- 12种错误类型分类
- 4级严重度评估
- 测试失败预测

**核心类**:
```python
class ErrorPredictor:
    def predict_compilation_errors(self, code: str, file_path: Optional[str] = None) -> PredictionResult
    def predict_test_failures(self, test_code: str, test_info: Dict[str, Any]) -> PredictionResult
    def suggest_fix(self, predicted_error: PredictedError, code: str) -> Optional[Dict[str, Any]]
```

### 2. Adaptive Strategy Manager

**文件**: `pyutagent/core/adaptive_strategy.py`

**职责**:
- 动态策略选择
- 上下文感知
- 探索vs利用（ε-贪婪算法）
- 策略效果跟踪

**核心类**:
```python
class AdaptiveStrategyManager:
    def select_strategy(self, error_category: ErrorCategory, available_strategies: List[RecoveryStrategy], context: Dict[str, Any]) -> StrategySelection
    def record_outcome(self, strategy_name: str, error_category: ErrorCategory, success: bool, execution_time: float, context: Dict[str, Any])
```

### 3. Sandbox Executor

**文件**: `pyutagent/core/sandbox_executor.py`

**职责**:
- 沙箱代码执行
- 3级安全控制（严格/中等/宽松）
- 文件系统隔离
- 网络限制
- 资源限制（CPU、内存、磁盘）

**核心类**:
```python
class SandboxExecutor:
    async def execute_sandboxed(self, code: str, class_name: str, method_name: Optional[str] = None, args: List[Any] = None) -> ExecutionResult
    def _analyze_security(self, code: str) -> SecurityReport
```

### 4. User Interaction Handler

**文件**: `pyutagent/agent/user_interaction.py`

**职责**:
- 修复建议展示
- 交互式确认
- 策略选择
- 用户偏好学习

**核心类**:
```python
class UserInteractionHandler:
    def display_suggestion(self, suggestion: RepairSuggestion, config: DisplayConfig = None) -> str
    async def request_confirmation(self, suggestion: RepairSuggestion, context: Dict[str, Any], auto_decide: bool = False) -> Tuple[UserChoice, Optional[str]]
```

### 5. Smart Code Analyzer

**文件**: `pyutagent/core/smart_analyzer.py`

**职责**:
- AST分析
- 语义分析
- 依赖关系图
- 影响分析
- 智能代码搜索

**核心类**:
```python
class SmartCodeAnalyzer:
    async def analyze_project(self, project_path: str) -> Dict[str, Any]
    def search_code(self, query: str, top_k: int = 10) -> List[CodeSearchResult]
    def analyze_change_impact(self, entity_id: str) -> ImpactAnalysisResult
```

---

## 数据流

### 测试生成流程

```
用户输入
    │
    ▼
┌─────────────────┐
│  EnhancedAgent  │
│  (集成入口)      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌─────────────┐
│Multi- │ │  Single     │
│Agent  │ │  Agent      │
└───┬───┘ └──────┬──────┘
    │            │
    ▼            ▼
┌─────────────────────────┐
│   AgentCoordinator      │
│   (任务分配)             │
└────────┬────────────────┘
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│Designer│ │Implementer│ │Reviewer│
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └─────────┼─────────┘
              ▼
    ┌─────────────────┐
    │   ReActAgent    │
    │   (核心执行)     │
    └────────┬────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│Context │ │Prompt │ │Stream │
│Manager │ │Optimizer│ │Generator│
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └─────────┼─────────┘
              ▼
    ┌─────────────────┐
    │   LLM Client    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   生成结果       │
    └─────────────────┘
```

### 错误恢复流程

```
错误发生
    │
    ▼
┌─────────────────┐
│ ErrorRecovery   │
│ Manager         │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌─────────────┐
│Error    │ │ Strategy    │
│Knowledge│ │ Optimizer   │
│Base     │ │             │
└───┬─────┘ └──────┬──────┘
    │              │
    └──────┬───────┘
           ▼
    ┌─────────────────┐
    │ ParallelRecovery │
    │ (并行尝试)       │
    └────────┬────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│Strategy│ │Strategy│ │Strategy│
│   A    │ │   B    │ │   C    │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    └─────────┼─────────┘
              ▼
    ┌─────────────────┐
    │  恢复结果        │
    │  (记录到经验回放) │
    └─────────────────┘
```

---

## 扩展指南

### 添加新的 Specialized Agent

```python
from pyutagent.agent.multi_agent import SpecializedAgent, AgentCapability, AgentTask

class MyCustomAgent(SpecializedAgent):
    def __init__(self, agent_id: str, message_bus, knowledge_base):
        super().__init__(
            agent_id=agent_id,
            capabilities={AgentCapability.TEST_DESIGN, AgentCapability.MOCK_GENERATION},
            message_bus=message_bus,
            knowledge_base=knowledge_base
        )
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        # 实现任务逻辑
        return {"success": True, "output": "..."}
```

### 添加新的工具

```python
from pyutagent.agent.actions import Action

class MyCustomTool(Action):
    name = "my_custom_tool"
    description = "My custom tool description"
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        # 实现工具逻辑
        return {"success": True, "result": "..."}
```

### 添加新的提示词优化策略

```python
from pyutagent.agent.prompt_optimizer import PromptOptimizer, ModelType

class MyPromptOptimizer(PromptOptimizer):
    def optimize_for_my_model(self, base_prompt: str) -> str:
        # 实现优化逻辑
        return optimized_prompt
```

---

## 配置参考

### EnhancedAgentConfig

```python
from pyutagent.agent import EnhancedAgentConfig, CompressionStrategy

config = EnhancedAgentConfig(
    # P0 Configuration
    context_max_tokens=8000,
    context_target_tokens=6000,
    context_strategy=CompressionStrategy.HYBRID,
    
    # P1 Configuration
    enable_prompt_optimization=True,
    enable_ab_testing=False,
    ab_test_id=None,
    
    # P2 Configuration
    enable_multi_agent=True,
    multi_agent_workers=3,
    task_allocation_strategy="capability_match",
    
    # Performance
    enable_metrics=True,
    metrics_report_interval=300,
    
    # Model
    model_name="gpt-4"
)
```

---

## 性能考虑

### 内存管理

- ContextManager 自动压缩大文件上下文
- VectorStore 使用 sqlite-vec，支持大规模数据
- ExperienceReplay 有容量限制，自动淘汰旧数据

### 并发处理

- MessageBus 支持异步消息处理
- ParallelRecovery 并行尝试多种恢复策略
- Multi-agent 支持并发任务执行

### 缓存策略

- ErrorKnowledgeBase 缓存常见错误模式
- SharedKnowledgeBase 缓存知识项
- MetricsCollector 缓存性能数据

---

## 监控和调试

### 日志级别

- `DEBUG`: 详细执行信息
- `INFO`: 关键操作记录
- `WARNING`: 潜在问题
- `ERROR`: 错误和异常

### 性能报告

```python
from pyutagent.agent import get_integration_manager

manager = get_integration_manager("/path/to/project")
agent = manager.create_enhanced_agent(llm_client, working_memory)

# 获取性能统计
stats = agent.get_enhanced_stats()
print(stats)
```

### 健康检查

```python
# 获取系统健康状态
health = manager.get_system_health()
print(f"Overall Health: {health['overall_health']}")
print(f"Health Score: {health['health_score']}")
```

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-03-01 | 初始架构，基础 ReAct Agent |
| v2.0 | 2026-03-01 | P0 核心能力：流式生成、增量编辑、上下文管理 |
| v3.0 | 2026-03-01 | P1 增强能力：提示词优化、错误学习、多构建工具 |
| v4.0 | 2026-03-01 | P2 协作能力：多智能体、消息总线、性能监控 |
| v5.0 | 2026-03-01 | P3 高级能力：错误预测、自适应策略、沙箱执行、用户交互、智能分析 |

---

## 参考文档

- [README.md](README.md) - 项目概览和使用指南
- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) - 改进计划详情
- [API Documentation](docs/api/) - API 详细文档
