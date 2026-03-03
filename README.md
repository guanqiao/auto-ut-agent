# PyUT Agent

AI 驱动的 Java 单元测试生成器，基于 Agent 架构，支持对话式交互。对标 Cursor/Devin/Cline 等顶级 Coding Agent，具备流式生成、增量编辑、错误学习、多智能体协作等高级能力。

## 特性

### 核心能力 (P0)
- 🤖 **Agent 架构**: 基于 LangChain 的 ReAct Agent，支持工具调用和规划
- 💬 **对话式 UI**: PyQt6 构建的图形界面，支持自然语言交互
- 🧠 **记忆系统**: 多层记忆（工作/短期/长期/向量），持续学习优化
- 🔍 **向量检索**: sqlite-vec 存储和检索相似测试模式
- ⏸️ **暂停/恢复**: 随时暂停生成任务，保存状态后可恢复
- 📊 **覆盖率分析**: 集成 JaCoCo，实时显示覆盖率报告
- 🔧 **LLM 配置**: 支持 OpenAI、Anthropic、DeepSeek、Ollama 等多种提供商

### 增强能力 (P1)
- 📝 **流式代码生成**: 实时流式输出，支持用户中断和预览
- ✏️ **智能增量编辑**: Search/Replace 精确修改，unified diff 格式支持
- 📚 **错误模式学习**: 从历史错误中学习，持久化存储和推荐最佳策略
- 🎯 **提示词优化**: 模型特定的提示词优化，A/B 测试框架
- 🔧 **多构建工具**: 支持 Maven、Gradle、Bazel 自动检测
- 📊 **静态分析集成**: SpotBugs、PMD 静态分析集成

### 高级能力 (P2)
- 👥 **多智能体协作**: 专业化智能体（设计/实现/审查/修复）协作
- 🧩 **上下文智能压缩**: 大文件处理，关键片段提取，分层摘要
- 🔗 **多文件协调**: 跨文件理解和修改，依赖分析
- 🔄 **并行恢复**: 多路径并行尝试错误恢复
- 📈 **性能监控**: 全面的性能指标收集和报告

### 企业级能力 (P3)
- 🔮 **错误预测**: 编译前预测潜在错误，12种错误类型分类
- 🎛️ **自适应策略**: 根据历史动态调整策略，ε-贪婪算法
- 🛡️ **工具沙箱**: 安全沙箱隔离执行，3级安全控制
- 💾 **检查点恢复**: 断点续传，状态持久化
- 🧠 **智能代码分析**: 语义分析、依赖图、影响分析
- 🎯 **用户交互**: 交互式修复建议和确认

## 安装

### 环境要求
- Python 3.9+
- Maven 3.6+
- Java 11+

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd auto-ut-agent

# 安装依赖
pip install -e .

# 或者开发模式安装
pip install -e ".[dev]"
```

## 运行

```bash
# 启动应用
pyutagent

# 或者
python -m pyutagent
```

## 使用指南

### 1. 配置 LLM
- 点击菜单 `设置 -> LLM 配置`
- 选择提供商（OpenAI、Anthropic、DeepSeek、Ollama）
- 输入 API Key 和模型名称
- 点击 `测试连接` 验证配置
- 支持参数：Temperature、Max Tokens、Timeout、Retries

### 2. 打开项目
- 点击菜单 `文件 -> 打开项目`
- 选择一个 Maven 项目目录（包含 pom.xml）

### 3. 生成测试
- 在左侧文件树中选择一个 Java 文件
- 在对话区域输入: "生成 UserService 的测试"
- 或使用快捷键 `Ctrl+G`

### 4. 控制生成过程
- **暂停**: 输入 "暂停" 或点击暂停按钮
- **继续**: 输入 "继续" 恢复生成
- **查看状态**: 输入 "状态" 查看当前进度

### 5. 查看结果
- 生成的测试文件保存在 `src/test/java` 目录
- 覆盖率报告在右侧进度面板显示

## 支持的 LLM 提供商

| 提供商 | 默认 Endpoint | 推荐模型 |
|--------|--------------|---------|
| OpenAI | https://api.openai.com/v1 | gpt-4, gpt-4-turbo, gpt-3.5-turbo |
| Anthropic | https://api.anthropic.com/v1 | claude-3-opus, claude-3-sonnet |
| DeepSeek | https://api.deepseek.com/v1 | deepseek-chat, deepseek-coder |
| Ollama | http://localhost:11434/v1 | llama2, codellama, mistral |
| Custom | 自定义 | 任意兼容 OpenAI API 的模型 |

## 配置

### 环境变量

```bash
PYUT_LLM_PROVIDER=openai
PYUT_LLM_API_KEY=your-api-key
PYUT_LLM_MODEL=gpt-4
PYUT_TARGET_COVERAGE=0.8
PYUT_MAX_ITERATIONS=10
```

### 配置文件

配置保存在 `~/.pyutagent/config.json`：

```json
{
  "llm": {
    "provider": "openai",
    "endpoint": "https://api.openai.com/v1",
    "api_key": "sk-...",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 4096,
    "timeout": 300,
    "max_retries": 5
  }
}
```

## 测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/unit/memory/
pytest tests/unit/tools/

# 带覆盖率报告
pytest --cov=pyutagent --cov-report=html
```

## 项目结构

```
pyutagent/
├── agent/                    # Agent 核心
│   ├── base_agent.py         # 基础 Agent
│   ├── react_agent.py        # ReAct Agent
│   ├── enhanced_agent.py     # 增强 Agent (P0/P1/P2/P3集成)
│   ├── integration_manager.py # 组件生命周期管理
│   ├── multi_agent/          # 多智能体系统 (P2)
│   │   ├── agent_coordinator.py    # 智能体协调器
│   │   ├── specialized_agent.py    # 专业化智能体
│   │   ├── message_bus.py          # 消息总线
│   │   └── shared_knowledge.py     # 共享知识库
│   ├── context_manager.py    # 上下文管理 (P0)
│   ├── generation_evaluator.py # 代码质量评估 (P0)
│   ├── partial_success_handler.py # 部分成功处理 (P0)
│   ├── prompt_optimizer.py   # 提示词优化 (P1)
│   ├── streaming.py          # 流式生成 (P0)
│   ├── smart_editor.py       # 智能编辑器 (P0)
│   ├── user_interaction.py   # 用户交互 (P3)
│   └── tool_validator.py     # 工具验证 (P3)
├── memory/                   # 记忆系统
│   ├── vector_store.py       # sqlite-vec 向量存储
│   ├── working_memory.py     # 工作记忆
│   ├── short_term_memory.py  # 短期记忆
│   └── context_compressor.py # 上下文压缩 (P1)
├── tools/                    # 工具
│   ├── java_parser.py        # Java 代码解析
│   ├── maven_tools.py        # Maven 工具
│   ├── build_tool_manager.py # 多构建工具支持 (P1)
│   ├── static_analysis_manager.py # 静态分析 (P1)
│   ├── smart_editor.py       # 智能编辑器 (P0)
│   ├── project_analyzer.py   # 项目分析器 (P1)
│   └── mcp_integration.py    # MCP 集成 (P1)
├── core/                     # 核心功能
│   ├── metrics.py            # 性能监控 (P2)
│   ├── error_recovery.py     # 错误恢复
│   ├── error_learner.py      # 错误学习 (P1)
│   ├── error_knowledge_base.py # 错误知识库 (P1)
│   ├── error_predictor.py    # 错误预测 (P3)
│   ├── adaptive_strategy.py  # 自适应策略 (P3)
│   ├── sandbox_executor.py   # 沙箱执行器 (P3)
│   ├── smart_analyzer.py     # 智能代码分析 (P3)
│   ├── code_interpreter.py   # 代码解释器 (竞争力)
│   ├── refactoring_engine.py # 重构引擎 (竞争力)
│   ├── test_quality_analyzer.py # 质量分析器 (竞争力)
│   ├── retry_manager.py      # 重试管理
│   ├── checkpoint.py         # 检查点管理 (P2)
│   ├── parallel_recovery.py  # 并行恢复 (P2)
│   └── container.py          # 依赖注入容器
├── llm/                      # LLM 相关
│   ├── config.py             # LLM 配置模型
│   ├── client.py             # LLM 客户端
│   └── model_router.py       # 模型路由器
├── ui/                       # UI 组件
│   ├── main_window.py        # 主窗口
│   ├── chat_widget.py        # 对话组件
│   └── dialogs/
│       └── llm_config_dialog.py  # LLM 配置对话框
├── main.py                   # 入口点
└── config.py                 # 配置管理
```

## 技术栈

- **Python 3.9+**
- **PyQt6**: GUI 框架
- **LangChain**: Agent 框架
- **tree-sitter**: Java 代码解析
- **sqlite-vec**: 纯 Python 向量存储
- **JaCoCo**: Java 覆盖率分析

## 开发计划

### 已完成

- [x] 项目基础结构
- [x] sqlite-vec 向量存储
- [x] 记忆系统
- [x] Java 代码解析
- [x] Maven 工具
- [x] PyQt6 UI
- [x] LLM 配置功能
- [x] Agent 核心 (ReAct)
- [x] 对话管理器
- [x] 暂停/恢复功能

### P0 - 核心能力 (已完成)

- [x] **流式代码生成** - 实时流式输出，支持中断和预览
- [x] **上下文智能管理** - ContextManager 处理大文件，关键片段提取
- [x] **代码质量预评估** - GenerationEvaluator 6维度质量评估
- [x] **部分成功处理** - PartialSuccessHandler 增量测试修复
- [x] **智能增量编辑** - SmartCodeEditor Search/Replace 精确修改

### P1 - 重要能力 (已完成)

- [x] **提示词优化** - PromptOptimizer 模型特定优化，A/B 测试
- [x] **错误知识库** - ErrorKnowledgeBase SQLite 持久化学习
- [x] **多构建工具** - BuildToolManager Maven/Gradle/Bazel 支持
- [x] **静态分析** - StaticAnalysisManager SpotBugs/PMD 集成
- [x] **MCP 集成** - MCPIntegration Model Context Protocol 支持
- [x] **上下文压缩** - ContextCompressor 相关性评分和压缩
- [x] **项目分析** - ProjectAnalyzer 依赖分析和多文件协调

### P2 - 增强能力 (已完成)

- [x] **多智能体协作** - AgentCoordinator + SpecializedAgent 专业化分工
- [x] **消息总线** - MessageBus 异步通信基础设施
- [x] **共享知识库** - SharedKnowledgeBase 知识共享机制
- [x] **经验回放** - ExperienceReplay 经验学习和复用
- [x] **性能监控** - MetricsCollector 全面指标收集和报告
- [x] **集成管理层** - IntegrationManager 组件生命周期管理

### P3 - 高级能力 (已完成)

- [x] **错误预测** - 编译前预测潜在错误，12种错误类型分类
- [x] **自适应策略** - 根据历史动态调整策略，ε-贪婪算法
- [x] **工具沙箱** - 安全沙箱隔离执行，3级安全控制
- [x] **用户交互** - 交互式修复建议和确认
- [x] **智能代码分析** - 语义分析、依赖图、影响分析

### 竞争力功能 (已完成)

- [x] **代码解释器** - 安全测试代码执行、运行时错误捕获
- [x] **智能重构** - 12种重构类型、自动重构执行
- [x] **质量分析器** - 6维度质量评估、问题检测

## 许可证

MIT License
