# PyUT Agent

AI 驱动的 Java 单元测试生成器，基于 Agent 架构，支持对话式交互。

## 特性

- 🤖 **Agent 架构**: 基于 LangChain 的 ReAct Agent，支持工具调用和规划
- 💬 **对话式 UI**: PyQt6 构建的图形界面，支持自然语言交互
- 🧠 **记忆系统**: 多层记忆（工作/短期/长期/向量），持续学习优化
- 🔍 **向量检索**: sqlite-vec 存储和检索相似测试模式
- ⏸️ **暂停/恢复**: 随时暂停生成任务，保存状态后可恢复
- 📊 **覆盖率分析**: 集成 JaCoCo，实时显示覆盖率报告

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

### 1. 打开项目
- 点击菜单 `文件 -> 打开项目`
- 选择一个 Maven 项目目录（包含 pom.xml）

### 2. 生成测试
- 在左侧文件树中选择一个 Java 文件
- 在对话区域输入: "生成 UserService 的测试"
- 或使用快捷键 `Ctrl+G`

### 3. 控制生成过程
- **暂停**: 输入 "暂停" 或点击暂停按钮
- **继续**: 输入 "继续" 恢复生成
- **查看状态**: 输入 "状态" 查看当前进度

### 4. 查看结果
- 生成的测试文件保存在 `src/test/java` 目录
- 覆盖率报告在右侧进度面板显示

## 配置

创建 `.env` 文件配置 LLM:

```env
PYUT_LLM_PROVIDER=openai
PYUT_LLM_API_KEY=your-api-key
PYUT_LLM_MODEL=gpt-4
PYUT_TARGET_COVERAGE=0.8
PYUT_MAX_ITERATIONS=10
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
├── memory/           # 记忆系统
│   ├── vector_store.py      # sqlite-vec 向量存储
│   ├── working_memory.py    # 工作记忆
│   └── short_term_memory.py # 短期记忆
├── tools/            # 工具
│   ├── java_parser.py       # Java 代码解析
│   └── maven_tools.py       # Maven 工具
├── ui/               # UI 组件
│   ├── main_window.py       # 主窗口
│   └── chat_widget.py       # 对话组件
├── main.py           # 入口点
└── config.py         # 配置
```

## 技术栈

- **Python 3.9+**
- **PyQt6**: GUI 框架
- **LangChain**: Agent 框架
- **tree-sitter**: Java 代码解析
- **sqlite-vec**: 纯 Python 向量存储
- **JaCoCo**: Java 覆盖率分析

## 开发计划

- [x] 项目基础结构
- [x] sqlite-vec 向量存储
- [x] 记忆系统
- [x] Java 代码解析
- [x] Maven 工具
- [x] PyQt6 UI
- [ ] Agent 核心 (ReAct)
- [ ] 对话管理器
- [ ] 暂停/恢复功能
- [ ] 集成测试

## 许可证

MIT License
