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

```bash
pip install -e .
```

## 开发安装

```bash
pip install -e ".[dev]"
```

## 运行

```bash
pyutagent
```

## 测试

```bash
pytest
```

## 使用示例

1. 打开应用后，通过菜单 `File -> Open Project` 选择 Maven 项目
2. 在对话区域输入: "生成 UserService 的测试"
3. Agent 会自动分析代码、生成测试、运行验证
4. 可随时输入 "暂停" 中断，输入 "继续" 恢复

## 技术栈

- Python 3.9+
- PyQt6 (GUI)
- LangChain (Agent)
- tree-sitter (Java 解析)
- sqlite-vec (向量存储)
- JaCoCo (覆盖率分析)
