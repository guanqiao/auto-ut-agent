# PyUT Agent CLI - TDD 开发方案

## TDD 开发流程

### 阶段1: 编写测试（Red）

1. 首先编写CLI模块的测试用例，定义期望的CLI行为
2. 运行测试，确认测试失败（因为功能尚未实现）

### 阶段2: 实现功能（Green）

1. 编写最简实现使测试通过
2. 运行测试，确认测试通过

### 阶段3: 重构（Refactor）

1. 优化代码结构，保持测试通过

## 具体实施步骤

### Step 1: 编写测试用例

创建 `tests/cli/test_cli_commands.py`：

* 测试 `scan` 命令：验证能正确扫描Maven项目

* 测试 `generate` 命令：验证能调用TestGenerator

* 测试 `config` 命令：验证配置CRUD操作

* 测试 `status` 命令：验证状态显示

### Step 2: 创建CLI骨架

创建 `pyutagent/cli/` 模块：

```
pyutagent/cli/
├── __init__.py
├── main.py          # CLI入口
├── commands/
│   ├── __init__.py
│   ├── scan.py
│   ├── generate.py
│   ├── config.py
│   └── status.py
└── console.py       # 控制台输出工具
```

### Step 3: 实现各命令

* 使用 `click` 框架

* 复用现有后端逻辑（react\_agent, test\_generator等）

* 使用 `rich` 库提供美观的命令行输出

### Step 4: 更新配置

* `pyproject.toml` 添加 `pyutagent-cli` 入口点

* 添加 `click` 和 `rich` 依赖

### Step 5: 运行测试

* 运行 `pytest tests/cli/` 确保所有测试通过

## 命令设计

```bash
# 扫描项目
pyutagent-cli scan /path/to/project

# 生成测试
pyutagent-cli generate /path/to/MyClass.java --watch

# 配置管理
pyutagent-cli config llm list
pyutagent-cli config llm add --name gpt4 --provider openai

# 查看状态
pyutagent-cli status
```

请确认此TDD方案后，我将开始第一步：编写测试用例。
