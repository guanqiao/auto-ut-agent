# PyUT Agent CLI 使用指南

## 概述

PyUT Agent 提供了完整的命令行界面（CLI），支持自动化测试生成、项目扫描和配置管理。CLI 适合集成到 CI/CD 流程或批量处理场景。

## 安装

```bash
# 安装 PyUT Agent
pip install pyutagent

# 或开发模式安装
pip install -e ".[dev]"
```

## 基本使用

### 启动 CLI

```bash
# 查看帮助
pyutagent --help

# 查看版本
pyutagent --version
```

## 命令参考

### 1. scan - 扫描项目

扫描 Maven 项目并列出所有 Java 文件。

```bash
# 基本扫描
pyutagent scan /path/to/project

# 以树形结构显示
pyutagent scan /path/to/project --tree
```

**输出示例：**
```
Found 25 Java files in my-project

  📄 com/example/UserService.java
  📄 com/example/OrderService.java
  📄 com/example/PaymentService.java
  ...
```

### 2. generate - 生成单个文件的测试

为指定的 Java 文件生成单元测试。

```bash
# 基本用法
pyutagent generate /path/to/MyClass.java

# 指定 LLM 配置
pyutagent generate /path/to/MyClass.java --llm deepseek-coder

# 设置覆盖率目标
pyutagent generate /path/to/MyClass.java --coverage-target 90

# 设置最大迭代次数
pyutagent generate /path/to/MyClass.java --max-iterations 15

# 实时查看进度
pyutagent generate /path/to/MyClass.java --watch
```

**选项：**
- `--llm <name>`: 使用的 LLM 配置名称或 ID（默认：default）
- `--output-dir <path>`: 测试文件输出目录
- `--coverage-target <int>`: 目标覆盖率百分比（默认：80）
- `--max-iterations <int>`: 最大迭代次数（默认：10）
- `--watch`: 实时显示生成进度

**输出示例：**
```
Generating tests for UserService.java...
  LLM: gpt-4
  Coverage target: 80%
  Max iterations: 10

✓ Tests generated successfully!
  Test file: src/test/java/com/example/UserServiceTest.java
  Coverage: 85.3%
```

### 3. generate-all - 批量生成测试

为项目中的所有 Java 文件生成测试。

```bash
# 基本用法
pyutagent generate-all /path/to/project

# 并行生成（4个worker）
pyutagent generate-all /path/to/project --parallel 4

# 设置超时
pyutagent generate-all /path/to/project --timeout 600

# 两阶段模式：先生成所有代码，再统一编译
pyutagent generate-all /path/to/project --defer-compilation

# 快速模式：生成后仅编译一次
pyutagent generate-all /path/to/project --compile-only-at-end
```

**选项：**
- `--llm <name>`: LLM 配置名称
- `--output-dir <path>`: 输出目录
- `--coverage-target <int>`: 目标覆盖率（默认：80）
- `--max-iterations <int>`: 每个文件的最大迭代次数（默认：10）
- `--parallel <int>`: 并行 worker 数量（默认：1，0 表示无限制）
- `--timeout <int>`: 每个文件的超时时间（秒，默认：300）
- `--continue-on-error`: 出错时继续（默认：True）
- `--stop-on-error`: 出错时停止
- `--defer-compilation`: 延迟编译直到所有文件生成完成
- `--compile-only-at-end`: 仅在最后编译一次（隐含 --defer-compilation）

**输出示例：**
```
╭─────────────────────────────────────────────────────────────╮
│ Batch Test Generation                                        │
│                                                              │
│ Project: my-project                                          │
│ Java files: 50                                               │
│ Parallel workers: 4                                          │
│ Coverage target: 80%                                         │
│ Timeout per file: 300s                                       │
│ Continue on error: True                                      │
│ Defer compilation: True                                      │
│ ✓ Two-phase mode enabled (generate all → compile all)        │
╰─────────────────────────────────────────────────────────────╯

Starting batch generation...

[进度表格...]

╭─────────────────────────────────────────────────────────────╮
│ Summary                                                      │
│                                                              │
│ Total files: 50                                              │
│ Successful: 48                                               │
│ Failed: 2                                                    │
│ Success rate: 96.0%                                          │
│ Total time: 1215.8s                                          │
╰─────────────────────────────────────────────────────────────╯
```

### 4. config - 配置管理

管理 PyUT Agent 的各种配置。

#### 4.1 LLM 配置

```bash
# 列出所有 LLM 配置
pyutagent config llm list

# 添加新的 LLM 配置
pyutagent config llm add \
  --name "My GPT-4" \
  --provider openai \
  --model gpt-4 \
  --api-key sk-xxx \
  --endpoint https://api.openai.com/v1

# 设置默认配置
pyutagent config llm set-default <config-id>

# 测试 LLM 连接
pyutagent config llm test <config-id>
```

#### 4.2 Maven 配置

```bash
# 显示 Maven 配置
pyutagent config maven show

# 设置 Maven 路径
pyutagent config maven set --path /usr/local/bin/mvn
```

#### 4.3 JDK 配置

```bash
# 显示 JDK 配置
pyutagent config jdk show

# 设置 JAVA_HOME
pyutagent config jdk set --path /usr/lib/jvm/java-17
```

#### 4.4 覆盖率配置

```bash
# 显示覆盖率配置
pyutagent config coverage show

# 设置覆盖率参数
pyutagent config coverage set \
  --target 0.85 \
  --max-iterations 15 \
  --max-compilation-attempts 5
```

#### 4.5 Aider 配置

```bash
# 显示 Aider 配置
pyutagent config aider show
```

#### 4.6 显示所有配置

```bash
# 显示当前配置概览
pyutagent config show
```

## 配置文件

### 配置文件位置

所有配置文件存储在 `~/.pyutagent/` 目录：

```
~/.pyutagent/
├── llm_config.json       # LLM 配置
├── aider_config.json     # Aider 配置
├── app_config.json       # 应用配置
└── app_state.json        # 应用状态（历史记录等）
```

### 环境变量

可以通过环境变量配置默认值：

```bash
# LLM 配置
export PYUT_LLM_PROVIDER=openai
export PYUT_LLM_API_KEY=sk-xxx
export PYUT_LLM_MODEL=gpt-4

# 覆盖率配置
export PYUT_TARGET_COVERAGE=0.8
export PYUT_MAX_ITERATIONS=10
```

## 使用场景

### 1. CI/CD 集成

```yaml
# GitHub Actions 示例
name: Generate Tests
on: [push]

jobs:
  generate-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install PyUT Agent
        run: pip install pyutagent
      
      - name: Configure LLM
        run: |
          pyutagent config llm add \
            --name "CI Bot" \
            --provider openai \
            --model gpt-4 \
            --api-key ${{ secrets.OPENAI_API_KEY }}
      
      - name: Generate Tests
        run: |
          pyutagent generate-all . \
            --parallel 4 \
            --coverage-target 80 \
            --defer-compilation
      
      - name: Run Tests
        run: mvn test
```

### 2. 批量处理

```bash
# 为整个项目生成测试
pyutagent generate-all /path/to/large-project \
  --parallel 8 \
  --defer-compilation \
  --timeout 600

# 分批处理（按模块）
for module in module1 module2 module3; do
  pyutagent generate-all /path/to/project/$module \
    --defer-compilation \
    --parallel 4
done

# 最后统一编译
cd /path/to/project
mvn test-compile
```

### 3. 快速原型

```bash
# 快速生成测试代码（不验证）
pyutagent generate-all /path/to/project \
  --compile-only-at-end \
  --parallel 0  # 无限制并行

# 查看生成的文件
find src/test/java -name "*Test.java" -newer .last_run

# 手动修复编译错误
mvn test-compile 2>&1 | grep "error:"
```

## 最佳实践

### 1. 选择合适的编译策略

- **标准模式**（默认）：适合关键业务代码，需要高质量保证
- **延迟编译**（`--defer-compilation`）：适合大规模批量生成（>20个文件）
- **快速模式**（`--compile-only-at-end`）：适合快速原型和初步测试

### 2. 并行度设置

- **CPU 充足**：`--parallel 8` 或更高
- **内存有限**：`--parallel 2-4`
- **无限制**：`--parallel 0`（注意内存消耗）

### 3. 超时设置

- **简单类**：`--timeout 120`
- **复杂类**：`--timeout 600`
- **默认**：`--timeout 300`

### 4. 错误处理

- **继续处理**：`--continue-on-error`（默认）
- **严格模式**：`--stop-on-error`

## 故障排除

### 1. LLM 连接失败

```bash
# 检查配置
pyutagent config llm list

# 测试连接
pyutagent config llm test <config-id>

# 检查 API Key
pyutagent config show
```

### 2. 编译失败

```bash
# 检查 Maven 配置
pyutagent config maven show

# 检查 JDK 配置
pyutagent config jdk show

# 手动测试编译
mvn test-compile
```

### 3. 超时问题

```bash
# 增加超时时间
pyutagent generate /path/to/ComplexClass.java --timeout 600

# 减少迭代次数
pyutagent generate /path/to/ComplexClass.java --max-iterations 5
```

## 与 GUI 对比

| 功能 | CLI | GUI |
|------|-----|-----|
| 单文件生成 | ✅ `generate` | ✅ 选择文件 → 生成 |
| 批量生成 | ✅ `generate-all` | ✅ 工具 → 批量生成 |
| 项目扫描 | ✅ `scan` | ✅ 工具 → 项目统计 |
| 配置管理 | ✅ `config` | ✅ 设置菜单 |
| 暂停/恢复 | ⚠️ 有限支持 | ✅ 完整支持 |
| 实时日志 | ⚠️ 有限 | ✅ 详细日志面板 |
| 项目历史 | ❌ | ✅ 最近项目列表 |
| CI/CD 集成 | ✅ 完美 | ❌ |

## 更多信息

- [README.md](../README.md) - 项目概览
- [BATCH_GENERATION_USAGE.md](../BATCH_GENERATION_USAGE.md) - 批量生成详细说明
- [ARCHITECTURE.md](../ARCHITECTURE.md) - 架构文档
