# 批量测试生成 - 两阶段模式使用说明

## 概述

批量测试生成支持两种模式：
- **标准模式**：逐文件生成并验证
- **两阶段模式**：先生成所有代码，再统一编译

PyUT Agent 同时提供 CLI 和 GUI 两种使用方式，功能完全对齐。

## 使用方式对比

### CLI 使用

```bash
# 标准模式（默认）
pyutagent generate-all /path/to/project

# 两阶段模式
pyutagent generate-all /path/to/project --defer-compilation

# 快速模式
pyutagent generate-all /path/to/project --compile-only-at-end
```

### GUI 使用

1. 打开项目：`文件 → 打开项目`
2. 批量生成：`工具 → Generate All Tests`
3. 在批量生成对话框中：
   - 选择要生成的文件
   - 选择编译策略：
     - **Standard Mode**：逐文件验证（推荐用于质量保证）
     - **Defer Compilation**：生成所有后统一编译（推荐用于大规模生成）
     - **Fast Mode**：仅生成代码，最后编译一次（最快）
   - 点击 "Start" 开始生成

## 工作原理

### 传统模式（默认）
```
文件 1: 生成 → 编译 → 测试 → 覆盖率分析 → 修复
文件 2: 生成 → 编译 → 测试 → 覆盖率分析 → 修复
文件 3: 生成 → 编译 → 测试 → 覆盖率分析 → 修复
...
```

### 两阶段模式（新功能）
```
阶段 1 - 生成所有文件:
  文件 1: 生成测试代码
  文件 2: 生成测试代码
  文件 3: 生成测试代码
  ...

阶段 2 - 统一编译检查:
  编译所有生成的测试文件
  报告编译错误
```

## 使用方法

### CLI 命令

#### 基本用法
```bash
# 使用两阶段模式生成测试
pyutagent generate-all /path/to/project --defer-compilation

# 或者使用简写选项
pyutagent generate-all /path/to/project --compile-only-at-end
```

#### 完整示例
```bash
# 并行生成，延迟编译
pyutagent generate-all /path/to/project \
  --parallel 4 \
  --coverage-target 80 \
  --defer-compilation \
  --timeout 300

# 仅在所有文件生成完成后编译一次
pyutagent generate-all /path/to/project \
  --parallel 8 \
  --compile-only-at-end \
  --coverage-target 90
```

### 选项说明

- `--defer-compilation`: 延迟编译直到所有文件生成完成
  - 生成阶段跳过编译和测试验证
  - 所有文件生成后统一编译
  - 适合快速生成大量测试代码

- `--compile-only-at-end`: 仅在最后编译一次
  - 隐含 `--defer-compilation` 选项
  - 完全跳过生成过程中的验证步骤
  - 最快的生成速度，但错误反馈延迟

### Python API 使用

```python
from pyutagent.services.batch_generator import BatchGenerator, BatchConfig
from pyutagent.llm.client import LLMClient

# 配置批量生成
config = BatchConfig(
    parallel_workers=4,
    timeout_per_file=300,
    coverage_target=80,
    defer_compilation=True,  # 启用两阶段模式
    compile_only_at_end=True  # 仅在最后编译
)

# 创建生成器
generator = BatchGenerator(
    llm_client=llm_client,
    project_path="/path/to/project",
    config=config
)

# 执行批量生成
files = ["src/main/java/com/example/Class1.java", ...]
result = generator.generate_all_sync(files)

# 检查结果
print(f"Generated: {result.success_count}/{result.total_files}")

if result.compilation_result:
    if result.compilation_result.success:
        print("✓ All tests compiled successfully")
    else:
        print(f"✗ Compilation failed: {result.compilation_result.errors}")
```

## 输出示例

### 生成阶段
```
Batch Test Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Project: my-project
Java files: 50
Parallel workers: 4
Coverage target: 80%
Timeout per file: 300s
Continue on error: True
Defer compilation: True
✓ Two-phase mode enabled (generate all → compile all)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Starting batch generation...

[生成进度...]

Batch generation complete - Success: 48, Failed: 2, Duration: 1200.5s
```

### 编译阶段（Phase 2）
```
Phase 2: Compilation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Batch Compilation Results

Status: ✓ Success
Compiled files: 48
Failed files: 0
Compilation time: 15.3s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 最终汇总
```
Batch Generation Complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary

Total files: 50
Successful: 48
Failed: 2
Success rate: 96.0%
Total time: 1215.8s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 注意事项

### 优点
- ✅ **更快的生成速度** - 避免每个文件的编译/测试开销
- ✅ **更好的资源利用** - 并行生成多个文件
- ✅ **集中错误处理** - 统一查看所有编译错误
- ✅ **适合大规模生成** - 一次性生成整个项目的测试

### 缺点
- ⚠️ **延迟错误反馈** - 生成完成后才能发现编译问题
- ⚠️ **无法实时修复** - 生成过程中不进行错误修复
- ⚠️ **可能需要手动修复** - 编译错误需要手动处理

### 建议场景

**推荐使用两阶段模式：**
- 大规模批量生成（>20 个文件）
- 快速原型测试
- 对已有代码的快速测试覆盖
- 时间紧迫的场景

**不推荐使用两阶段模式：**
- 关键业务代码测试
- 需要高质量保证的场景
- 复杂逻辑的测试生成
- 第一次使用项目时的测试生成

## 故障排除

### 编译失败处理

如果批量编译失败，可以：

1. **查看编译错误**
   ```bash
   # 错误信息会在输出中显示
   Compilation errors:
     ✗ Class1Test.java:15: error: cannot find symbol
     ✗ Class2Test.java:23: error: method does not override
   ```

2. **手动修复错误**
   - 导航到出错的测试文件
   - 根据编译错误信息修复代码

3. **重新生成特定文件**
   ```bash
   # 单独生成有问题的文件（使用默认模式进行验证）
   pyutagent generate /path/to/project --file Class1.java
   ```

### 性能优化建议

1. **调整并行 worker 数量**
   - CPU 充足：`--parallel 8` 或更高
   - 内存有限：`--parallel 2-4`

2. **设置合理的超时**
   - 简单类：`--timeout 120`
   - 复杂类：`--timeout 600`

3. **分批处理大项目**
   ```bash
   # 第一批：核心模块
   pyutagent generate-all src/main/java/com/example/core --defer-compilation
   
   # 第二批：工具类
   pyutagent generate-all src/main/java/com/example/util --defer-compilation
   
   # 最后统一编译
   mvn test-compile
   ```

## 技术实现

### 关键组件

1. **BatchConfig** - 配置选项
   ```python
   @dataclass
   class BatchConfig:
       defer_compilation: bool = False
       compile_only_at_end: bool = False
   ```

2. **WorkingMemory** - 工作状态控制
   ```python
   @dataclass
   class WorkingMemory:
       skip_verification: bool = False  # 跳过验证标志
   ```

3. **ReActAgent** - 智能代理
   - 检查 `skip_verification` 标志
   - 跳过编译和测试步骤
   - 仅生成代码

### 工作流程

```
BatchGenerator.generate_all()
  ├─ Phase 1: 并行生成所有文件
  │   ├─ _generate_single(file1) → 跳过验证
  │   ├─ _generate_single(file2) → 跳过验证
  │   └─ ...
  │
  └─ Phase 2: 批量编译（如果 defer_compilation=True）
      └─ _compile_all_tests() → 统一编译
```

## 更多信息

- [CLI 命令参考](docs/cli/commands.md)
- [批量生成最佳实践](docs/best-practices/batch-generation.md)
- [问题报告](https://github.com/auto-ut-agent/issues)
