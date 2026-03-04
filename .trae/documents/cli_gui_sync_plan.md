# CLI和GUI功能同步计划

## 背景

PyUT Agent 提供了两种使用方式：
- **CLI (命令行界面)**: 通过 `pyutagent-cli` 命令使用
- **GUI (图形界面)**: 通过 `pyutagent` 命令启动的 PyQt6 应用

目前两者功能存在差异，需要进行同步以确保用户体验一致性。

## 当前功能对比

### CLI 功能列表

| 命令 | 功能描述 | GUI对应功能 |
|------|---------|------------|
| `scan <project>` | 扫描Maven项目，列出Java文件 | ❌ 无专门功能（仅文件树） |
| `generate <file>` | 为单个Java文件生成测试 | ✅ 有 |
| `generate-all <project>` | 批量生成所有测试 | ✅ 有 |
| `config llm list` | 列出所有LLM配置 | ⚠️ 部分（在对话框中） |
| `config llm add` | 添加LLM配置 | ✅ 有 |
| `config llm set-default` | 设置默认LLM配置 | ✅ 有 |
| `config llm test` | 测试LLM连接 | ✅ 有 |
| `config show` | 显示当前配置 | ❌ 无 |
| `config aider show` | 显示Aider配置 | ⚠️ 未实现 |

### GUI 功能列表

| 功能 | 描述 | CLI对应功能 |
|------|------|------------|
| 打开项目 | 选择Maven项目目录 | ❌ 无 |
| 最近项目历史 | 管理最近打开的项目 | ❌ 无 |
| 自动加载上次项目 | 启动时自动打开上次项目 | ❌ 无 |
| 文件树浏览 | 可视化浏览项目结构 | ⚠️ scan命令（仅列表） |
| 单文件生成 | 为选中的文件生成测试 | ✅ generate命令 |
| 批量生成 | 批量生成多个文件的测试 | ✅ generate-all命令 |
| 暂停/恢复/终止 | 控制生成过程 | ❌ 无 |
| LLM配置对话框 | 配置LLM提供商和模型 | ✅ config llm |
| Maven配置对话框 | 配置Maven路径 | ❌ 无 |
| JDK配置对话框 | 配置JAVA_HOME | ❌ 无 |
| 覆盖率配置对话框 | 配置测试生成参数 | ❌ 无 |
| Aider配置对话框 | 配置Aider高级功能 | ⚠️ 未实现 |
| 实时进度显示 | 显示生成进度和状态 | ⚠️ 有进度条但不详细 |
| 日志显示 | 实时日志输出 | ❌ 无 |
| 对话式交互 | 自然语言交互 | ❌ 无 |

## 功能同步计划

### 第一阶段：CLI功能增强

#### 1.1 添加项目历史管理
- **优先级**: P2
- **描述**: 为CLI添加项目历史记录功能
- **实现**:
  - 添加 `pyutagent history` 命令组
  - `pyutagent history list` - 列出最近项目
  - `pyutagent history clear` - 清除历史
  - 在 generate/generate-all 命令中自动记录项目路径

#### 1.2 添加配置管理命令
- **优先级**: P1
- **描述**: 补全CLI的配置管理功能
- **实现**:
  - `pyutagent config maven show` - 显示Maven配置
  - `pyutagent config maven set` - 设置Maven路径
  - `pyutagent config jdk show` - 显示JDK配置
  - `pyutagent config jdk set` - 设置JAVA_HOME
  - `pyutagent config coverage show` - 显示覆盖率配置
  - `pyutagent config coverage set` - 设置覆盖率参数
  - 完善 `pyutagent config aider show` - 显示Aider配置

#### 1.3 添加暂停/恢复支持
- **优先级**: P2
- **描述**: 为CLI生成命令添加暂停/恢复功能
- **实现**:
  - 在 generate 命令中添加 `--interactive` 模式
  - 支持键盘快捷键（Ctrl+P 暂停，Ctrl+R 恢复，Ctrl+C 终止）
  - 显示实时状态提示

### 第二阶段：GUI功能增强

#### 2.1 添加项目扫描功能
- **优先级**: P1
- **描述**: 在GUI中添加类似CLI scan的功能
- **实现**:
  - 在工具菜单添加"项目统计"功能
  - 显示Java文件数量、包结构、代码行数等统计信息
  - 支持导出扫描报告

#### 2.2 添加配置显示功能
- **优先级**: P2
- **描述**: 在GUI中添加配置概览功能
- **实现**:
  - 在设置菜单添加"显示当前配置"功能
  - 以对话框形式显示所有配置信息
  - 支持复制配置信息

#### 2.3 增强批量生成对话框
- **优先级**: P1
- **描述**: 对齐CLI的generate-all功能
- **实现**:
  - 添加 `--defer-compilation` 选项的UI对应
  - 添加 `--compile-only-at-end` 选项的UI对应
  - 显示两阶段模式的进度

### 第三阶段：文档更新

#### 3.1 更新README.md
- **优先级**: P0
- **描述**: 在README中添加CLI使用说明
- **内容**:
  - CLI安装和基本使用
  - CLI命令参考
  - CLI与GUI功能对比表
  - CLI使用场景建议

#### 3.2 创建CLI使用文档
- **优先级**: P1
- **描述**: 创建详细的CLI使用文档
- **文件**: `docs/cli_usage.md`
- **内容**:
  - 所有CLI命令详细说明
  - 使用示例
  - 配置管理
  - 批量生成最佳实践
  - 常见问题

#### 3.3 更新BATCH_GENERATION_USAGE.md
- **优先级**: P1
- **描述**: 更新批量生成文档，添加GUI使用说明
- **内容**:
  - GUI批量生成使用方法
  - GUI与CLI批量生成对比
  - 两阶段模式在GUI中的使用

## 实施步骤

### 步骤1：CLI配置命令增强（P1）
1. 创建 `pyutagent/cli/commands/config.py` 的扩展命令
2. 添加 Maven、JDK、Coverage、Aider 配置命令
3. 编写单元测试
4. 更新相关文档

### 步骤2：GUI项目扫描功能（P1）
1. 创建项目统计对话框
2. 实现项目扫描逻辑
3. 添加菜单项和快捷键
4. 编写单元测试

### 步骤3：GUI批量生成增强（P1）
1. 更新 BatchGenerateDialog 添加两阶段模式选项
2. 实现两阶段模式的进度显示
3. 编写单元测试
4. 更新文档

### 步骤4：文档更新（P0）
1. 更新 README.md 添加CLI说明
2. 创建 docs/cli_usage.md
3. 更新 BATCH_GENERATION_USAGE.md

### 步骤5：CLI项目历史管理（P2）
1. 创建 history 命令组
2. 实现项目历史记录功能
3. 编写单元测试

### 步骤6：CLI暂停/恢复支持（P2）
1. 设计交互式模式
2. 实现键盘控制
3. 编写单元测试

### 步骤7：GUI配置显示功能（P2）
1. 创建配置概览对话框
2. 实现配置信息展示
3. 编写单元测试

## 测试计划

### 单元测试
- 每个新增CLI命令至少2个测试用例
- 每个新增GUI功能至少3个测试用例
- 测试覆盖率目标：>90%

### 集成测试
- CLI和GUI功能一致性测试
- 配置持久化测试
- 批量生成功能测试

### 手动测试
- CLI命令在Windows/Linux/macOS上的兼容性
- GUI在不同分辨率下的显示效果
- CLI与GUI配置文件共享测试

## 风险和注意事项

1. **配置兼容性**: CLI和GUI共享同一配置文件，需确保配置格式一致
2. **用户体验**: CLI和GUI的目标用户不同，功能实现方式需适配各自特点
3. **向后兼容**: 新增功能不应破坏现有功能
4. **文档同步**: 代码更新后需及时更新文档

## 预期成果

1. CLI和GUI功能基本对齐
2. 用户可以根据场景选择使用CLI或GUI
3. 文档完善，用户可以快速上手
4. 代码质量高，测试覆盖充分

## 时间估算

- 第一阶段（CLI增强）：2-3天
- 第二阶段（GUI增强）：2-3天
- 第三阶段（文档更新）：1天
- 测试和修复：1-2天
- **总计**：6-9天
