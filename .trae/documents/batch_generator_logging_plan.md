# 为 Batch Generator 添加详细日志并动态显示到 UI 的实施计划

## 目标
为 batch generator 添加更多详细日志，确保生成测试过程中的日志和其他 action 的日志能够动态显示到 UI 上。

## 当前状态分析

### 已有的日志机制
1. **BatchGenerator** (`pyutagent/services/batch_generator.py`)
   - 使用标准 Python logging
   - 有基本的日志记录（开始、完成、错误等）
   - 有 `progress_callback` 机制更新进度

2. **UI 日志显示** (`pyutagent/ui/batch_generate_dialog.py`)
   - 有 `log_area` (QTextEdit) 显示日志
   - 使用 `LogEmitter` 捕获日志并通过信号传递
   - `on_log_message` 方法格式化并显示日志

3. **ReActAgent** 和 **StepExecutor**
   - 执行各个步骤时有详细日志
   - 有 `progress_callback` 参数

### 需要改进的地方
1. BatchGenerator 中的日志不够详细，缺少关键步骤的日志
2. Action 执行过程中的日志没有被充分记录
3. 需要确保所有日志都能实时传递到 UI

## 实施步骤

### 第一阶段：增强 BatchGenerator 日志

#### 1.1 在文件处理开始时添加详细日志
- 文件路径、文件大小、方法数量
- 使用的生成模式（standard/multi-agent）
- 配置参数（timeout, coverage_target 等）

#### 1.2 在关键步骤添加日志
- **代码分析阶段**：分析进度、发现的方法、复杂度等
- **测试生成阶段**：生成进度、使用的策略、生成的测试数量
- **编译阶段**：编译命令、编译结果、错误信息
- **测试执行阶段**：执行的测试、通过/失败数量
- **覆盖率分析阶段**：覆盖率结果、未覆盖的代码

#### 1.3 在错误处理时添加详细日志
- 错误类型、错误详情
- 重试次数、重试策略
- 恢复动作、恢复结果

#### 1.4 在 Multi-Agent 模式添加日志
- Agent 协作过程
- 任务分配和执行
- Agent 间的消息传递

### 第二阶段：增强 Action 执行日志

#### 2.1 在 ActionExecutor 中添加详细日志
- Action 类型、参数
- 执行前后的状态
- 执行结果、耗时

#### 2.2 在各个 Action 实现中添加日志
- FIX_IMPORTS：修复的导入列表
- ADD_DEPENDENCY：添加的依赖
- FIX_SYNTAX：修复的语法错误
- REGENERATE_TEST：重新生成的原因
- 等等

### 第三阶段：确保日志实时传递到 UI

#### 3.1 验证 LogEmitter 配置
- 确保 LogEmitter 正确安装在 'pyutagent' logger 上
- 确保日志级别设置正确（INFO 及以上）

#### 3.2 优化日志显示
- 添加日志分类（使用不同的图标和颜色）
- 添加时间戳
- 自动滚动到最新日志
- 添加日志过滤功能（可选）

#### 3.3 添加进度信息到日志
- 在 progress_callback 中同时记录日志
- 显示当前步骤的详细信息
- 显示预估剩余时间（可选）

### 第四阶段：测试和优化

#### 4.1 测试日志输出
- 单文件生成测试
- 批量文件生成测试
- 错误场景测试
- Multi-Agent 模式测试

#### 4.2 性能优化
- 确保日志不影响性能
- 避免过多的 debug 日志
- 使用异步日志写入（如果需要）

## 具体实施细节

### 文件修改清单

1. **pyutagent/services/batch_generator.py**
   - 在 `_generate_single_standard` 方法中添加详细日志
   - 在 `_generate_single_multi_agent` 方法中添加详细日志
   - 在 `_on_agent_progress` 方法中增强日志
   - 在错误处理部分添加详细日志

2. **pyutagent/agent/tools/action_executor.py**
   - 在 `execute_action` 方法中添加详细日志
   - 在 `execute_action_plan` 方法中添加进度日志
   - 在各个 `_execute_*` 方法中添加具体操作的日志

3. **pyutagent/agent/components/execution_steps.py**
   - 在各个步骤执行方法中添加更详细的日志
   - 在重试和恢复时添加日志

4. **pyutagent/ui/batch_generate_dialog.py**
   - 优化 `on_log_message` 方法的显示效果
   - 添加日志分类和图标
   - 确保自动滚动

### 日志格式规范

```
[时间戳] [日志级别] [模块名] emoji 消息内容
```

示例：
```
[14:30:25] [INFO] [BatchGenerator] 📁 开始处理文件: UserService.java
[14:30:25] [INFO] [BatchGenerator] 📊 文件信息 - 大小: 15.2KB, 方法: 12个
[14:30:26] [INFO] [BatchGenerator] 🔍 分析代码结构...
[14:30:27] [INFO] [BatchGenerator] ✨ 生成测试代码...
[14:30:30] [INFO] [BatchGenerator] ✅ 测试生成完成 - 覆盖率: 85.3%
```

### Emoji 图标规范
- 📁 文件操作
- 🔍 分析/检查
- ✨ 生成/创建
- ⚙️ 编译/构建
- 🧪 测试执行
- 📊 覆盖率/统计
- ✅ 成功
- ❌ 失败
- ⚠️ 警告
- 🔄 重试/恢复
- 🤖 Multi-Agent
- 🛠️ Action 执行

## 预期效果

1. **更详细的日志信息**
   - 用户可以清楚看到每个步骤的执行情况
   - 错误时有详细的上下文信息
   - 性能指标（耗时、重试次数等）

2. **实时动态显示**
   - 日志实时出现在 UI 上
   - 不同级别的日志有不同颜色
   - 自动滚动到最新日志

3. **更好的可调试性**
   - 问题定位更容易
   - 可以追踪整个生成过程
   - 便于用户理解系统行为

## 风险和注意事项

1. **性能影响**
   - 过多的日志可能影响性能
   - 解决方案：只在 INFO 级别记录关键信息，DEBUG 级别记录详细信息

2. **日志量控制**
   - 批量处理时日志可能非常多
   - 解决方案：提供日志清理功能，限制日志区域大小

3. **线程安全**
   - 确保多线程环境下的日志安全
   - 使用 Qt 信号机制保证线程安全

## 实施顺序

1. 先修改 BatchGenerator 添加核心日志（最关键）
2. 修改 ActionExecutor 添加 action 日志
3. 优化 UI 日志显示效果
4. 测试和调整
