# PyUT Agent 容错机制增强总结

## 核心改进

实现了**除非人工停止，否则 Agent 自己借助 AI 能力解决所有错误**的容错机制。

## 新增模块

### 1. Error Recovery Manager (`error_recovery.py`)

统一错误恢复入口，支持 11 种错误类型：
- COMPILATION_ERROR - 编译错误
- TEST_FAILURE - 测试失败
- TOOL_EXECUTION_ERROR - 工具执行错误
- PARSING_ERROR - 解析错误
- GENERATION_ERROR - 生成错误
- COVERAGE_ANALYSIS_ERROR - 覆盖率分析错误
- FILE_IO_ERROR - 文件 IO 错误
- NETWORK_ERROR - 网络错误
- LLM_API_ERROR - LLM API 错误
- UNKNOWN_ERROR - 未知错误

**双层分析机制**:
1. **本地分析层**: 使用 error_analyzer/failure_analyzer 进行初步分析
2. **LLM 深度分析层**: 将本地分析结果 + 原始错误 + 历史记录交给 LLM，让 LLM 给出智能修复建议

**恢复策略**:
- RETRY_IMMEDIATE - 立即重试
- RETRY_WITH_BACKOFF - 指数退避重试
- ANALYZE_AND_FIX - 分析并修复
- SKIP_AND_CONTINUE - 跳过继续
- RESET_AND_REGENERATE - 重置重新生成
- FALLBACK_ALTERNATIVE - 使用替代方案
- ESCALATE_TO_USER - 升级给用户（最后手段）

### 2. Retry Manager (`retry_manager.py`)

**无限重试机制**:
- `InfiniteRetryManager` - 除非用户停止，否则无限重试
- 支持 6 种重试策略：立即、固定延迟、指数退避、线性退避、随机抖动、自适应
- 指数退避：delay = base * (exponential_base ^ attempt)
- 最大延迟限制，防止等待时间过长

**智能重试**:
- 根据错误类型自动选择重试策略
- 记录每次重试历史
- 支持回调函数：on_retry, on_success, on_failure

### 3. 增强的 Prompt Builder

新增两个关键 Prompt：
- `build_error_analysis_prompt()` - LLM 错误分析 Prompt
- `build_comprehensive_fix_prompt()` - 综合修复 Prompt

包含：
- 错误分类和消息
- 本地分析结果
- 历史尝试记录
- 当前测试代码
- 目标类信息

## 重构的 ReAct Agent

### 核心改进

**无限重试循环**:
```python
# 旧逻辑：失败直接返回
if not compile_result.success:
    fix_result = await self._fix_compilation_errors(errors)
    if not fix_result.success:
        return AgentResult(success=False, ...)  # 直接失败

# 新逻辑：无限重试直到成功或用户停止
while not self._stop_requested:
    compile_success = await self._compile_with_recovery()
    if not compile_success:
        if self._stop_requested:
            break
        continue  # 自动重试
```

**统一的错误恢复**:
```python
async def _execute_with_recovery(self, operation, *args, step_name, **kwargs):
    """所有操作都通过此函数执行，自动获得错误恢复能力"""
    while not self._stop_requested:
        try:
            result = await operation(*args, **kwargs)
            if result.success:
                return result
            else:
                # 自动恢复
                recovery_result = await self._try_recover(error, context)
                if recovery_result.action == "fix":
                    await self._write_test_file(recovery_result.fixed_code)
                continue  # 重试
        except Exception as e:
            # 异常也自动恢复
            recovery_result = await self._try_recover(e, context)
            continue  # 重试
```

**专门的恢复方法**:
- `_compile_with_recovery()` - 编译失败自动修复
- `_run_tests_with_recovery()` - 测试失败自动修复
- `_execute_with_recovery()` - 通用操作自动恢复

### 停止信号机制

```python
# 用户可以随时停止
def request_stop(self):
    self._stop_requested = True
    self.retry_manager.stop()
    self.error_recovery.clear_history()

# Agent 定期检查
while not self._stop_requested:
    # 继续执行
```

## 容错流程示例

### 场景 1: 编译错误

```
1. 编译失败
2. ErrorRecoveryManager 分析错误类型
3. 本地分析器提取错误信息
4. LLM 分析错误原因和修复策略
5. LLM 生成修复后的代码
6. 写入修复后的代码
7. 重新编译
8. 如果还失败，重复步骤 2-7
9. 如果用户点击停止，退出
```

### 场景 2: 测试失败

```
1. 测试运行失败
2. TestFailureAnalyzer 解析 Surefire 报告
3. 分类失败类型（断言失败、空指针、Mock 问题等）
4. LLM 分析失败原因
5. LLM 生成修复后的测试代码
6. 写入修复后的代码
7. 重新运行测试
8. 如果还失败，重复步骤 2-7
9. 如果用户点击停止，退出
```

### 场景 3: 工具调用错误

```
1. Maven 命令执行失败（如网络问题）
2. ErrorRecoveryManager 识别为 TOOL_EXECUTION_ERROR
3. 根据错误类型选择策略：
   - 网络错误 → RETRY_WITH_BACKOFF
   - 命令未找到 → ESCALATE_TO_USER
   - 权限错误 → RETRY_IMMEDIATE
4. 执行恢复策略
5. 重试工具调用
```

### 场景 4: 多次修复失败

```
1. 第一次修复尝试失败
2. 记录尝试历史
3. LLM 收到历史记录，调整策略
4. 尝试不同的修复方法
5. 如果多次失败，尝试 RESET_AND_REGENERATE
6. 重新生成测试代码
7. 继续尝试直到成功或用户停止
```

## 关键特性

### 1. 除非人工停止，否则永不放弃

```python
while not self._stop_requested:
    # 无限重试循环
    # 编译失败 → 修复 → 重试
    # 测试失败 → 修复 → 重试
    # 任何错误 → 恢复 → 重试
```

### 2. AI 驱动的错误分析

不只是简单的正则匹配，而是让 LLM：
- 理解错误上下文
- 分析根本原因
- 学习历史失败
- 选择最佳策略
- 生成智能修复

### 3. 自适应策略

根据失败历史自动调整：
- 第一次失败 → 直接修复
- 第二次失败 → 指数退避后重试
- 第三次失败 → 尝试替代方案
- 多次失败 → 重置重新生成

### 4. 完整的错误覆盖

所有可能的错误都有处理：
- 编译错误 ✓
- 测试失败 ✓
- 工具执行错误 ✓
- 解析错误 ✓
- 生成错误 ✓
- 覆盖率分析错误 ✓
- 文件 IO 错误 ✓
- 网络错误 ✓
- LLM API 错误 ✓
- 未知错误 ✓

## 使用示例

```python
# 创建 Agent
agent = ReActAgent(
    llm_client=llm_client,
    working_memory=working_memory,
    project_path="/path/to/project",
    progress_callback=on_progress
)

# 启动生成（会自动处理所有错误）
result = await agent.generate_tests("MyClass.java")

# 用户可以随时停止
agent.stop()  #  graceful shutdown
```

## 与 Java UT Agent 对比

| 能力 | Java UT Agent | PyUT Agent (增强后) |
|------|---------------|---------------------|
| 编译错误修复 | 有限重试 | 无限重试 + AI 分析 |
| 测试失败修复 | 有限重试 | 无限重试 + AI 分析 |
| 工具错误处理 | 基本 | 完整分类 + 自适应策略 |
| 停止机制 | 无 | 用户随时可停止 |
| 历史学习 | 无 | 记录历史，调整策略 |
| 双层分析 | 无 | 本地 + LLM 双层 |

## 总结

这次增强实现了**真正的自治 Agent**：

1. **自我诊断**: 自动分析所有错误类型
2. **自我修复**: 使用 AI 生成修复方案
3. **自我调整**: 根据历史调整策略
4. **永不放弃**: 除非用户停止，否则持续尝试

这标志着 PyUT Agent 从"工具"进化为"自治代理"。