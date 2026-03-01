# PyUT Agent 持续优化计划

基于深入分析，发现以下需要改进的问题：

---

## 阶段 1：代码清理（P2）

### 1.1 清理未使用的导入

**目标文件：**
- [pyutagent/agent/generators/aider_generator.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/agent/generators/aider_generator.py) - 移除 `from pathlib import Path`
- [pyutagent/tools/code_editor.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/tools/code_editor.py) - 移除 tree_sitter 相关导入
- [pyutagent/core/container.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/core/container.py) - 将 `import inspect` 移到文件顶部
- [pyutagent/memory/vector_store.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/memory/vector_store.py) - 将函数内部导入移到文件顶部
- [pyutagent/ui/main_window.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/main_window.py) - 将函数内部导入移到文件顶部

---

## 阶段 2：国际化（P1）

### 2.1 中文日志改为英文

**目标文件（按优先级）：**
1. [pyutagent/agent/test_generator.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/agent/test_generator.py) - 约 50+ 处中文日志
2. [pyutagent/services/batch_generator.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/services/batch_generator.py) - 中文日志
3. [pyutagent/core/error_recovery.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/core/error_recovery.py) - 中文日志
4. [pyutagent/core/retry_manager.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/core/retry_manager.py) - 中文日志

### 2.2 UI 字符串国际化

**目标文件：**
- [pyutagent/ui/dialogs/llm_config_dialog.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/dialogs/llm_config_dialog.py) - 大量中文UI字符串
- [pyutagent/ui/dialogs/aider_config_dialog.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/dialogs/aider_config_dialog.py) - 大量中文UI字符串
- [pyutagent/ui/chat_widget.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/ui/chat_widget.py) - 中文UI字符串

**建议方案：**
创建 `pyutagent/ui/i18n.py` 国际化模块，支持多语言切换

---

## 阶段 3：异常处理优化（P0）

### 3.1 细化异常处理

**高危文件（优先修复）：**
- [pyutagent/agent/react_agent.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/agent/react_agent.py) - 15+ 处裸 except
- [pyutagent/agent/test_generator.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/agent/test_generator.py) - 10+ 处裸 except
- [pyutagent/tools/aider_integration.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/tools/aider_integration.py) - 多处裸 except
- [pyutagent/core/error_recovery.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/core/error_recovery.py) - 多处裸 except
- [pyutagent/core/config.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/core/config.py) - 配置相关异常

**改进方案：**
```python
# 改进前
except Exception as e:
    logger.error(f"Error: {e}")

# 改进后
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
except PermissionError as e:
    logger.error(f"Permission denied: {e}")
except subprocess.CalledProcessError as e:
    logger.error(f"Command failed: {e}")
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise
```

---

## 阶段 4：函数重构（P1）

### 4.1 拆分超长函数

**目标函数：**
| 函数 | 当前行数 | 目标行数 |
|------|---------|---------|
| `run_feedback_loop` | ~160 | < 50 |
| `generate_tests_with_aider` | ~193 | < 50 |
| `_attempt_fix` | ~107 | < 50 |
| `_execute_with_recovery` | ~105 | < 50 |
| `_compile_with_recovery` | ~80 | < 50 |
| `_run_tests_with_recovery` | ~78 | < 50 |

**重构策略：**
- 提取循环体为独立方法
- 使用策略模式替代条件分支
- 提取状态转换逻辑

---

## 阶段 5：类型注解完善（P2）

### 5.1 补充缺失的类型注解

**目标文件：**
- [pyutagent/agent/react_agent.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/agent/react_agent.py) - `_extract_java_code`, `_get_uncovered_info` 等
- [pyutagent/agent/test_generator.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/agent/test_generator.py) - 多个方法参数
- [pyutagent/tools/aider_integration.py](file:///d:/opensource/github/auto-ut-agent/pyutagent/tools/aider_integration.py) - `_extract_code_from_markdown`

**统一类型注解风格：**
- 使用 `from __future__ import annotations` (Python 3.7+)
- 或使用 `typing` 模块的 `List`, `Dict`, `Optional` 等

---

## 阶段 6：性能优化（P2）

### 6.1 缓存机制

**添加 LRU 缓存：**
- Java 文件解析结果缓存
- LLM 响应缓存
- Maven 依赖 classpath 缓存

### 6.2 异步优化

**改进同步调用：**
- Maven 命令执行改为异步
- 文件 I/O 使用 `aiofiles`

---

## 执行计划

```
第1周：代码清理 + 国际化
├── 清理未使用导入
├── 中文日志改为英文
└── UI 字符串国际化

第2周：异常处理优化
├── 修复高危文件的裸 except
└── 添加具体异常类型处理

第3周：函数重构
├── 拆分超长函数
└── 提取重复逻辑

第4周：类型注解 + 性能优化
├── 完善类型注解
├── 添加缓存机制
└── 异步优化
```

---

## 预期收益

1. **代码质量**：消除裸 except，提高错误诊断能力
2. **国际化**：支持多语言，便于全球用户使用
3. **可维护性**：函数拆分后更易理解和测试
4. **性能**：缓存机制减少重复计算
5. **类型安全**：完整类型注解便于静态检查

---

## 风险评估

- **低风险**：代码清理、类型注解
- **中风险**：国际化（需要测试UI显示）
- **高风险**：异常处理优化（可能暴露潜在bug）

建议按阶段逐步实施，每个阶段完成后进行充分测试。