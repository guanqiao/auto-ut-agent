# 计划：默认迭代次数设置为3 + LLM覆盖率评估

## 背景

当前系统：
- 默认迭代次数 `max_iterations = 10`（在 `working_memory.py` 中）
- 覆盖率统计依赖 JaCoCo（Maven 项目）
- 已有 `GenerationEvaluator` 可以预估覆盖率潜力

## 目标

1. **修改默认迭代次数**：从 10 改为 3
2. **添加 LLM 覆盖率评估**：当 JaCoCo 等工具不可用时，通过 LLM 评估覆盖率

## 实现方案

### 任务 1：修改默认迭代次数

**文件**: `pyutagent/memory/working_memory.py`

**修改内容**:
```python
# 修改前
max_iterations: int = 10

# 修改后
max_iterations: int = 3
```

### 任务 2：创建 LLM 覆盖率评估器

**新文件**: `pyutagent/agent/llm_coverage_evaluator.py`

**功能**:
- 分析源代码和测试代码
- 使用 LLM 评估测试覆盖率
- 返回覆盖率评估结果（行覆盖率、分支覆盖率、方法覆盖率）

**核心逻辑**:
1. 解析源代码，提取方法、分支等信息
2. 解析测试代码，识别测试的方法和场景
3. 调用 LLM 进行覆盖率评估
4. 返回结构化的覆盖率报告

### 任务 3：集成 LLM 覆盖率评估到现有流程

**修改文件**: `pyutagent/agent/handlers/coverage_handler.py`

**修改内容**:
- 在 `analyze_coverage` 方法中添加 fallback 机制
- 当 JaCoCo 失败时，使用 LLM 评估器

**修改文件**: `pyutagent/tools/maven_tools.py`

**修改内容**:
- 在 `CoverageAnalyzer.parse_report()` 中添加 fallback 逻辑

### 任务 4：更新配置

**修改文件**: `pyutagent/core/config.py`（如果需要）

**修改内容**:
- 添加 LLM 覆盖率评估的配置选项

## 技术细节

### LLM 覆盖率评估 Prompt 设计

```
你是一个代码覆盖率分析专家。请分析以下源代码和测试代码，评估测试覆盖率。

## 源代码
{source_code}

## 测试代码
{test_code}

## 类信息
{class_info}

请评估：
1. 行覆盖率：测试覆盖了多少行代码（0.0-1.0）
2. 分支覆盖率：测试覆盖了多少分支（0.0-1.0）
3. 方法覆盖率：测试覆盖了多少方法（0.0-1.0）
4. 未覆盖的方法列表
5. 未覆盖的分支列表

返回 JSON 格式的评估结果。
```

### Fallback 策略

1. 首先尝试使用 JaCoCo 获取精确覆盖率
2. 如果 JaCoCo 不可用（非 Maven 项目、报告解析失败等），使用 LLM 评估
3. LLM 评估结果标记为 "estimated"，与精确覆盖率区分

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `pyutagent/memory/working_memory.py` | 修改 | max_iterations: 10 → 3 |
| `pyutagent/agent/llm_coverage_evaluator.py` | 新增 | LLM 覆盖率评估器 |
| `pyutagent/agent/handlers/coverage_handler.py` | 修改 | 添加 LLM fallback |
| `pyutagent/tools/maven_tools.py` | 修改 | 添加 fallback 逻辑 |

## 测试计划

1. 单元测试：LLM 覆盖率评估器
2. 集成测试：覆盖率分析流程（JaCoCo 可用/不可用场景）
3. 端到端测试：完整测试生成流程

## 风险评估

- **LLM 评估准确性**：LLM 评估的覆盖率不如 JaCoCo 精确，需要标记为 "estimated"
- **API 成本**：LLM 评估会增加 API 调用成本
- **性能影响**：LLM 评估比本地分析慢

## 预期结果

1. 默认迭代次数减少到 3，减少不必要的迭代
2. 非 Maven 项目或 JaCoCo 不可用时，仍能获得覆盖率评估
3. 覆盖率评估结果包含来源标识（精确/估算）
