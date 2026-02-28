# 集成 Aider 能力增强 Java UT 生成和修复方案

## 当前项目分析

项目 `pyutagent` 是一个 Java 单元测试自动生成工具，核心功能包括：
- `test_generator.py` - 测试生成主逻辑
- `error_analyzer.py` - 编译错误分析
- `failure_analyzer.py` - 测试失败分析
- `java_parser.py` - Java 代码解析（基于 tree-sitter）

## 集成方案

### 1. 核心改进：引入 Aider 的 Diff 编辑模式

Aider 的核心优势是使用 **Search/Replace diff 格式** 进行精确的代码编辑，而非全量生成。这可以：
- 避免 LLM 生成完整文件时的"幻觉"问题
- 精确定位修改位置
- 支持多文件协同编辑

### 2. 新增模块设计

#### 2.1 `code_editor.py` - 代码编辑器（核心）
- 实现 Aider 风格的 Search/Replace 编辑
- 基于 tree-sitter 验证语法正确性
- 支持批量应用多个编辑操作

#### 2.2 `aider_integration.py` - Aider 能力集成
- 封装 Aider 的编辑策略
- 提供测试代码专用的编辑提示模板
- 集成编译/测试验证反馈循环

#### 2.3 `edit_validator.py` - 编辑验证器
- 验证编辑操作的有效性
- 检查语法完整性
- 确保编辑后代码可编译

### 3. 工作流程增强

```
生成阶段:
  1. LLM 生成初始测试代码（全量）
  2. 保存并编译
  3. 如有编译错误 → 使用 Search/Replace 精确修复

修复阶段:
  1. 分析测试失败
  2. LLM 生成 Diff 格式的修复建议
  3. 应用编辑并验证
  4. 运行测试确认修复
```

### 4. 关键技术点

#### 4.1 Search/Replace 格式
```python
# 格式示例
<<<<<<< SEARCH
    @Test
    void testAdd() {
        assertEquals(3, calculator.add(1, 2));
    }
=======
    @Test
    void testAdd() {
        // Test positive numbers
        assertEquals(3, calculator.add(1, 2));
        // Test edge case
        assertEquals(0, calculator.add(0, 0));
    }
>>>>>>> REPLACE
```

#### 4.2 与现有模块集成
- 复用 `JavaCodeParser` 进行代码解析
- 复用 `CompilationErrorAnalyzer` 分析编译错误
- 复用 `TestFailureAnalyzer` 分析测试失败

### 5. 实施步骤

1. **创建 `code_editor.py`** - 核心编辑引擎
2. **创建 `aider_integration.py`** - Aider 策略封装
3. **创建 `edit_validator.py`** - 编辑验证
4. **修改 `test_generator.py`** - 集成新的编辑能力
5. **添加单元测试** - 验证编辑准确性

### 6. 预期收益

- **准确性提升**: 从全量生成改为精确编辑，减少错误
- **可维护性**: 编辑操作可追溯、可回滚
- **效率提升**: 只修改必要部分，减少 LLM token 消耗
- **验证闭环**: 每次编辑后自动验证编译和测试

请确认此方案后，我将开始实施具体的代码实现。