# 增量模式检查和优化计划

## 1. 发现的问题

### 1.1 🔴 高优先级问题

#### 问题 1: skip_analysis 模式下缺少编译检查
**位置**: `incremental_manager.py:273-276`
**问题**: 当 `skip_analysis=True` 时，所有测试都被标记为通过，但实际上可能存在编译错误或运行时错误。
```python
if not run_tests or self.config.skip_analysis:
    analysis.passing_tests = [m.method_name for m in test_methods]
    return analysis  # 没有验证测试是否真的能通过
```
**影响**: 可能保留实际上有问题的测试代码

#### 问题 2: tests_to_fix 缺少错误详情
**位置**: `incremental_manager.py:427-435`
**问题**: 构建 `tests_to_fix` 时没有包含错误消息和堆栈跟踪
```python
context.tests_to_fix.append(TestMethodResult(
    method_name=test_name,
    status=TestStatus.FAILED,
    # 缺少 error_message 和 stack_trace
))
```
**影响**: LLM 无法知道具体错误原因，修复效果差

#### 问题 3: 覆盖率分析返回空数据
**位置**: `incremental_manager.py:376-382`
**问题**: `_analyze_coverage` 返回的 `uncovered_lines` 和 `uncovered_branches` 总是空列表
```python
return {
    "line_coverage": report.line_coverage,
    "uncovered_lines": [],  # 总是空
    "uncovered_branches": [],  # 总是空
}
```
**影响**: 无法识别需要新增测试覆盖的代码

### 1.2 🟡 中优先级问题

#### 问题 4: 测试合并逻辑过于简单
**位置**: `incremental_manager.py:486-535`
**问题**: 
- 只是简单拼接方法代码
- 没有处理导入冲突
- 没有处理字段/辅助方法的重复

#### 问题 5: 增量提示词缺少关键信息
**位置**: `prompts.py:build_incremental_preserve_prompt`
**问题**:
- 没有提供失败测试的具体错误消息
- 没有提供未覆盖代码的具体内容
- 保留测试的代码格式可能不完整

#### 问题 6: extract_class_skeleton 可能丢失重要代码
**位置**: `partial_success_handler.py:191-232`
**问题**: 
- 可能丢失 `@BeforeEach`, `@AfterEach` 等设置方法
- 可能丢失辅助方法和字段

### 1.3 🟢 低优先级问题

#### 问题 7: 缺少增量模式统计
**问题**: 没有记录增量模式的效果统计（保留多少、修复多少、新增多少）

#### 问题 8: 缺少增量模式的详细日志
**问题**: 日志不够详细，难以调试增量模式问题

---

## 2. 优化方案

### 2.1 修复 skip_analysis 模式 (高优先级)

**方案**: 添加编译验证步骤

```python
async def analyze_existing_tests(self, test_file_path: str, run_tests: bool = True) -> ExistingTestAnalysis:
    # ... 现有代码 ...
    
    if not run_tests or self.config.skip_analysis:
        logger.info("[IncrementalTestManager] Skipping test execution analysis")
        
        # 新增: 至少验证编译
        if self.maven_runner:
            compile_success = await self._compile_test_file(test_file_path)
            if not compile_success:
                analysis.has_compilation_errors = True
                logger.warning("[IncrementalTestManager] Existing tests have compilation errors")
                return analysis
        
        analysis.passing_tests = [m.method_name for m in test_methods]
        return analysis
```

### 2.2 传递完整错误信息 (高优先级)

**方案**: 在分析测试结果时保存错误详情

```python
def build_incremental_context(self, analysis: ExistingTestAnalysis, ...):
    # ... 现有代码 ...
    
    if analysis.has_failing_tests and self.config.force_regenerate_failed:
        for test_name in analysis.failing_tests + analysis.error_tests:
            # 查找对应的测试结果（包含错误信息）
            test_result = self._find_test_result(test_name, analysis)
            if test_result:
                context.tests_to_fix.append(test_result)
```

### 2.3 修复覆盖率分析 (高优先级)

**方案**: 从覆盖率报告中提取未覆盖行

```python
def _analyze_coverage(self) -> Optional[Dict[str, Any]]:
    if not self.coverage_analyzer:
        return None
    
    try:
        report = self.coverage_analyzer.parse_report()
        
        if report:
            # 提取未覆盖行
            uncovered_lines = []
            if report.files:
                for file_coverage in report.files:
                    for line_num, is_covered in file_coverage.lines:
                        if not is_covered:
                            uncovered_lines.append(line_num)
            
            return {
                "line_coverage": report.line_coverage,
                "branch_coverage": report.branch_coverage,
                "uncovered_lines": uncovered_lines,
                "uncovered_branches": [],
            }
    except Exception as e:
        logger.warning(f"[IncrementalTestManager] Coverage analysis error: {e}")
    
    return None
```

### 2.4 增强测试合并逻辑 (中优先级)

**方案**: 智能合并，处理导入和辅助方法

```python
def merge_tests(self, preserved_test_code: str, new_test_code: str, class_name: str) -> str:
    # 解析两个版本的代码
    preserved_methods = self.parser.parse_test_methods(preserved_test_code)
    new_methods = self.parser.parse_test_methods(new_test_code)
    
    # 合并导入
    merged_imports = self._merge_imports(preserved_test_code, new_test_code)
    
    # 合并字段和辅助方法
    merged_fields = self._merge_fields(preserved_test_code, new_test_code)
    
    # 合并测试方法（去重）
    new_method_names = {m.method_name for m in new_methods}
    preserved_to_keep = [m for m in preserved_methods if m.method_name not in new_method_names]
    
    # 构建最终代码
    return self._build_merged_class(
        imports=merged_imports,
        fields=merged_fields,
        methods=preserved_to_keep + new_methods
    )
```

### 2.5 增强增量提示词 (中优先级)

**方案**: 提供更详细的上下文

```python
def build_incremental_preserve_prompt(self, ...):
    # ... 现有代码 ...
    
    # 新增: 提供失败测试的完整错误信息
    if tests_to_fix:
        tests_to_fix_section = "## Tests to Fix (REGENERATE THESE)\n"
        for test in tests_to_fix:
            tests_to_fix_section += f"### {test.method_name}\n"
            if test.error_message:
                tests_to_fix_section += f"Error: {test.error_message}\n"
            if test.stack_trace:
                tests_to_fix_section += f"Stack Trace:\n```\n{test.stack_trace[:500]}\n```\n"
    
    # 新增: 提供未覆盖代码的具体内容
    if uncovered_info.get("lines"):
        uncovered_section = "## Uncovered Code Lines\n"
        uncovered_section += f"Lines: {uncovered_info['lines'][:20]}\n"
```

### 2.6 保留设置方法 (中优先级)

**方案**: 修改 extract_class_skeleton 保留 @BeforeEach/@AfterEach

```python
def extract_class_skeleton(self, test_code: str) -> str:
    lines = test_code.split('\n')
    skeleton_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # 保留 @BeforeEach, @AfterEach, @BeforeAll, @AfterAll
        if re.match(r'\s*@(BeforeEach|AfterEach|BeforeAll|AfterAll)\b', stripped):
            # 保留整个方法
            while i < len(lines) and not re.match(r'\s*void\s+\w+', lines[i]):
                skeleton_lines.append(lines[i])
                i += 1
            # 保留方法体
            brace_depth = 0
            while i < len(lines):
                if '{' in lines[i]:
                    brace_depth += lines[i].count('{')
                if '}' in lines[i]:
                    brace_depth -= lines[i].count('}')
                skeleton_lines.append(lines[i])
                i += 1
                if brace_depth <= 0:
                    break
            continue
        
        # 跳过 @Test 方法
        if re.match(r'\s*@Test\b', stripped):
            # ... 跳过测试方法 ...
            continue
        
        skeleton_lines.append(line)
        i += 1
    
    return '\n'.join(skeleton_lines)
```

---

## 3. 任务分解

### Phase 1: 高优先级修复 (必须)

| ID | 任务 | 文件 | 预估 |
|----|------|------|------|
| 1.1 | 添加编译验证到 skip_analysis 模式 | incremental_manager.py | 30min |
| 1.2 | 传递完整错误信息到 tests_to_fix | incremental_manager.py | 20min |
| 1.3 | 修复覆盖率分析返回空数据 | incremental_manager.py | 20min |

### Phase 2: 中优先级优化 (建议)

| ID | 任务 | 文件 | 预估 |
|----|------|------|------|
| 2.1 | 增强测试合并逻辑 | incremental_manager.py | 1h |
| 2.2 | 增强增量提示词 | prompts.py | 30min |
| 2.3 | 保留设置方法 | partial_success_handler.py | 30min |

### Phase 3: 低优先级增强 (可选)

| ID | 任务 | 文件 | 预估 |
|----|------|------|------|
| 3.1 | 添加增量模式统计 | incremental_manager.py | 20min |
| 3.2 | 增强日志记录 | 多个文件 | 20min |

---

## 4. 验收标准

1. ✅ skip_analysis 模式至少验证编译
2. ✅ tests_to_fix 包含完整错误信息
3. ✅ 覆盖率分析返回真实的未覆盖行
4. ✅ 测试合并不丢失导入和设置方法
5. ✅ 增量提示词包含足够的上下文信息
6. ✅ 所有现有测试通过
7. ✅ 新增测试覆盖优化功能

---

## 5. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 编译验证增加时间 | 低 | 可配置跳过 |
| 合并逻辑复杂度增加 | 中 | 充分测试 |
| 提示词变长增加 token | 低 | 控制信息量 |
