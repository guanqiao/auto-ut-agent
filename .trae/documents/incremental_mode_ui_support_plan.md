# 增量模式UI支持实施计划

## 问题分析

### 当前状态
增量模式（Incremental Mode）已在后端完整实现，但UI界面未完全暴露此功能：

| 组件 | 增量模式支持 | 状态 |
|------|-------------|------|
| 后端核心 (`incremental_manager.py`) | ✅ 完整实现 | 已完成 |
| CLI命令 (`generate.py`, `generate_all.py`) | ✅ `-i/--incremental` 参数 | 已完成 |
| 批量生成对话框 (`batch_generate_dialog.py`) | ✅ 增量模式复选框 | 已完成 |
| **单文件测试生成（GUI）** | ❌ 无增量选项 | **需实现** |
| **VSCode扩展** | ❌ 无增量参数 | **需实现** |
| **IntelliJ插件** | ❌ 无增量选项 | **需实现** |
| **后端API** | ❌ 无增量参数 | **需实现** |

## 实施步骤

### 第一阶段：后端API支持

#### 1.1 修改API客户端类型定义
**文件**: `pyutagent-vscode/src/backend/apiClient.ts`

```typescript
// 添加增量模式参数
interface GenerateOptions {
    incremental?: boolean;
    skipAnalysis?: boolean;
}

async generateTest(filePath: string, options?: GenerateOptions): Promise<GenerationResult>
```

#### 1.2 修改IntelliJ客户端
**文件**: `pyutagent-intellij/src/main/java/com/pyutagent/intellij/PyUTClient.java`

- 在 `GenerateOptions` 类中添加 `incremental` 和 `skipAnalysis` 字段
- 修改API请求以传递这些参数

### 第二阶段：GUI单文件生成支持

#### 2.1 修改AgentWorker
**文件**: `pyutagent/ui/main_window.py`

- 在 `AgentWorker.__init__()` 中添加 `incremental_mode` 和 `skip_test_analysis` 参数
- 传递参数给 `ReActAgent` 或 `EnhancedAgent`

#### 2.2 修改主窗口UI
**文件**: `pyutagent/ui/main_window.py`

- 在 `start_generation()` 方法中添加增量模式选项对话框或快捷设置
- 可选方案：
  - **方案A**: 在生成前弹出配置对话框
  - **方案B**: 在工具栏添加增量模式开关按钮
  - **方案C**: 在ChatWidget中添加复选框

推荐方案C：在ChatWidget中添加复选框，与批量生成对话框保持一致

#### 2.3 修改ChatWidget
**文件**: `pyutagent/ui/chat_widget.py`

- 添加增量模式复选框
- 添加跳过测试分析复选框（依赖增量模式）
- 添加信号传递选项状态

### 第三阶段：VSCode扩展支持

#### 3.1 修改生成测试命令
**文件**: `pyutagent-vscode/src/commands/generateTest.ts`

- 添加快速选择对话框，让用户选择生成模式
- 选项：完整生成 / 增量生成 / 增量生成（跳过分析）

```typescript
const mode = await vscode.window.showQuickPick([
    { label: '完整生成', description: '重新生成所有测试', value: 'full' },
    { label: '增量生成', description: '保留通过的测试', value: 'incremental' },
    { label: '增量生成（跳过分析）', description: '不运行现有测试', value: 'incremental-skip' }
]);
```

#### 3.2 更新API调用
**文件**: `pyutagent-vscode/src/backend/apiClient.ts`

- 修改 `generateTest()` 方法接受选项参数
- 更新API请求格式

### 第四阶段：IntelliJ插件支持

#### 4.1 修改生成测试动作
**文件**: `pyutagent-intellij/src/main/java/com/pyutagent/intellij/actions/GenerateTestAction.java`

- 在生成前显示选项对话框
- 添加增量模式复选框
- 传递选项到API调用

#### 4.2 创建选项对话框
**新文件**: `pyutagent-intellij/src/main/java/com/pyutagent/intellij/dialogs/GenerateOptionsDialog.java`

- 创建简单的选项对话框
- 包含增量模式和跳过分析复选框

### 第五阶段：测试验证

#### 5.1 单元测试
- 为修改的组件添加单元测试
- 测试增量模式参数正确传递

#### 5.2 集成测试
- 测试GUI单文件生成的增量模式
- 测试VSCode扩展的增量模式
- 测试IntelliJ插件的增量模式

## 详细实施清单

### 文件修改列表

| 序号 | 文件 | 修改内容 |
|------|------|----------|
| 1 | `pyutagent/ui/chat_widget.py` | 添加增量模式复选框UI |
| 2 | `pyutagent/ui/main_window.py` | 传递增量模式参数到AgentWorker |
| 3 | `pyutagent-vscode/src/backend/apiClient.ts` | 添加增量模式参数支持 |
| 4 | `pyutagent-vscode/src/commands/generateTest.ts` | 添加生成模式选择对话框 |
| 5 | `pyutagent-intellij/.../PyUTClient.java` | 添加增量模式字段 |
| 6 | `pyutagent-intellij/.../GenerateTestAction.java` | 添加选项对话框 |
| 7 | `pyutagent-intellij/.../GenerateOptionsDialog.java` | 新建选项对话框类 |

### 代码变更详情

#### 1. chat_widget.py 变更

```python
# 在生成按钮区域添加：
self.incremental_check = QCheckBox("增量模式")
self.incremental_check.setToolTip("保留现有通过的测试")
self.skip_analysis_check = QCheckBox("跳过分析")
self.skip_analysis_check.setEnabled(False)
self.incremental_check.stateChanged.connect(self.on_incremental_changed)
```

#### 2. main_window.py 变更

```python
# AgentWorker.__init__ 添加参数：
def __init__(self, ..., incremental_mode: bool = False, skip_test_analysis: bool = False):
    self.incremental_mode = incremental_mode
    self.skip_test_analysis = skip_test_analysis

# ReActAgent 初始化传递参数：
self.agent = ReActAgent(
    ...,
    incremental_mode=self.incremental_mode,
    skip_test_analysis=self.skip_test_analysis
)
```

#### 3. apiClient.ts 变更

```typescript
interface GenerateOptions {
    incremental?: boolean;
    skipAnalysis?: boolean;
}

async generateTest(filePath: string, options?: GenerateOptions): Promise<GenerationResult> {
    const response = await this.client.post('/api/generate', {
        action: 'generate_test',
        file_path: filePath,
        incremental: options?.incremental ?? false,
        skip_analysis: options?.skipAnalysis ?? false
    });
    return response.data;
}
```

#### 4. generateTest.ts 变更

```typescript
const mode = await vscode.window.showQuickPick([
    { label: '$(refresh) 完整生成', description: '重新生成所有测试用例', value: 'full' },
    { label: '$(diff) 增量生成', description: '保留已通过的测试用例', value: 'incremental' },
    { label: '$(debug-reverse) 增量生成（跳过分析）', description: '不运行现有测试，直接分析', value: 'incremental-skip' }
], { placeHolder: '选择测试生成模式' });

const result = await api.generateTest(filePath, {
    incremental: mode?.value !== 'full',
    skipAnalysis: mode?.value === 'incremental-skip'
});
```

## 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| UI变更可能影响现有用户习惯 | 中 | 默认不勾选增量模式，保持原有行为 |
| 多平台UI一致性 | 低 | 参考批量生成对话框的实现模式 |
| API向后兼容 | 低 | 增量参数设为可选，默认false |

## 验收标准

1. ✅ GUI单文件测试生成可选择增量模式
2. ✅ VSCode扩展支持增量模式选择
3. ✅ IntelliJ插件支持增量模式选择
4. ✅ 增量模式参数正确传递到后端
5. ✅ 跳过分析选项仅在增量模式下可用
6. ✅ 所有现有测试通过
7. ✅ 新功能有相应测试覆盖
