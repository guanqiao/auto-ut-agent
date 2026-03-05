# JaCoCo 自动配置功能计划

## 需求概述

在 GUI 上提供一个选项，供用户选择是否自动利用 LLM 生成 JaCoCo 的依赖和 POM 配置，然后自动配置到 pom.xml 并安装相关依赖。

## 功能分析

### 1. 核心功能模块

#### 1.1 JaCoCo 配置检测器
- 检测 pom.xml 中是否已存在 JaCoCo 配置
- 检测 JaCoCo 插件和依赖
- 检测 JaCoCo Maven 插件配置

#### 1.2 LLM 驱动的配置生成器
- 读取当前 pom.xml 内容
- 使用 LLM 分析现有配置
- 生成 JaCoCo 依赖和插件配置
- 确保配置与现有配置兼容

#### 1.3 POM 配置应用器
- 安全地修改 pom.xml
- 添加 JaCoCo 依赖
- 添加 JaCoCo Maven 插件配置
- 创建备份机制

#### 1.4 依赖安装器
- 运行 Maven 命令安装新依赖
- 验证安装是否成功
- 处理安装失败的情况

### 2. GUI 界面设计

#### 2.1 新增配置对话框: `JacocoConfigDialog`
位置: `pyutagent/ui/dialogs/jacoco_config_dialog.py`

功能:
- 显示当前 JaCoCo 配置状态（已配置/未配置）
- 提供"自动配置 JaCoCo" 复选框选项
- 显示配置预览（生成后的 pom.xml 变更）
- 提供"应用配置"按钮
- 显示配置进度和结果

#### 2.2 集成到 CoverageConfigDialog
- 在覆盖率配置对话框中添加 JaCoCo 自动配置选项
- 当检测到 JaCoCo 未配置时显示警告/提示

#### 2.3 主窗口菜单集成
- 在 Tools 菜单中添加 "JaCoCo 配置" 选项

### 3. 配置存储

#### 3.1 新增配置类: `JacocoSettings`
位置: `pyutagent/core/config.py`

属性:
- `auto_configure`: bool - 是否启用自动配置
- `skip_if_exists`: bool - 如果已存在配置是否跳过
- `preferred_version`: str - 首选 JaCoCo 版本

### 4. 实现步骤

#### 步骤 1: 创建 JaCoCo 配置服务
**文件**: `pyutagent/services/jacoco_config_service.py`

功能:
- `JacocoConfigService` 类
- 检测 pom.xml 中 JaCoCo 配置
- 使用 LLM 生成配置建议
- 应用配置到 pom.xml
- 安装依赖

#### 步骤 2: 创建 LLM 提示模板
**文件**: `pyutagent/agent/prompts/jacoco_config_prompts.py`

提示模板:
- `JACOCO_CONFIG_GENERATION_PROMPT`: 生成 JaCoCo 配置的提示
- `JACOCO_CONFIG_ANALYSIS_PROMPT`: 分析现有 pom.xml 的提示

#### 步骤 3: 创建 GUI 对话框
**文件**: `pyutagent/ui/dialogs/jacoco_config_dialog.py`

界面元素:
- 当前状态显示区域
- 自动配置选项复选框
- 配置预览文本框
- 操作按钮（检测、生成、应用）
- 进度显示

#### 步骤 4: 更新配置系统
**文件**: `pyutagent/core/config.py`

添加:
- `JacocoSettings` 数据类
- 在 `Settings` 类中集成 `JacocoSettings`

#### 步骤 5: 集成到主窗口
**文件**: `pyutagent/ui/main_window.py`

修改:
- 添加菜单项
- 在测试生成前检查 JaCoCo 配置

#### 步骤 6: 集成到 Coverage 配置对话框
**文件**: `pyutagent/ui/dialogs/coverage_config_dialog.py`

修改:
- 添加 JaCoCo 配置状态显示
- 添加快速配置按钮

### 5. 详细代码设计

#### 5.1 JacocoConfigService

```python
class JacocoConfigService:
    def __init__(self, project_path: str, llm_client: LLMClient):
        self.project_path = Path(project_path)
        self.llm_client = llm_client
        self.pom_editor = PomEditor(project_path)
    
    def check_jacoco_configured(self) -> bool:
        """检查是否已配置 JaCoCo"""
        pass
    
    async def generate_config_with_llm(self, pom_content: str) -> dict:
        """使用 LLM 生成 JaCoCo 配置"""
        pass
    
    def apply_config(self, config: dict) -> bool:
        """应用配置到 pom.xml"""
        pass
    
    async def install_dependencies(self) -> bool:
        """安装依赖"""
        pass
```

#### 5.2 JaCoCo 配置提示模板

```python
JACOCO_CONFIG_GENERATION_PROMPT = """
你是一个 Maven 配置专家。请分析以下 pom.xml 内容，并生成 JaCoCo 代码覆盖率工具的配置。

当前 pom.xml:
```xml
{pom_content}
```

请生成以下 JSON 格式的配置建议:
{{
    "dependencies": [
        {{
            "group_id": "org.jacoco",
            "artifact_id": "jacoco-maven-plugin",
            "version": "0.8.11",
            "scope": "test"
        }}
    ],
    "build_plugins": [
        {{
            "group_id": "org.jacoco",
            "artifact_id": "jacoco-maven-plugin",
            "version": "0.8.11",
            "executions": [
                {{
                    "goals": ["prepare-agent"],
                    "phase": "test-compile"
                }},
                {{
                    "goals": ["report"],
                    "phase": "test"
                }}
            ]
        }}
    ],
    "explanation": "配置说明..."
}}

注意:
1. 版本号应该与项目中的其他依赖版本兼容
2. 如果已有类似配置，请提供升级建议
3. 确保配置符合 Maven 标准
"""
```

#### 5.3 JacocoConfigDialog UI

```python
class JacocoConfigDialog(QDialog):
    def setup_ui(self):
        # 状态显示
        self.status_label = QLabel()
        
        # 自动配置选项
        self.auto_config_checkbox = QCheckBox("自动配置 JaCoCo")
        
        # 配置预览
        self.preview_text = QTextEdit()
        
        # 按钮
        self.detect_btn = QPushButton("检测当前配置")
        self.generate_btn = QPushButton("生成配置")
        self.apply_btn = QPushButton("应用配置")
```

### 6. 测试策略

#### 6.1 单元测试
- `test_jacoco_config_service.py`: 测试配置服务
- `test_jacoco_config_dialog.py`: 测试对话框

#### 6.2 集成测试
- 测试完整配置流程
- 测试与 LLM 的集成
- 测试 pom.xml 修改

### 7. 错误处理

#### 7.1 可能的错误场景
1. pom.xml 不存在
2. pom.xml 格式错误
3. LLM 生成失败
4. 依赖安装失败
5. 权限不足

#### 7.2 错误处理策略
- 每个操作都有回滚机制
- 详细的错误信息展示
- 提供手动修复建议

### 8. 安全考虑

1. **备份机制**: 修改 pom.xml 前自动创建备份
2. **验证机制**: 应用配置前验证 XML 格式
3. **权限检查**: 检查文件写入权限
4. **沙盒测试**: 可选的测试环境验证

## 实施计划

### Phase 1: 核心服务 (第 1 步)
- [ ] 创建 `JacocoConfigService` 类
- [ ] 实现配置检测功能
- [ ] 实现 LLM 配置生成功能
- [ ] 实现配置应用功能
- [ ] 编写单元测试

### Phase 2: GUI 界面 (第 2-3 步)
- [ ] 创建 `JacocoConfigDialog`
- [ ] 实现界面交互逻辑
- [ ] 集成进度显示
- [ ] 编写 UI 测试

### Phase 3: 系统集成 (第 4-6 步)
- [ ] 更新配置系统
- [ ] 集成到主窗口
- [ ] 集成到 Coverage 配置对话框
- [ ] 端到端测试

### Phase 4: 完善和优化
- [ ] 错误处理优化
- [ ] 用户体验优化
- [ ] 文档编写
- [ ] 性能优化

## 文件变更清单

### 新增文件
1. `pyutagent/services/jacoco_config_service.py` - JaCoCo 配置服务
2. `pyutagent/agent/prompts/jacoco_config_prompts.py` - LLM 提示模板
3. `pyutagent/ui/dialogs/jacoco_config_dialog.py` - 配置对话框
4. `tests/unit/services/test_jacoco_config_service.py` - 服务测试
5. `tests/unit/ui/test_jacoco_config_dialog.py` - UI 测试

### 修改文件
1. `pyutagent/core/config.py` - 添加 JaCoCo 配置设置
2. `pyutagent/ui/main_window.py` - 添加菜单项和集成
3. `pyutagent/ui/dialogs/coverage_config_dialog.py` - 添加 JaCoCo 配置选项
4. `pyutagent/tools/pom_editor.py` - 增强 POM 编辑功能（如需要）

## 预期成果

1. 用户可以在 GUI 中一键配置 JaCoCo
2. 系统自动检测现有配置并给出建议
3. LLM 智能生成与项目兼容的配置
4. 自动备份和应用配置
5. 自动安装依赖并验证
6. 完整的错误处理和用户反馈
