## 增强计划

### 1. 数据模型层修改

#### 1.1 修改 `pyutagent/llm/config.py`
- 创建 `LLMConfigCollection` 类来管理多个 LLM 配置
- 每个配置保留现有的 `ca_cert` 字段（已存在）
- 添加配置的唯一标识符（id/name）
- 添加默认配置标记

#### 1.2 修改 `pyutagent/tools/aider_integration.py`
- 修改 `AiderConfig` 类，将 `architect_model` 和 `editor_model` 从字符串改为引用 LLM 配置的 ID
- 添加方法根据 ID 获取完整的 LLM 配置（包含 CA 证书）

### 2. UI 层修改

#### 2.1 重构 `pyutagent/ui/dialogs/llm_config_dialog.py`
- 改为支持多配置管理（列表+详情模式）
- 左侧：配置列表（可添加/删除/选择默认）
- 右侧：配置详情编辑（保留现有字段）
- 为每个配置添加 CA 证书路径选择（文件选择器）

#### 2.2 修改 `pyutagent/ui/dialogs/aider_config_dialog.py`
- Architect 模型和 Editor 模型改为下拉选择框
- 选项从 LLM 配置列表中动态加载
- 显示每个模型的基本信息（名称、提供商）

### 3. 应用层修改

#### 3.1 修改 `pyutagent/ui/main_window.py`
- 更新配置管理逻辑，支持多配置
- 修改 LLM 客户端初始化，使用选中的配置

#### 3.2 修改 `pyutagent/llm/client.py`
- 确保 CA 证书正确传递给 HTTP 客户端（已实现）

### 4. 配置文件存储
- 配置将保存为 JSON 格式，包含多个 LLM 配置和 Aider 配置
- 保持向后兼容性

### 文件修改清单
1. `pyutagent/llm/config.py` - 添加多配置集合类
2. `pyutagent/tools/aider_integration.py` - 修改 AiderConfig 模型引用方式
3. `pyutagent/ui/dialogs/llm_config_dialog.py` - 重构为多配置管理界面
4. `pyutagent/ui/dialogs/aider_config_dialog.py` - 模型选择改为下拉框
5. `pyutagent/ui/main_window.py` - 更新配置管理逻辑

请确认此计划后，我将开始实施具体的代码修改。