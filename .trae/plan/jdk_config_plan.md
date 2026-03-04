# JDK配置功能开发计划

## 1. 需求概述

在GUI上新增JDK配置路径功能，实现：
- 用户可手动配置JDK路径
- 优先使用配置的JDK路径
- 未配置时智能检测系统JDK（类似Maven的检测策略）
- 完美集成到现有JDK相关工具中

## 2. 现有代码分析

### 2.1 配置管理 (`pyutagent/core/config.py`)
- 已有 `MavenSettings` 数据类作为参考模板
- `Settings` 类集中管理所有配置
- 配置持久化到 `~/.pyutagent/app_config.json`

### 2.2 Maven智能检测 (`pyutagent/tools/maven_tools.py`)
- `find_maven_executable()` 函数实现了智能搜索策略
- 搜索顺序：PATH → 环境变量 → 平台特定位置
- 支持 Windows 和 Unix/macOS 双平台

### 2.3 JDK使用位置
- `compilation_handler.py` - 使用 `javac` 编译测试代码
- `code_interpreter.py` - 使用 `java` 和 `javac` 执行测试
- `environment.py` - Java环境检测

### 2.4 GUI配置对话框 (`pyutagent/ui/dialogs/maven_config_dialog.py`)
- 完整的配置对话框模板
- 支持手动配置和自动检测切换
- 路径验证和用户友好的提示

## 3. 实现计划

### 3.1 配置层改造

**文件**: `pyutagent/core/config.py`

添加 `JDKSettings` 数据类：
```python
@dataclass
class JDKSettings:
    """JDK configuration.
    
    Attributes:
        java_home: Path to JDK home directory. If empty, auto-detect will be used.
    """
    java_home: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JDKSettings":
        return cls(**data)
```

在 `Settings` 类中添加：
```python
jdk: JDKSettings = Field(
    default_factory=JDKSettings,
    description="JDK configuration"
)
```

### 3.2 智能检测层

**文件**: `pyutagent/tools/java_tools.py` (新建)

创建类似 `find_maven_executable()` 的智能检测函数：

```python
def find_java_executable() -> Optional[str]:
    """Find Java executable with smart search strategy.
    
    Search order:
    1. Check JAVA_HOME environment variable
    2. Check JDK_HOME environment variable  
    3. Check PATH using shutil.which
    4. Windows-specific locations
    5. Unix/macOS common locations
    
    Returns:
        Path to java executable if found, None otherwise
    """

def find_javac_executable() -> Optional[str]:
    """Find javac executable with smart search strategy."""

def find_java_home() -> Optional[str]:
    """Find JAVA_HOME directory."""
```

搜索策略：
- **环境变量**: JAVA_HOME, JDK_HOME, JRE_HOME
- **Windows**: Program Files, Chocolatey, Scoop, 常见用户安装位置
- **Unix/macOS**: /usr/lib/jvm, /Library/Java/JavaVirtualMachines, /opt/java, Homebrew路径

### 3.3 GUI配置对话框

**文件**: `pyutagent/ui/dialogs/jdk_config_dialog.py` (新建)

参考 `MavenConfigDialog` 实现：
- JDK路径输入框（支持浏览选择目录）
- 自动检测结果显示
- 使用检测到的JDK复选框
- 检测到的Java版本信息显示
- 配置说明文档

### 3.4 主窗口集成

**文件**: `pyutagent/ui/main_window.py`

在 Settings 菜单中添加 JDK 配置入口：
```python
jdk_config_action = QAction("&JDK Configuration...", self)
jdk_config_action.triggered.connect(self.on_jdk_config)
settings_menu.addAction(jdk_config_action)
```

添加 `on_jdk_config()` 方法处理对话框。

### 3.5 JDK工具集成

**文件**: `pyutagent/core/code_interpreter.py`

修改 `InterpreterConfig`：
```python
@dataclass
class InterpreterConfig:
    # ... existing fields ...
    java_path: str = ""  # 改为空，运行时解析
    javac_path: str = ""  # 改为空，运行时解析
```

修改 `CodeInterpreter.__init__`：
- 优先使用配置的JDK路径
- 未配置时调用智能检测

**文件**: `pyutagent/agent/handlers/compilation_handler.py`

修改 `_run_javac()` 方法：
- 使用配置的或智能检测的 `javac` 路径

**文件**: `pyutagent/core/environment.py`

增强 `detect_java()` 方法：
- 集成新的智能检测逻辑
- 支持配置优先级

## 4. 文件变更清单

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| 修改 | `pyutagent/core/config.py` | 添加 JDKSettings 类 |
| 新建 | `pyutagent/tools/java_tools.py` | JDK智能检测工具 |
| 新建 | `pyutagent/ui/dialogs/jdk_config_dialog.py` | JDK配置对话框 |
| 修改 | `pyutagent/ui/main_window.py` | 添加菜单入口 |
| 修改 | `pyutagent/core/code_interpreter.py` | 使用配置的JDK |
| 修改 | `pyutagent/agent/handlers/compilation_handler.py` | 使用配置的JDK |
| 修改 | `pyutagent/core/environment.py` | 增强检测逻辑 |

## 5. 测试计划

### 5.1 单元测试
- `test_java_tools.py` - 测试智能检测函数
- `test_config.py` - 测试JDK配置序列化/反序列化

### 5.2 集成测试
- 配置保存和加载
- GUI对话框交互
- JDK路径优先级验证

### 5.3 手动测试场景
1. 未配置时自动检测
2. 配置有效JDK路径
3. 配置无效路径时的回退
4. Windows/macOS/Linux跨平台测试

## 6. 开发顺序

1. **Step 1**: 配置层 - 添加 `JDKSettings` 到 `config.py`
2. **Step 2**: 检测层 - 创建 `java_tools.py` 智能检测函数
3. **Step 3**: GUI层 - 创建 `jdk_config_dialog.py`
4. **Step 4**: 集成 - 更新主窗口菜单
5. **Step 5**: 工具集成 - 更新 `code_interpreter.py` 和 `compilation_handler.py`
6. **Step 6**: 测试 - 编写和运行测试
7. **Step 7**: 提交 - 小步提交每个完成的功能

## 7. 风险和注意事项

1. **向后兼容**: 现有配置文件不受影响，新字段使用默认值
2. **跨平台**: 需要测试 Windows、macOS、Linux 三平台
3. **JDK vs JRE**: 需要区分 JDK（有javac）和 JRE（无javac）
4. **版本检测**: 显示检测到的Java版本帮助用户确认
