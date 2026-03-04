# 增强工具减少失败和无意义重试的计划

## 问题分析

根据日志分析，当前系统存在以下问题：

### 1. 核心问题：缺少依赖检测和自动安装

日志显示的错误：
```
error: package org.junit.jupiter.api does not exist
error: package org.mockito does not exist
error: package org.assertj.core.api does not exist
```

这是典型的**测试依赖缺失**问题，但当前系统：
- 将其归类为 `COMPILATION_ERROR`
- 使用 `ANALYZE_AND_FIX` 策略尝试修复代码
- 进行了无意义的重试（等待16秒后重试）
- 最终失败，浪费了时间和资源

### 2. 当前错误恢复机制的局限

| 问题类型 | 当前处理 | 正确处理 |
|---------|---------|---------|
| 代码语法错误 | ANALYZE_AND_FIX ✅ | ANALYZE_AND_FIX |
| 缺少依赖 | ANALYZE_AND_FIX ❌ | INSTALL_DEPENDENCIES |
| 环境配置问题 | RETRY_WITH_BACKOFF ❌ | FIX_ENVIRONMENT |
| 类型不匹配 | ANALYZE_AND_FIX ✅ | ANALYZE_AND_FIX |

### 3. 缺失的能力

1. **依赖缺失检测**：无法识别 `package xxx does not exist` 是依赖问题
2. **自动安装依赖**：没有运行 `mvn dependency:resolve` 的能力
3. **环境预检查**：编译前不检查依赖完整性
4. **智能策略选择**：对环境问题使用错误的恢复策略

---

## 实施计划

### Phase 1: 增强错误分类（error_classification.py）

#### 1.1 新增错误子类型

```python
class ErrorSubCategory(Enum):
    """错误子类型细分"""
    # 编译错误子类型
    MISSING_DEPENDENCY = "missing_dependency"      # 缺少依赖
    MISSING_IMPORT = "missing_import"              # 缺少导入
    TYPE_MISMATCH = "type_mismatch"                # 类型不匹配
    SYNTAX_ERROR = "syntax_error"                  # 语法错误
    SYMBOL_NOT_FOUND = "symbol_not_found"          # 符号未找到
    
    # 工具执行错误子类型
    MAVEN_DEPENDENCY_ERROR = "maven_dependency_error"
    MAVEN_BUILD_ERROR = "maven_build_error"
    MAVEN_TEST_ERROR = "maven_test_error"
```

#### 1.2 增强错误分类器

在 `ErrorClassifier` 中添加依赖缺失检测：

```python
DEPENDENCY_ERROR_PATTERNS = [
    r"package\s+([\w.]+)\s+does not exist",
    r"cannot find symbol.*package",
    r"could not resolve dependencies",
    r"dependency.*not found",
    r"Failed to execute goal.*dependency",
]

def detect_missing_dependencies(compiler_output: str) -> List[str]:
    """检测缺失的依赖包"""
    missing = []
    for pattern in DEPENDENCY_ERROR_PATTERNS:
        matches = re.findall(pattern, compiler_output, re.IGNORECASE)
        missing.extend(matches)
    return list(set(missing))
```

### Phase 2: 新增恢复策略（error_recovery.py）

#### 2.1 新增恢复策略

```python
class RecoveryStrategy(Enum):
    # ... 现有策略 ...
    INSTALL_DEPENDENCIES = auto()        # 安装缺失依赖
    FIX_ENVIRONMENT = auto()             # 修复环境问题
    PRECOMPILE_CHECK = auto()            # 预编译检查
    RESOLVE_DEPENDENCIES = auto()        # 解析依赖
```

#### 2.2 依赖安装恢复处理器

```python
class DependencyRecoveryHandler:
    """依赖问题恢复处理器"""
    
    async def install_missing_dependencies(
        self, 
        missing_packages: List[str],
        project_path: str
    ) -> RecoveryResult:
        """安装缺失的依赖"""
        # 1. 运行 mvn dependency:resolve
        # 2. 如果失败，尝试 mvn dependency:tree 分析
        # 3. 如果是测试依赖，提示用户添加到 pom.xml
        
    async def resolve_test_dependencies(self, project_path: str) -> bool:
        """解析并下载测试依赖"""
        # 运行 mvn test-compile -DskipTests
```

### Phase 3: 增强编译流程（execution_steps.py）

#### 3.1 编译前依赖检查

```python
async def compile_tests(self) -> StepResult:
    """编译测试前先检查依赖"""
    
    # 1. 预检查：运行 mvn dependency:resolve
    deps_ok = await self._check_and_resolve_dependencies()
    if not deps_ok:
        return StepResult(
            success=False,
            state=AgentState.FAILED,
            message="Failed to resolve dependencies",
            data={"needs_dependency_install": True}
        )
    
    # 2. 正常编译流程
    result = await self._do_compile()
    
    # 3. 如果编译失败，分析是否是依赖问题
    if not result.success:
        missing_deps = self._detect_missing_dependencies(result.errors)
        if missing_deps:
            return StepResult(
                success=False,
                state=AgentState.FIXING,
                message=f"Missing dependencies: {missing_deps}",
                data={"missing_dependencies": missing_deps}
            )
    
    return result
```

#### 3.2 智能恢复策略选择

```python
async def _try_recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """根据错误类型选择正确的恢复策略"""
    
    error_info = self.error_classifier.get_detailed_error_info(error, context)
    
    # 根据错误子类型选择策略
    if error_info.get("sub_category") == "missing_dependency":
        return await self._recover_from_missing_dependency(error, context)
    
    if error_info.get("sub_category") == "maven_dependency_error":
        return await self._recover_from_maven_error(error, context)
    
    # 原有逻辑...
```

### Phase 4: Maven 工具增强（maven_tools.py）

#### 4.1 新增依赖管理方法

```python
class MavenRunner:
    # ... 现有方法 ...
    
    async def resolve_dependencies(self) -> Tuple[bool, str]:
        """解析并下载所有依赖
        
        Returns:
            (success, output) 元组
        """
        mvn = self._get_maven_executable()
        try:
            process = await asyncio.create_subprocess_exec(
                mvn, "dependency:resolve", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0, stderr.decode() if stderr else ""
        except Exception as e:
            return False, str(e)
    
    async def resolve_test_dependencies(self) -> Tuple[bool, str]:
        """解析并下载测试依赖
        
        运行 mvn test-compile -DskipTests 来下载测试依赖
        """
        mvn = self._get_maven_executable()
        try:
            process = await asyncio.create_subprocess_exec(
                mvn, "test-compile", "-DskipTests", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0, stderr.decode() if stderr else ""
        except Exception as e:
            return False, str(e)
    
    async def download_sources(self) -> bool:
        """下载依赖源码（可选）"""
        mvn = self._get_maven_executable()
        try:
            process = await asyncio.create_subprocess_exec(
                mvn, "dependency:sources", "-q",
                cwd=str(self.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False
    
    def check_pom_has_test_dependencies(self) -> Dict[str, bool]:
        """检查 pom.xml 是否包含常见测试依赖"""
        pom_path = self.project_path / "pom.xml"
        if not pom_path.exists():
            return {}
        
        content = pom_path.read_text()
        return {
            "junit_jupiter": "junit-jupiter" in content,
            "mockito": "mockito" in content,
            "assertj": "assertj" in content,
            "hamcrest": "hamcrest" in content,
        }
```

### Phase 5: 错误学习器增强（error_learner.py）

#### 5.1 学习依赖错误模式

```python
def _extract_compilation_keywords(self, error_message: str, context: Dict[str, Any]) -> List[str]:
    """增强编译错误关键词提取"""
    keywords = []
    
    # 检测依赖缺失
    if re.search(r"package\s+[\w.]+\s+does not exist", error_message):
        keywords.append("missing_dependency")
        
        # 提取缺失的包名
        packages = re.findall(r"package\s+([\w.]+)\s+does not exist", error_message)
        for pkg in packages:
            if "junit" in pkg.lower():
                keywords.append("missing_junit")
            elif "mockito" in pkg.lower():
                keywords.append("missing_mockito")
            elif "assertj" in pkg.lower():
                keywords.append("missing_assertj")
    
    # ... 其他检测 ...
    
    return keywords
```

### Phase 6: 配置增强

#### 6.1 新增配置项

在 `config.py` 中添加：

```python
class MavenSettings(BaseSettings):
    # ... 现有配置 ...
    
    auto_resolve_dependencies: bool = True
    dependency_resolve_timeout: int = 300  # 5分钟
    precompile_dependency_check: bool = True
```

---

## 实施步骤

### Step 1: 增强错误分类
- [ ] 在 `error_classification.py` 中添加 `ErrorSubCategory` 枚举
- [ ] 添加 `detect_missing_dependencies()` 函数
- [ ] 增强 `get_detailed_error_info()` 方法

### Step 2: 新增恢复策略
- [ ] 在 `error_recovery.py` 中添加新策略枚举值
- [ ] 创建 `DependencyRecoveryHandler` 类
- [ ] 实现 `install_missing_dependencies()` 方法

### Step 3: 增强 Maven 工具
- [ ] 在 `maven_tools.py` 中添加 `resolve_dependencies()` 方法
- [ ] 添加 `resolve_test_dependencies()` 方法
- [ ] 添加 `check_pom_has_test_dependencies()` 方法

### Step 4: 增强编译流程
- [ ] 在 `execution_steps.py` 中添加编译前依赖检查
- [ ] 增强 `_try_recover()` 方法，根据错误子类型选择策略
- [ ] 添加依赖安装恢复流程

### Step 5: 增强错误学习
- [ ] 在 `error_learner.py` 中增强关键词提取
- [ ] 学习依赖错误的最佳恢复策略

### Step 6: 测试验证
- [ ] 编写单元测试
- [ ] 集成测试：模拟缺少依赖的场景
- [ ] 验证自动安装依赖功能

---

## 预期效果

### Before (当前行为)
```
编译失败 -> ANALYZE_AND_FIX -> 等待16秒 -> 重试编译 -> 失败 -> 重试 -> 失败 -> 放弃
```

### After (增强后)
```
编译失败 -> 检测到缺少依赖 -> INSTALL_DEPENDENCIES -> mvn dependency:resolve -> 成功 -> 重试编译 -> 成功
```

### 收益
1. **减少无意义重试**：识别环境问题后直接修复，不重试代码
2. **提高成功率**：自动安装缺失依赖
3. **节省时间**：避免等待 backoff 时间
4. **更好的用户体验**：明确告知用户是依赖问题还是代码问题

---

## 风险评估

| 风险 | 影响 | 缓解措施 |
|-----|-----|---------|
| mvn dependency:resolve 超时 | 编译流程变长 | 设置合理超时，允许用户跳过 |
| 网络问题导致下载失败 | 依赖安装失败 | 提供重试机制，记录失败原因 |
| 错误识别不准确 | 使用错误策略 | 增加模式匹配准确度，允许回退 |

---

## 文件变更清单

| 文件 | 变更类型 | 描述 |
|-----|---------|-----|
| `pyutagent/core/error_classification.py` | 增强 | 添加依赖错误检测 |
| `pyutagent/core/error_recovery.py` | 增强 | 添加新恢复策略和处理器 |
| `pyutagent/tools/maven_tools.py` | 增强 | 添加依赖管理方法 |
| `pyutagent/agent/components/execution_steps.py` | 增强 | 添加编译前检查和智能恢复 |
| `pyutagent/core/error_learner.py` | 增强 | 学习依赖错误模式 |
| `pyutagent/core/config.py` | 增强 | 添加依赖相关配置 |
| `tests/unit/core/test_error_classification.py` | 新增 | 测试依赖错误检测 |
| `tests/unit/core/test_error_recovery.py` | 新增 | 测试依赖恢复策略 |
