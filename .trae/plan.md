# LLM 增强测试依赖添加实现计划

## 概述

本计划旨在利用 LLM 智能分析编译错误，自动识别缺失的测试依赖，生成准确的 Maven 依赖配置，并将其添加到 pom.xml 文件中，最后执行 `mvn clean install` 安装依赖。

## 当前架构分析

### 现有依赖管理机制

1. **依赖检测** ([error_classification.py](pyutagent/core/error_classification.py))
   - `detect_missing_dependencies()`: 使用正则表达式检测缺失的包
   - `DEPENDENCY_ERROR_PATTERNS`: 预定义的错误模式
   - `TEST_DEPENDENCY_PACKAGES`: 硬编码的测试依赖映射

2. **依赖恢复** ([error_recovery.py](pyutagent/core/error_recovery.py))
   - `DependencyRecoveryHandler`: 处理依赖问题恢复
   - `resolve_dependencies()`: 执行 `mvn dependency:resolve`
   - `resolve_test_dependencies()`: 执行 `mvn test-compile -DskipTests`
   - `suggest_pom_additions()`: 提供依赖添加建议（但不自动添加）

3. **Maven 工具** ([maven_tools.py](pyutagent/tools/maven_tools.py))
   - `MavenRunner`: Maven 命令执行器
   - `check_pom_has_test_dependencies()`: 检查 pom.xml 是否包含测试依赖

### 存在的问题

1. **依赖检测不够智能**
   - 基于简单正则表达式，可能误判或漏判
   - 无法识别复杂的依赖关系

2. **依赖信息不完整**
   - 只有包名，缺少 groupId、artifactId、version
   - 无法处理版本冲突

3. **无法自动添加依赖**
   - 只能提供建议，需要用户手动添加
   - 无法修改 pom.xml 文件

4. **缺少验证机制**
   - 添加依赖后无法验证是否正确
   - 无法处理依赖下载失败的情况

## 实现方案

### 1. LLM 增强的依赖分析器

#### 1.1 创建 `DependencyAnalyzer` 类

**位置**: `pyutagent/tools/dependency_analyzer.py`

**职责**:
- 利用 LLM 分析编译错误
- 识别缺失的依赖包
- 生成完整的 Maven 依赖坐标 (groupId, artifactId, version, scope)
- 处理传递依赖和版本冲突

**核心方法**:
```python
class DependencyAnalyzer:
    def __init__(self, llm_client, prompt_builder):
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
    
    async def analyze_missing_dependencies(
        self, 
        compiler_output: str,
        current_pom_content: str
    ) -> Dict[str, Any]:
        """分析编译错误，识别缺失的依赖
        
        Returns:
            {
                "missing_dependencies": [
                    {
                        "group_id": "org.junit.jupiter",
                        "artifact_id": "junit-jupiter",
                        "version": "5.9.3",
                        "scope": "test",
                        "reason": "Used for unit testing"
                    }
                ],
                "confidence": 0.95,
                "analysis": "..."
            }
        }
        """
    
    def build_dependency_analysis_prompt(
        self,
        compiler_output: str,
        current_pom_content: str
    ) -> str:
        """构建依赖分析提示词"""
```

#### 1.2 Prompt 设计

**依赖分析 Prompt**:
```
You are a Maven dependency expert. Analyze the following compilation errors and identify missing dependencies.

Compilation Errors:
```
{compiler_output}
```

Current pom.xml dependencies section:
```
{current_pom_dependencies}
```

Task:
1. Identify all missing dependencies from the compilation errors
2. For each missing dependency, provide:
   - groupId: Maven group ID
   - artifactId: Maven artifact ID
   - version: Recommended version (use latest stable if unsure)
   - scope: Dependency scope (compile, test, provided, runtime)
   - reason: Why this dependency is needed

Output in JSON format:
{
  "missing_dependencies": [
    {
      "group_id": "...",
      "artifact_id": "...",
      "version": "...",
      "scope": "...",
      "reason": "..."
    }
  ],
  "confidence": 0.0-1.0,
  "analysis": "Brief analysis of the errors"
}
```

### 2. POM 文件编辑器

#### 2.1 创建 `PomEditor` 类

**位置**: `pyutagent/tools/pom_editor.py`

**职责**:
- 解析 pom.xml 文件
- 添加依赖到 pom.xml
- 处理依赖冲突
- 备份和恢复 pom.xml

**核心方法**:
```python
class PomEditor:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.pom_path = self.project_path / "pom.xml"
        self.backup_dir = self.project_path / ".pyutagent" / "pom_backups"
    
    def read_pom(self) -> str:
        """读取 pom.xml 内容"""
    
    def backup_pom(self) -> str:
        """备份 pom.xml，返回备份路径"""
    
    def restore_pom(self, backup_path: str) -> bool:
        """从备份恢复 pom.xml"""
    
    def add_dependency(
        self, 
        dependency: Dict[str, str],
        position: str = "end"  # "end" or "after_existing"
    ) -> Tuple[bool, str]:
        """添加依赖到 pom.xml
        
        Args:
            dependency: {
                "group_id": "...",
                "artifact_id": "...",
                "version": "...",
                "scope": "..."
            }
            position: 添加位置
            
        Returns:
            (success, message)
        """
    
    def add_dependencies(
        self, 
        dependencies: List[Dict[str, str]]
    ) -> Tuple[bool, List[str]]:
        """批量添加依赖"""
    
    def has_dependency(self, group_id: str, artifact_id: str) -> bool:
        """检查是否已存在依赖"""
    
    def get_dependencies_section(self) -> str:
        """获取 dependencies 部分的内容"""
    
    def find_dependencies_section(self, content: str) -> Tuple[int, int]:
        """找到 <dependencies> 标签的位置"""
    
    def format_dependency_xml(self, dependency: Dict[str, str]) -> str:
        """格式化依赖为 XML"""
```

#### 2.2 XML 处理策略

使用 `xml.etree.ElementTree` 进行安全的 XML 操作：

```python
def add_dependency_safe(self, dependency: Dict[str, str]) -> bool:
    """使用 XML 解析器安全添加依赖"""
    try:
        tree = ET.parse(self.pom_path)
        root = tree.getroot()
        
        # Maven POM namespace
        ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
        
        # Find or create dependencies section
        dependencies = root.find('m:dependencies', ns)
        if dependencies is None:
            dependencies = ET.SubElement(root, '{%s}dependencies' % ns['m'])
        
        # Check if dependency already exists
        for dep in dependencies.findall('m:dependency', ns):
            if (dep.find('m:groupId', ns).text == dependency['group_id'] and
                dep.find('m:artifactId', ns).text == dependency['artifact_id']):
                return False  # Already exists
        
        # Add new dependency
        new_dep = ET.SubElement(dependencies, '{%s}dependency' % ns['m'])
        ET.SubElement(new_dep, '{%s}groupId' % ns['m']).text = dependency['group_id']
        ET.SubElement(new_dep, '{%s}artifactId' % ns['m']).text = dependency['artifact_id']
        ET.SubElement(new_dep, '{%s}version' % ns['m']).text = dependency['version']
        if dependency.get('scope'):
            ET.SubElement(new_dep, '{%s}scope' % ns['m']).text = dependency['scope']
        
        # Write back with proper formatting
        tree.write(self.pom_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        logger.error(f"Failed to add dependency: {e}")
        return False
```

### 3. 依赖安装验证器

#### 3.1 创建 `DependencyInstaller` 类

**位置**: `pyutagent/tools/dependency_installer.py`

**职责**:
- 执行 `mvn clean install`
- 验证依赖是否成功安装
- 处理安装失败的情况
- 提供回滚机制

**核心方法**:
```python
class DependencyInstaller:
    def __init__(self, project_path: str, maven_runner: MavenRunner):
        self.project_path = project_path
        self.maven_runner = maven_runner
        self.pom_editor = PomEditor(project_path)
    
    async def install_dependencies(
        self,
        dependencies: List[Dict[str, str]],
        skip_tests: bool = True
    ) -> InstallResult:
        """安装依赖
        
        流程:
        1. 备份 pom.xml
        2. 添加依赖到 pom.xml
        3. 执行 mvn clean install
        4. 验证安装结果
        5. 如果失败，恢复备份
        
        Returns:
            InstallResult(success, message, installed_deps, failed_deps)
        """
    
    async def verify_dependencies(
        self,
        dependencies: List[Dict[str, str]]
    ) -> Tuple[bool, List[str]]:
        """验证依赖是否可用"""
    
    async def rollback(self, backup_path: str) -> bool:
        """回滚到之前的 pom.xml"""
```

### 4. 集成到错误恢复流程

#### 4.1 增强 `DependencyRecoveryHandler`

**位置**: `pyutagent/core/error_recovery.py`

**修改内容**:
```python
class DependencyRecoveryHandler:
    def __init__(
        self,
        project_path: str,
        llm_client: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        maven_runner: Optional[Any] = None,
        timeout: int = 300,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ):
        self.project_path = project_path
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.maven_runner = maven_runner
        self.timeout = timeout
        self.progress_callback = progress_callback
        
        # 新增组件
        self.dependency_analyzer = DependencyAnalyzer(llm_client, prompt_builder)
        self.pom_editor = PomEditor(project_path)
        self.dependency_installer = DependencyInstaller(project_path, maven_runner)
    
    async def install_missing_dependencies_enhanced(
        self,
        compiler_output: str
    ) -> RecoveryResult:
        """增强的依赖安装流程
        
        流程:
        1. 使用 LLM 分析编译错误
        2. 识别缺失的依赖
        3. 添加依赖到 pom.xml
        4. 执行 mvn clean install
        5. 验证安装结果
        """
        if self.progress_callback:
            self.progress_callback("ANALYZING_DEPS", "正在分析缺失的依赖...")
        
        # 1. LLM 分析
        analysis_result = await self.dependency_analyzer.analyze_missing_dependencies(
            compiler_output,
            self.pom_editor.read_pom()
        )
        
        missing_deps = analysis_result.get("missing_dependencies", [])
        if not missing_deps:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                attempts_made=1,
                error_message="No missing dependencies detected",
                action="skip",
                should_continue=True
            )
        
        if self.progress_callback:
            self.progress_callback(
                "INSTALLING_DEPS", 
                f"正在安装 {len(missing_deps)} 个依赖..."
            )
        
        # 2. 安装依赖
        install_result = await self.dependency_installer.install_dependencies(
            missing_deps,
            skip_tests=True
        )
        
        if install_result.success:
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                attempts_made=1,
                recovered_data={
                    "installed_dependencies": install_result.installed_deps,
                    "analysis": analysis_result.get("analysis", "")
                },
                action="retry",
                should_continue=True
            )
        else:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                attempts_made=1,
                error_message=install_result.message,
                action="escalate",
                should_continue=False,
                details={
                    "failed_dependencies": install_result.failed_deps,
                    "suggested_fixes": analysis_result.get("suggested_fixes", [])
                }
            )
```

#### 4.2 更新错误恢复策略

**位置**: `pyutagent/core/error_recovery.py`

**修改 `_install_dependencies` 方法**:
```python
async def _install_dependencies(
    self,
    context: RecoveryContext,
    llm_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """安装缺失的依赖（增强版）"""
    if self.progress_callback:
        self.progress_callback("INSTALLING_DEPS", "正在安装缺失的依赖...")
    
    logger.info("[ErrorRecoveryManager] Installing missing dependencies")
    
    compiler_output = context.error_details.get("compiler_output", context.error_message)
    
    # 使用增强的依赖恢复处理器
    handler = DependencyRecoveryHandler(
        project_path=self.project_path,
        llm_client=self.llm_client,
        prompt_builder=self.prompt_builder,
        progress_callback=self.progress_callback
    )
    
    result = await handler.install_missing_dependencies_enhanced(compiler_output)
    
    if result.success:
        return {
            "success": True,
            "action": "retry",
            "message": f"Dependencies installed: {result.details.get('installed_dependencies', [])}",
            "should_continue": True,
            "strategy": "install_dependencies",
            "installed_packages": result.details.get('installed_dependencies', [])
        }
    else:
        return {
            "success": False,
            "action": "escalate",
            "message": f"Failed to install dependencies: {result.error_message}",
            "should_continue": False,
            "strategy": "install_dependencies",
            "details": result.details
        }
```

### 5. Prompt 模板扩展

#### 5.1 添加依赖分析 Prompt

**位置**: `pyutagent/agent/prompts.py`

**新增方法**:
```python
def build_dependency_analysis_prompt(
    self,
    compiler_output: str,
    current_pom_content: str
) -> str:
    """构建依赖分析提示词"""
    return f"""You are a Maven dependency expert. Analyze the following compilation errors and identify missing dependencies.

Compilation Errors:
```
{compiler_output}
```

Current pom.xml:
```
{current_pom_content}
```

Task:
1. Identify all missing dependencies from the compilation errors
2. For each missing dependency, provide complete Maven coordinates
3. Determine the appropriate scope (test, compile, provided, runtime)
4. Recommend stable versions

Output in JSON format:
{{
  "missing_dependencies": [
    {{
      "group_id": "org.junit.jupiter",
      "artifact_id": "junit-jupiter",
      "version": "5.10.0",
      "scope": "test",
      "reason": "JUnit 5 testing framework"
    }}
  ],
  "confidence": 0.95,
  "analysis": "Brief analysis of missing dependencies"
}}

Important:
- Use latest stable versions for common libraries
- Test dependencies should have scope "test"
- Be precise with groupId and artifactId
- If uncertain, set lower confidence score"""

def build_comprehensive_fix_prompt(
    self,
    error_category: str,
    error_message: str,
    error_details: Dict[str, Any],
    local_analysis: Dict[str, Any],
    llm_insights: str,
    specific_fixes: List[str],
    current_test_code: Optional[str] = None,
    target_class_info: Optional[Dict[str, Any]] = None,
    attempt_history: List[Dict[str, Any]] = None
) -> str:
    """构建综合修复提示词（已存在，可能需要增强）"""
    # ... 现有实现
```

## 实现步骤

### Phase 1: 核心组件开发 (2-3 天)

1. **创建 `DependencyAnalyzer`** (0.5 天)
   - 实现 LLM 分析逻辑
   - 设计 Prompt 模板
   - 编写单元测试

2. **创建 `PomEditor`** (1 天)
   - 实现 XML 解析和编辑
   - 实现备份和恢复机制
   - 处理边界情况（无 dependencies 标签等）
   - 编写单元测试

3. **创建 `DependencyInstaller`** (0.5 天)
   - 集成 Maven 命令执行
   - 实现验证逻辑
   - 编写单元测试

### Phase 2: 集成和测试 (1-2 天)

4. **增强 `DependencyRecoveryHandler`** (0.5 天)
   - 集成新组件
   - 更新恢复流程

5. **更新错误恢复流程** (0.5 天)
   - 修改 `_install_dependencies` 方法
   - 测试集成

6. **端到端测试** (1 天)
   - 创建测试 Maven 项目
   - 测试各种依赖缺失场景
   - 验证完整流程

### Phase 3: 优化和文档 (1 天)

7. **性能优化**
   - 缓存常用依赖信息
   - 优化 LLM 调用次数

8. **错误处理增强**
   - 处理网络超时
   - 处理 Maven 仓库不可用
   - 处理版本冲突

9. **文档和示例**
   - 更新 README
   - 添加使用示例
   - 编写最佳实践

## 测试计划

### 单元测试

1. **DependencyAnalyzer 测试**
   - 测试 LLM 响应解析
   - 测试各种编译错误格式
   - 测试边界情况

2. **PomEditor 测试**
   - 测试 XML 解析
   - 测试依赖添加
   - 测试备份和恢复
   - 测试重复依赖检测

3. **DependencyInstaller 测试**
   - 测试安装流程
   - 测试验证逻辑
   - 测试回滚机制

### 集成测试

1. **完整流程测试**
   - 创建缺少 JUnit 依赖的项目
   - 创建缺少 Mockito 依赖的项目
   - 创建缺少多个依赖的项目

2. **错误场景测试**
   - Maven 仓库不可用
   - 版本冲突
   - 无效的依赖坐标

### 端到端测试

1. **真实项目测试**
   - 在实际项目中测试
   - 验证覆盖率提升
   - 验证测试通过率

## 风险和缓解措施

### 风险 1: LLM 分析不准确

**缓解措施**:
- 使用结构化输出（JSON）
- 设置置信度阈值
- 提供用户确认机制
- 回退到传统正则表达式方法

### 风险 2: Maven 仓库不可用

**缓解措施**:
- 实现重试机制
- 支持配置镜像仓库
- 提供离线模式建议

### 风险 3: 版本冲突

**缓解措施**:
- 使用 `mvn dependency:tree` 分析
- 提供版本冲突检测
- 建议使用 dependencyManagement

### 风险 4: pom.xml 损坏

**缓解措施**:
- 自动备份机制
- 提供回滚功能
- XML 格式验证

## 成功指标

1. **功能指标**
   - 依赖识别准确率 > 90%
   - 自动安装成功率 > 85%
   - 平均安装时间 < 60 秒

2. **质量指标**
   - 单元测试覆盖率 > 80%
   - 集成测试覆盖率 > 70%
   - 零 pom.xml 损坏案例

3. **用户体验指标**
   - 减少手动添加依赖的次数
   - 提高测试生成成功率
   - 降低错误恢复时间

## 后续优化方向

1. **智能版本推荐**
   - 基于项目其他依赖推荐兼容版本
   - 支持版本范围指定

2. **依赖冲突自动解决**
   - 自动检测版本冲突
   - 提供解决方案建议

3. **依赖缓存和预加载**
   - 缓存常用依赖信息
   - 预加载常见测试依赖

4. **多构建工具支持**
   - 支持 Gradle
   - 支持 Bazel

## 总结

本计划通过引入 LLM 智能分析、自动编辑 pom.xml、验证安装结果等机制，实现了测试依赖的自动添加和安装。该方案能够显著提升用户体验，减少手动配置工作，提高测试生成的成功率。
