"""Project Configuration System - 项目配置系统

类似 Claude Code 的 CLAUDE.md，实现 PYUT.md 项目配置系统，
让 Agent 能快速理解项目上下文。
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from enum import Enum
import json
import logging
import time
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


class BuildTool(Enum):
    """构建工具枚举"""
    MAVEN = "maven"
    GRADLE = "gradle"
    BAZEL = "bazel"
    ANT = "ant"
    UNKNOWN = "unknown"


class TestFramework(Enum):
    """测试框架枚举"""
    JUNIT5 = "junit5"
    JUNIT4 = "junit4"
    TESTNG = "testng"
    SPOCK = "spock"
    UNKNOWN = "unknown"


class MockFramework(Enum):
    """Mock 框架枚举"""
    MOCKITO = "mockito"
    EASYMOCK = "easymock"
    JMOCK = "jmock"
    POWERMOCK = "powermock"
    UNKNOWN = "unknown"


@dataclass
class BuildCommands:
    """构建命令配置"""
    build: str = "mvn compile"
    test: str = "mvn test"
    test_single: str = "mvn test -Dtest={test_class}"
    coverage: str = "mvn jacoco:report"
    clean: str = "mvn clean"
    package: str = "mvn package"
    install: str = "mvn install"
    
    def format_command(self, command: str, **kwargs) -> str:
        """格式化命令"""
        cmd = getattr(self, command, "")
        return cmd.format(**kwargs) if kwargs else cmd


@dataclass
class CodingStandards:
    """编码规范配置"""
    style_guide: str = "google_java_format"
    max_line_length: int = 120
    indent_size: int = 4
    use_tabs: bool = False
    
    # 命名规范
    class_naming: str = "PascalCase"
    method_naming: str = "camelCase"
    constant_naming: str = "UPPER_SNAKE_CASE"
    
    # 代码质量
    require_javadoc: bool = True
    max_method_lines: int = 50
    max_class_lines: int = 500


@dataclass
class TestPreferences:
    """测试偏好配置"""
    test_framework: TestFramework = TestFramework.JUNIT5
    mock_framework: MockFramework = MockFramework.MOCKITO
    coverage_threshold: float = 0.8
    
    # 测试命名
    test_class_suffix: str = "Test"
    test_method_prefix: str = "test"
    
    # 测试策略
    generate_positive_cases: bool = True
    generate_negative_cases: bool = True
    generate_edge_cases: bool = True
    generate_boundary_cases: bool = True
    
    # 测试模式
    prefer_parametrized_tests: bool = False
    use_given_when_then: bool = False
    use_arrange_act_assert: bool = True


@dataclass
class ProjectContext:
    """项目上下文配置"""
    # 基本信息
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # 技术栈
    language: str = "java"
    java_version: str = "17"
    build_tool: BuildTool = BuildTool.MAVEN
    
    # 配置对象
    build_commands: BuildCommands = field(default_factory=BuildCommands)
    coding_standards: CodingStandards = field(default_factory=CodingStandards)
    test_preferences: TestPreferences = field(default_factory=TestPreferences)
    
    # 项目结构
    source_dirs: List[str] = field(default_factory=lambda: ["src/main/java"])
    test_dirs: List[str] = field(default_factory=lambda: ["src/test/java"])
    resource_dirs: List[str] = field(default_factory=lambda: ["src/main/resources"])
    
    # 模块信息
    key_modules: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    
    # 测试配置
    test_patterns: List[str] = field(default_factory=lambda: ["**/*Test.java"])
    exclude_patterns: List[str] = field(default_factory=list)
    
    # 工作流
    common_workflows: Dict[str, str] = field(default_factory=dict)
    
    # 架构信息
    architecture: str = ""
    design_patterns: List[str] = field(default_factory=list)
    
    # 元数据
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换枚举类型
        data['build_tool'] = self.build_tool.value
        data['test_preferences']['test_framework'] = self.test_preferences.test_framework.value
        data['test_preferences']['mock_framework'] = self.test_preferences.mock_framework.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectContext':
        """从字典创建"""
        # 处理枚举
        if 'build_tool' in data:
            data['build_tool'] = BuildTool(data['build_tool'])
        if 'test_preferences' in data:
            tp = data['test_preferences']
            if 'test_framework' in tp:
                tp['test_framework'] = TestFramework(tp['test_framework'])
            if 'mock_framework' in tp:
                tp['mock_framework'] = MockFramework(tp['mock_framework'])
        
        # 处理嵌套 dataclass
        if 'build_commands' in data:
            data['build_commands'] = BuildCommands(**data['build_commands'])
        if 'coding_standards' in data:
            data['coding_standards'] = CodingStandards(**data['coding_standards'])
        if 'test_preferences' in data:
            data['test_preferences'] = TestPreferences(**data['test_preferences'])
        
        return cls(**data)


class ProjectConfigManager:
    """项目配置管理器
    
    管理 PYUT.md 配置文件，类似于 Claude Code 的 CLAUDE.md
    """
    
    CONFIG_FILENAME = "PYUT.md"
    CONFIG_JSON = ".pyutagent/config.json"
    
    def __init__(self, project_root: Union[str, Path]):
        self.project_root = Path(project_root).resolve()
        self.config_path = self.project_root / self.CONFIG_FILENAME
        self.json_config_path = self.project_root / self.CONFIG_JSON
        self._context: Optional[ProjectContext] = None
    
    def init_config(self, force: bool = False) -> bool:
        """初始化项目配置文件
        
        类似于 Claude Code 的 /init 命令
        
        Args:
            force: 是否强制覆盖已有配置
            
        Returns:
            bool: 是否成功创建配置
        """
        if self.config_path.exists() and not force:
            logger.info(f"Config already exists: {self.config_path}")
            return False
        
        try:
            # 分析项目结构
            context = self._analyze_project()
            
            # 生成配置文件
            self._write_config(context)
            
            logger.info(f"Created config: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to init config: {e}")
            return False
    
    def _analyze_project(self) -> ProjectContext:
        """分析项目结构"""
        context = ProjectContext(
            name=self.project_root.name,
            description=f"Project {self.project_root.name}"
        )
        
        # 检测构建工具
        context.build_tool = self._detect_build_tool()
        context.build_commands = self._get_build_commands(context.build_tool)
        
        # 检测 Java 版本
        context.java_version = self._detect_java_version(context.build_tool)
        
        # 检测测试框架
        context.test_preferences.test_framework = self._detect_test_framework()
        context.test_preferences.mock_framework = self._detect_mock_framework()
        
        # 分析项目结构
        context.key_modules = self._detect_modules()
        context.source_dirs = self._detect_source_dirs()
        context.test_dirs = self._detect_test_dirs()
        
        # 检测外部依赖
        context.external_dependencies = self._detect_external_dependencies()
        
        return context
    
    def _detect_build_tool(self) -> BuildTool:
        """检测构建工具"""
        if (self.project_root / "pom.xml").exists():
            return BuildTool.MAVEN
        elif (self.project_root / "build.gradle").exists():
            return BuildTool.GRADLE
        elif (self.project_root / "build.gradle.kts").exists():
            return BuildTool.GRADLE
        elif (self.project_root / "BUILD").exists() or (self.project_root / "WORKSPACE").exists():
            return BuildTool.BAZEL
        elif (self.project_root / "build.xml").exists():
            return BuildTool.ANT
        return BuildTool.UNKNOWN
    
    def _get_build_commands(self, build_tool: BuildTool) -> BuildCommands:
        """获取构建命令"""
        commands_map = {
            BuildTool.MAVEN: BuildCommands(
                build="mvn compile",
                test="mvn test",
                test_single="mvn test -Dtest={test_class}",
                coverage="mvn jacoco:report",
                clean="mvn clean",
                package="mvn package",
                install="mvn install"
            ),
            BuildTool.GRADLE: BuildCommands(
                build="gradle build",
                test="gradle test",
                test_single="gradle test --tests {test_class}",
                coverage="gradle jacocoTestReport",
                clean="gradle clean",
                package="gradle jar",
                install="gradle publishToMavenLocal"
            ),
            BuildTool.BAZEL: BuildCommands(
                build="bazel build //...",
                test="bazel test //...",
                test_single="bazel test {test_class}",
                coverage="bazel coverage //...",
                clean="bazel clean",
                package="bazel build //:package",
                install=""
            ),
            BuildTool.ANT: BuildCommands(
                build="ant compile",
                test="ant test",
                test_single="ant test -Dtest.class={test_class}",
                coverage="",
                clean="ant clean",
                package="ant jar",
                install=""
            ),
            BuildTool.UNKNOWN: BuildCommands()
        }
        return commands_map.get(build_tool, BuildCommands())
    
    def _detect_java_version(self, build_tool: BuildTool) -> str:
        """检测 Java 版本"""
        if build_tool == BuildTool.MAVEN:
            pom_path = self.project_root / "pom.xml"
            if pom_path.exists():
                try:
                    tree = ET.parse(pom_path)
                    root = tree.getroot()
                    
                    # 查找 java.version 或 maven.compiler.source
                    ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
                    
                    # 先查找 properties
                    for prop in root.findall('.//m:properties/*', ns):
                        if prop.tag.endswith('java.version') or prop.tag.endswith('maven.compiler.source'):
                            return prop.text or "17"
                    
                    # 再查找 plugin configuration
                    for config in root.findall('.//m:configuration/*', ns):
                        if config.tag.endswith('source') or config.tag.endswith('target'):
                            return config.text or "17"
                            
                except Exception as e:
                    logger.warning(f"Failed to parse pom.xml: {e}")
        
        elif build_tool == BuildTool.GRADLE:
            gradle_path = self.project_root / "build.gradle"
            if gradle_path.exists():
                content = gradle_path.read_text()
                # 查找 sourceCompatibility 或 JavaVersion
                if 'sourceCompatibility' in content:
                    import re
                    match = re.search(r'sourceCompatibility\s*=\s*[\'"]?([\d.]+)[\'"]?', content)
                    if match:
                        return match.group(1)
        
        return "17"  # 默认 Java 17
    
    def _detect_test_framework(self) -> TestFramework:
        """检测测试框架"""
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            content = pom_path.read_text()
            if 'junit-jupiter' in content or 'junit5' in content:
                return TestFramework.JUNIT5
            elif 'junit' in content:
                return TestFramework.JUNIT4
            elif 'testng' in content:
                return TestFramework.TESTNG
            elif 'spock' in content:
                return TestFramework.SPOCK
        
        gradle_path = self.project_root / "build.gradle"
        if gradle_path.exists():
            content = gradle_path.read_text()
            if 'junit-jupiter' in content:
                return TestFramework.JUNIT5
            elif 'testng' in content:
                return TestFramework.TESTNG
            elif 'spock' in content:
                return TestFramework.SPOCK
        
        return TestFramework.JUNIT5
    
    def _detect_mock_framework(self) -> MockFramework:
        """检测 Mock 框架"""
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            content = pom_path.read_text()
            if 'mockito' in content:
                return MockFramework.MOCKITO
            elif 'easymock' in content:
                return MockFramework.EASYMOCK
            elif 'jmock' in content:
                return MockFramework.JMOCK
            elif 'powermock' in content:
                return MockFramework.POWERMOCK
        
        gradle_path = self.project_root / "build.gradle"
        if gradle_path.exists():
            content = gradle_path.read_text()
            if 'mockito' in content:
                return MockFramework.MOCKITO
            elif 'easymock' in content:
                return MockFramework.EASYMOCK
        
        return MockFramework.MOCKITO
    
    def _detect_modules(self) -> List[str]:
        """检测项目模块"""
        modules = []
        
        # 检查 Maven 多模块项目
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            try:
                tree = ET.parse(pom_path)
                root = tree.getroot()
                ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
                
                for module in root.findall('.//m:modules/m:module', ns):
                    if module.text:
                        modules.append(module.text)
                
                if modules:
                    return modules
            except Exception:
                pass
        
        # 检查源码目录
        src_main = self.project_root / "src" / "main" / "java"
        if src_main.exists():
            for item in src_main.iterdir():
                if item.is_dir():
                    modules.append(item.name)
        
        return modules[:10]  # 最多返回10个
    
    def _detect_source_dirs(self) -> List[str]:
        """检测源码目录"""
        dirs = []
        
        # 标准 Maven/Gradle 结构
        if (self.project_root / "src" / "main" / "java").exists():
            dirs.append("src/main/java")
        
        # 检查其他常见位置
        for path in ["src", "java", "source", "sources"]:
            if (self.project_root / path).is_dir():
                dirs.append(path)
        
        return dirs if dirs else ["src/main/java"]
    
    def _detect_test_dirs(self) -> List[str]:
        """检测测试目录"""
        dirs = []
        
        # 标准 Maven/Gradle 结构
        if (self.project_root / "src" / "test" / "java").exists():
            dirs.append("src/test/java")
        
        return dirs if dirs else ["src/test/java"]
    
    def _detect_external_dependencies(self) -> List[str]:
        """检测外部依赖"""
        dependencies = []
        
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            try:
                tree = ET.parse(pom_path)
                root = tree.getroot()
                ns = {'m': 'http://maven.apache.org/POM/4.0.0'}
                
                for dep in root.findall('.//m:dependencies/m:dependency', ns):
                    group_id = dep.find('m:groupId', ns)
                    artifact_id = dep.find('m:artifactId', ns)
                    if group_id is not None and artifact_id is not None:
                        dependencies.append(f"{group_id.text}:{artifact_id.text}")
            except Exception:
                pass
        
        return dependencies[:20]  # 最多返回20个
    
    def _write_config(self, context: ProjectContext) -> None:
        """写入配置文件"""
        # 生成 Markdown 配置
        md_content = self._generate_md_config(context)
        self.config_path.write_text(md_content, encoding='utf-8')
        
        # 同时写入 JSON 格式便于程序读取
        self.json_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_config_path.write_text(
            json.dumps(context.to_dict(), indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
    
    def _generate_md_config(self, context: ProjectContext) -> str:
        """生成 Markdown 格式配置"""
        tf = context.test_preferences.test_framework.value
        mf = context.test_preferences.mock_framework.value
        bt = context.build_tool.value
        
        return f"""# PyUT Agent Configuration

## Project Context

When working with this codebase, prioritize readability over cleverness.
Ask clarifying questions when requirements are ambiguous.

### Project Information
- **Name**: {context.name}
- **Description**: {context.description}
- **Version**: {context.version}
- **Language**: {context.language}
- **Build Tool**: {bt}
- **Java Version**: {context.java_version}
- **Test Framework**: {tf}
- **Mock Framework**: {mf}

### Architecture
{context.architecture or "Standard " + bt + " project structure"}

### Key Modules
{chr(10).join(f"- {m}" for m in context.key_modules) or "- Main source module"}

### Source Directories
{chr(10).join(f"- {d}" for d in context.source_dirs)}

### Test Directories
{chr(10).join(f"- {d}" for d in context.test_dirs)}

## Build Commands

```bash
# Build the project
{context.build_commands.build}

# Run all tests
{context.build_commands.test}

# Run single test class
{context.build_commands.test_single}

# Generate coverage report
{context.build_commands.coverage}

# Clean build artifacts
{context.build_commands.clean}

# Package the project
{context.build_commands.package}

# Install to local repository
{context.build_commands.install}
```

## Coding Standards

- Follow existing code style in the project
- Use meaningful variable and method names
- Add Javadoc for public APIs
- Keep methods focused and small
- Write unit tests for new functionality
- Maximum line length: {context.coding_standards.max_line_length}
- Indent size: {context.coding_standards.indent_size}

## Test Generation Preferences

- Use **{tf}** for all new tests
- Mock external dependencies with **{mf}**
- Target **{context.test_preferences.coverage_threshold*100:.0f}%** code coverage
- Include positive and negative test cases
- Use descriptive test method names
- Test class suffix: `{context.test_preferences.test_class_suffix}`

## Common Workflows

### Generate tests for a class
```
Generate unit tests for UserService
```

### Fix compilation errors
```
Fix compilation errors in OrderServiceTest
```

### Improve coverage
```
Improve test coverage for payment module
```

### Refactor code
```
Refactor UserService to use dependency injection
```

### Fix a bug
```
Fix the null pointer exception in OrderProcessor
```

## External Dependencies

{chr(10).join(f"- {dep}" for dep in context.external_dependencies[:10]) or "- See pom.xml or build.gradle for full list"}

---

*This configuration was auto-generated by PyUT Agent.*
*Last updated: {time.strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    def load_context(self) -> Optional[ProjectContext]:
        """加载项目上下文"""
        if self._context is not None:
            return self._context
        
        # 优先读取 JSON 配置
        if self.json_config_path.exists():
            try:
                data = json.loads(self.json_config_path.read_text())
                self._context = ProjectContext.from_dict(data)
                logger.info(f"Loaded config from {self.json_config_path}")
                return self._context
            except Exception as e:
                logger.warning(f"Failed to load JSON config: {e}")
        
        # 回退到重新分析项目
        logger.info("Analyzing project structure...")
        self._context = self._analyze_project()
        return self._context
    
    def reload_context(self) -> Optional[ProjectContext]:
        """重新加载项目上下文"""
        self._context = None
        return self.load_context()
    
    def save_context(self, context: ProjectContext) -> bool:
        """保存项目上下文"""
        try:
            context.updated_at = time.time()
            self._write_config(context)
            self._context = context
            return True
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
            return False
    
    def get_prompt_context(self) -> str:
        """获取用于 LLM prompt 的上下文"""
        context = self.load_context()
        if not context:
            return ""
        
        return f"""
Project Context:
- Name: {context.name}
- Language: {context.language}
- Build Tool: {context.build_tool.value}
- Java Version: {context.java_version}
- Test Framework: {context.test_preferences.test_framework.value}
- Mock Framework: {context.test_preferences.mock_framework.value}

Build Commands:
- Build: {context.build_commands.build}
- Test: {context.build_commands.test}
- Coverage: {context.build_commands.coverage}

Key Modules: {', '.join(context.key_modules[:5])}
"""
    
    def config_exists(self) -> bool:
        """检查配置是否存在"""
        return self.config_path.exists() or self.json_config_path.exists()
    
    def get_config_path(self) -> Path:
        """获取配置文件路径"""
        return self.config_path if self.config_path.exists() else self.json_config_path


# 全局配置管理器缓存
_config_managers: Dict[str, ProjectConfigManager] = {}


def get_config_manager(project_root: Union[str, Path]) -> ProjectConfigManager:
    """获取项目配置管理器（带缓存）"""
    path = str(Path(project_root).resolve())
    if path not in _config_managers:
        _config_managers[path] = ProjectConfigManager(path)
    return _config_managers[path]
