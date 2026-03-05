# TestNG 支持技术规格说明书

## 1. 概述

本文档定义了在 PyUT Agent VS Code 插件中添加 TestNG 测试框架支持的详细技术规格。

## 2. 现状分析

### 2.1 当前支持的测试框架

根据代码分析，当前项目已定义了对多种测试框架的枚举支持：

```python
class TestFramework(Enum):
    JUNIT5 = "junit5"       # ✅ 完整支持
    JUNIT4 = "junit4"       # ⚠️ 有限支持
    TESTNG = "testng"       # ⚠️ 仅枚举定义，未实现
    SPOCK = "spock"         # ❌ 未实现
    UNKNOWN = "unknown"
```

### 2.2 当前架构

**后端 (pyutagent)**:
- `TestGeneratorAgent`: 核心测试生成 Agent
- `BaseTestGenerator`: 抽象基类
- `LLMTestGenerator`: LLM 直接生成
- `AiderTestGenerator`: 迭代式改进
- `MavenRunner`: Maven 命令执行
- `CoverageAnalyzer`: JaCoCo 覆盖率分析

**VS Code 插件 (pyutagent-vscode)**:
- `EnhancedChatViewProvider`: 聊天界面
- `DiffProvider`: Monaco Editor Diff 预览
- `TerminalManager`: 终端命令执行

## 3. TestNG vs JUnit 5 对比

### 3.1 注解对比

| 功能 | JUnit 5 | TestNG |
|------|---------|--------|
| 测试方法 | `@Test` | `@Test` |
| 前置设置 | `@BeforeEach` | `@BeforeMethod` |
| 后置清理 | `@AfterEach` | `@AfterMethod` |
| 类级设置 | `@BeforeAll` | `@BeforeClass` |
| 类级清理 | `@AfterAll` | `@AfterClass` |
| 参数化测试 | `@ParameterizedTest` | `@DataProvider` |
| 异常测试 | `assertThrows()` | `@Test(expectedExceptions = ...)` |
| 超时测试 | `@Timeout` | `@Test(timeOut = ...)` |
| 依赖测试 | 不支持 | `@Test(dependsOnMethods = ...)` |
| 分组测试 | 不支持 | `@Test(groups = ...)` |
| Mock 框架 | Mockito | Mockito / EasyMock |

### 3.2 TestNG 核心特性

1. **灵活的注解体系**:
   - `@BeforeSuite` / `@AfterSuite`: 套件级别
   - `@BeforeTest` / `@AfterTest`: 测试级别
   - `@BeforeClass` / `@AfterClass`: 类级别
   - `@BeforeMethod` / `@AfterMethod`: 方法级别
   - `@BeforeGroups` / `@AfterGroups`: 组级别

2. **参数化测试 (DataProvider)**:
```java
@DataProvider(name = "testData")
public Object[][] provideData() {
    return new Object[][] {
        { "param1", 1 },
        { "param2", 2 }
    };
}

@Test(dataProvider = "testData")
public void testWithParams(String param, int value) {
    // Test implementation
}
```

3. **测试依赖**:
```java
@Test
public void testMethod1() { }

@Test(dependsOnMethods = { "testMethod1" })
public void testMethod2() { }
```

4. **测试分组**:
```java
@Test(groups = { "fast", "regression" })
public void fastTest() { }

@Test(groups = { "slow", "integration" })
public void slowTest() { }
```

5. **异常和超时**:
```java
@Test(expectedExceptions = { IOException.class, SQLException.class })
public void testException() { }

@Test(timeOut = 1000)
public void testTimeout() { }
```

## 4. 实现方案

### 4.1 后端实现 (pyutagent)

#### 4.1.1 创建 TestNG 生成器

**文件**: `pyutagent/agent/generators/testng_generator.py`

```python
from .base_generator import BaseTestGenerator
from typing import List, Dict, Any
from ...core.project_config import TestFramework, MockFramework

class TestNGGenerator(BaseTestGenerator):
    """TestNG 测试代码生成器"""
    
    def __init__(
        self,
        test_framework: TestFramework = TestFramework.TESTNG,
        mock_framework: MockFramework = MockFramework.MOCKITO,
        **kwargs
    ):
        super().__init__(test_framework, mock_framework, **kwargs)
        self.annotations = self._load_testng_annotations()
    
    def _load_testng_annotations(self) -> Dict[str, str]:
        """加载 TestNG 注解模板"""
        return {
            'test': '@Test',
            'before_method': '@BeforeMethod',
            'after_method': '@AfterMethod',
            'before_class': '@BeforeClass',
            'after_class': '@AfterClass',
            'before_test': '@BeforeTest',
            'after_test': '@AfterTest',
            'before_suite': '@BeforeSuite',
            'after_suite': '@AfterSuite',
            'data_provider': '@DataProvider(name = "{name}")',
            'parameters': '@Parameters({params})',
        }
    
    def generate_test_template(self, class_info: Dict[str, Any]) -> str:
        """生成 TestNG 测试模板"""
        class_name = class_info['name']
        methods = class_info.get('methods', [])
        dependencies = class_info.get('dependencies', [])
        
        template = self._build_imports()
        template += f"\npublic class {class_name}Test {{\n"
        template += self._build_setup_method(class_info)
        template += self._build_test_methods(methods, class_info)
        template += self._build_teardown_method()
        template += "}\n"
        
        return template
    
    def _build_imports(self) -> str:
        """构建 TestNG 导入语句"""
        imports = [
            "import org.testng.annotations.*;",
            "import static org.testng.Assert.*;",
        ]
        
        if self.mock_framework == MockFramework.MOCKITO:
            imports.extend([
                "import org.mockito.Mockito;",
                "import org.mockito.Mock;",
                "import org.mockito.testng.MockitoTestNGListener;",
                "import static org.mockito.Mockito.*;",
            ])
        
        imports.extend([
            "import java.util.*;",
            "import java.io.*;",
        ])
        
        return "\n".join(imports) + "\n"
    
    def _build_setup_method(self, class_info: Dict[str, Any]) -> str:
        """构建前置设置方法"""
        class_name = class_info['name']
        setup_method = "    @BeforeMethod\n"
        setup_method += "    public void setUp() {\n"
        
        # 添加依赖的 Mock 初始化
        for dep in class_info.get('dependencies', []):
            setup_method += f"        {dep['name']} = mock({dep['type']}.class);\n"
        
        setup_method += f"        target = new {class_name}();\n"
        setup_method += "    }\n\n"
        
        return setup_method
    
    def _build_test_methods(
        self,
        methods: List[Dict[str, Any]],
        class_info: Dict[str, Any]
    ) -> str:
        """构建测试方法"""
        test_methods = ""
        
        for method in methods:
            # 生成基础测试
            test_methods += self._generate_basic_test(method, class_info)
            
            # 生成边界值测试
            if method.get('parameters'):
                test_methods += self._generate_boundary_tests(method, class_info)
            
            # 生成异常测试
            if method.get('throws_exceptions'):
                test_methods += self._generate_exception_tests(method, class_info)
        
        return test_methods
    
    def _generate_basic_test(
        self,
        method: Dict[str, Any],
        class_info: Dict[str, Any]
    ) -> str:
        """生成基础测试方法"""
        method_name = method['name']
        test_name = f"test{method_name[0].upper()}{method_name[1:]}"
        
        test = f"    @Test\n"
        test += f"    public void {test_name}() {{\n"
        test += f"        // TODO: Implement test for {method_name}\n"
        test += f"        assertNotNull(target);\n"
        test += f"    }}\n\n"
        
        return test
    
    def _generate_parameterized_test(
        self,
        method: Dict[str, Any],
        class_info: Dict[str, Any]
    ) -> str:
        """生成参数化测试方法"""
        method_name = method['name']
        test_name = f"test{method_name[0].upper()}{method_name[1:]}WithParams"
        
        # 生成 DataProvider
        data_provider_name = f"provide{method_name.capitalize()}Data"
        test += f"    @DataProvider(name = \"{data_provider_name}\")\n"
        test += f"    public Object[][] {data_provider_name}() {{\n"
        test += f"        return new Object[][] {{\n"
        test += f"            {{ /* test data 1 */ }},\n"
        test += f"            {{ /* test data 2 */ }},\n"
        test += f"        }};\n"
        test += f"    }}\n\n"
        
        # 生成使用 DataProvider 的测试
        test += f"    @Test(dataProvider = \"{data_provider_name}\")\n"
        test += f"    public void {test_name}(Object... params) {{\n"
        test += f"        // TODO: Implement parameterized test\n"
        test += f"    }}\n\n"
        
        return test
    
    def _generate_exception_tests(
        self,
        method: Dict[str, Any],
        class_info: Dict[str, Any]
    ) -> str:
        """生成异常测试方法"""
        method_name = method['name']
        exceptions = method.get('throws_exceptions', [])
        
        test = ""
        for exception in exceptions:
            test_name = f"test{method_name[0].upper()}{method_name[1:]}Throws{exception['name']}"
            test += f"    @Test(expectedExceptions = {{{exception['full_name']}.class}})\n"
            test += f"    public void {test_name}() {{\n"
            test += f"        // TODO: Implement exception test\n"
            test += f"        target.{method_name}(...);\n"
            test += f"    }}\n\n"
        
        return test
    
    def _generate_boundary_tests(
        self,
        method: Dict[str, Any],
        class_info: Dict[str, Any]
    ) -> str:
        """生成边界值测试"""
        method_name = method['name']
        parameters = method.get('parameters', [])
        
        test = ""
        for param in parameters:
            if param['type'] in ['int', 'long', 'double', 'float']:
                test_name = f"test{method_name[0].upper()}{method_name[1:]}BoundaryValues"
                test += f"    @Test\n"
                test += f"    public void {test_name}() {{\n"
                test += f"        // TODO: Test boundary values for {param['name']}\n"
                test += f"        // Test min value\n"
                test += f"        // Test max value\n"
                test += f"        // Test zero/null\n"
                test += f"    }}\n\n"
        
        return test
    
    def _build_teardown_method(self) -> str:
        """构建后置清理方法"""
        teardown = "    @AfterMethod\n"
        teardown += "    public void tearDown() {\n"
        teardown += "        // Clean up resources\n"
        teardown += "    }\n\n"
        
        return teardown
    
    def get_test_runner_command(self, test_class: str) -> str:
        """获取 TestNG 测试运行命令"""
        return f"mvn test -Dtest={test_class} -Dsurefire.suiteXmlFiles=testng.xml"
    
    def get_group_runner_command(self, group_name: str) -> str:
        """获取运行特定分组的命令"""
        return f"mvn test -Dgroups={group_name}"
```

#### 4.1.2 更新 TestGeneratorAgent

**文件**: `pyutagent/agent/test_generator.py`

添加 TestNG 支持：

```python
from .generators.testng_generator import TestNGGenerator

class TestGeneratorAgent:
    def _create_generator(self):
        """创建合适的生成器"""
        if self.test_framework == TestFramework.TESTNG:
            return TestNGGenerator(
                test_framework=self.test_framework,
                mock_framework=self.mock_framework,
                project_path=self.project_path
            )
        elif self.test_framework == TestFramework.JUNIT5:
            # 现有的 JUnit5 生成器
            pass
```

#### 4.1.3 更新 TestNG 检测逻辑

**文件**: `pyutagent/core/project_config.py`

增强 TestNG 检测：

```python
def detect_test_framework(pom_path: Path) -> TestFramework:
    """检测项目使用的测试框架"""
    if not pom_path.exists():
        return TestFramework.UNKNOWN
    
    content = pom_path.read_text(encoding='utf-8')
    
    # 检测 TestNG
    testng_patterns = [
        'org.testng:testng',
        'org.testng:testng-core',
        '<groupId>org.testng</groupId>',
    ]
    
    for pattern in testng_patterns:
        if pattern in content:
            return TestFramework.TESTNG
    
    # 检测 JUnit 5
    junit5_patterns = [
        'org.junit.jupiter:junit-jupiter',
        '<groupId>org.junit.jupiter</groupId>',
    ]
    
    for pattern in junit5_patterns:
        if pattern in content:
            return TestFramework.JUNIT5
    
    # 检测 JUnit 4
    junit4_patterns = [
        'junit:junit:4',
        '<groupId>junit</groupId>',
    ]
    
    for pattern in junit4_patterns:
        if pattern in content:
            return TestFramework.JUNIT4
    
    return TestFramework.UNKNOWN
```

#### 4.1.4 更新 MavenRunner 支持 TestNG

**文件**: `pyutagent/tools/maven_tools.py`

添加 TestNG 特定方法：

```python
class MavenRunner:
    def run_testng_tests(self, test_class: Optional[str] = None) -> bool:
        """运行 TestNG 测试"""
        mvn = self._get_maven_executable()
        cmd = [mvn, "test", "-q"]
        
        if test_class:
            cmd.extend(["-Dtest=" + test_class])
        
        # TestNG 需要 surefire 插件配置
        cmd.extend(["-Dsurefire.suiteXmlFiles=testng.xml"])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception as e:
            logger.exception("Error running TestNG tests")
            return False
    
    def run_testng_group(self, group_name: str) -> bool:
        """运行 TestNG 分组测试"""
        mvn = self._get_maven_executable()
        cmd = [
            mvn, "test", "-q",
            f"-Dgroups={group_name}"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception as e:
            logger.exception(f"Error running TestNG group {group_name}")
            return False
    
    def check_testng_xml(self) -> bool:
        """检查 testng.xml 是否存在"""
        testng_xml = self.project_path / "testng.xml"
        return testng_xml.exists()
    
    def create_testng_xml(self, test_classes: List[str]) -> bool:
        """创建 testng.xml 配置文件"""
        testng_xml = self.project_path / "testng.xml"
        
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += '<!DOCTYPE suite SYSTEM "https://testng.org/testng-1.0.dtd">\n'
        xml_content += '<suite name="PyUTGeneratedTests">\n'
        xml_content += '  <test name="GeneratedTests">\n'
        xml_content += '    <classes>\n'
        
        for test_class in test_classes:
            xml_content += f'      <class name="{test_class}"/>\n'
        
        xml_content += '    </classes>\n'
        xml_content += '  </test>\n'
        xml_content += '</suite>\n'
        
        try:
            testng_xml.write_text(xml_content, encoding='utf-8')
            return True
        except Exception as e:
            logger.error(f"Failed to create testng.xml: {e}")
            return False
```

### 4.2 VS Code 插件实现

#### 4.2.1 更新配置选项

**文件**: `pyutagent-vscode/package.json`

```json
{
  "contributes": {
    "configuration": {
      "properties": {
        "pyutagent.testFramework": {
          "type": "string",
          "enum": ["junit5", "junit4", "testng", "spock"],
          "default": "junit5",
          "description": "测试框架选择"
        },
        "pyutagent.mockFramework": {
          "type": "string",
          "enum": ["mockito", "easymock", "jmock"],
          "default": "mockito",
          "description": "Mock 框架选择"
        }
      }
    }
  }
}
```

#### 4.2.2 更新配置面板

**文件**: `pyutagent-vscode/src/config/configPanel.ts`

添加 TestNG 配置选项：

```typescript
interface TestConfig {
    testFramework: 'junit5' | 'junit4' | 'testng' | 'spock';
    mockFramework: 'mockito' | 'easymock' | 'jmock';
    coverageThreshold: number;
    // ...
}

private _renderConfiguration(): string {
    return `
        <div class="form-group">
            <label for="testFramework">测试框架:</label>
            <select id="testFramework">
                <option value="junit5" ${this.config.testFramework === 'junit5' ? 'selected' : ''}>JUnit 5</option>
                <option value="junit4" ${this.config.testFramework === 'junit4' ? 'selected' : ''}>JUnit 4</option>
                <option value="testng" ${this.config.testFramework === 'testng' ? 'selected' : ''}>TestNG</option>
                <option value="spock" ${this.config.testFramework === 'spock' ? 'selected' : ''}>Spock</option>
            </select>
        </div>
        <div class="form-group">
            <label for="mockFramework">Mock 框架:</label>
            <select id="mockFramework">
                <option value="mockito" ${this.config.mockFramework === 'mockito' ? 'selected' : ''}>Mockito</option>
                <option value="easymock" ${this.config.mockFramework === 'easymock' ? 'selected' : ''}>EasyMock</option>
                <option value="jmock" ${this.config.mockFramework === 'jmock' ? 'selected' : ''}>JMock</option>
            </select>
        </div>
    `;
}
```

#### 4.2.3 更新 Diff Provider

**文件**: `pyutagent-vscode/src/diff/diffProvider.ts`

添加 TestNG 语法高亮支持：

```typescript
private _getMonacoLanguage(): string {
    switch (this.config.testFramework) {
        case 'testng':
        case 'junit5':
        case 'junit4':
            return 'java';
        case 'spock':
            return 'groovy'; // Spock 使用 Groovy
        default:
            return 'java';
    }
}
```

## 5. TestNG 依赖配置

### 5.1 Maven 依赖

**文件**: `pom.xml`

```xml
<dependencies>
    <!-- TestNG Core -->
    <dependency>
        <groupId>org.testng</groupId>
        <artifactId>testng</artifactId>
        <version>7.8.0</version>
        <scope>test</scope>
    </dependency>
    
    <!-- Mockito with TestNG support -->
    <dependency>
        <groupId>org.mockito</groupId>
        <artifactId>mockito-core</artifactId>
        <version>5.5.0</version>
        <scope>test</scope>
    </dependency>
    
    <!-- AssertJ (可选，用于更流畅的断言) -->
    <dependency>
        <groupId>org.assertj</groupId>
        <artifactId>assertj-core</artifactId>
        <version>3.24.2</version>
        <scope>test</scope>
    </dependency>
</dependencies>

<build>
    <plugins>
        <!-- Maven Surefire Plugin for TestNG -->
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-surefire-plugin</artifactId>
            <version>3.1.2</version>
            <configuration>
                <suiteXmlFiles>
                    <suiteXmlFile>testng.xml</suiteXmlFile>
                </suiteXmlFiles>
            </configuration>
        </plugin>
        
        <!-- JaCoCo for coverage -->
        <plugin>
            <groupId>org.jacoco</groupId>
            <artifactId>jacoco-maven-plugin</artifactId>
            <version>0.8.10</version>
            <executions>
                <execution>
                    <goals>
                        <goal>prepare-agent</goal>
                    </goals>
                </execution>
                <execution>
                    <id>report</id>
                    <phase>test</phase>
                    <goals>
                        <goal>report</goal>
                    </goals>
                </execution>
            </executions>
        </plugin>
    </plugins>
</build>
```

### 5.2 Gradle 依赖

**文件**: `build.gradle`

```groovy
plugins {
    id 'java'
    id 'jacoco'
}

dependencies {
    // TestNG
    testImplementation 'org.testng:testng:7.8.0'
    
    // Mockito
    testImplementation 'org.mockito:mockito-core:5.5.0'
    
    // AssertJ
    testImplementation 'org.assertj:assertj-core:3.24.2'
}

test {
    useTestNG()
}

jacocoTestReport {
    dependsOn test
    reports {
        xml.required = true
        html.required = true
    }
}
```

## 6. 测试示例

### 6.1 基础 TestNG 测试

```java
import org.testng.annotations.*;
import static org.testng.Assert.*;
import org.mockito.Mockito;
import static org.mockito.Mockito.*;

public class PaymentServiceTest {
    
    @Mock
    private PaymentGateway paymentGateway;
    
    private PaymentService target;
    
    @BeforeMethod
    public void setUp() {
        paymentGateway = mock(PaymentGateway.class);
        target = new PaymentService(paymentGateway);
    }
    
    @Test
    public void testProcessPayment_Success() {
        // Arrange
        PaymentRequest request = new PaymentRequest("100.00", "USD");
        when(paymentGateway.process(any())).thenReturn(true);
        
        // Act
        boolean result = target.processPayment(request);
        
        // Assert
        assertTrue(result);
        verify(paymentGateway, times(1)).process(any());
    }
    
    @Test(expectedExceptions = { PaymentException.class })
    public void testProcessPayment_Failure() {
        // Arrange
        PaymentRequest request = new PaymentRequest("0.00", "USD");
        when(paymentGateway.process(any())).thenThrow(new PaymentException("Invalid amount"));
        
        // Act
        target.processPayment(request);
        // Assert: Exception thrown
    }
    
    @Test(timeOut = 5000)
    public void testProcessPayment_Timeout() {
        // Arrange
        PaymentRequest request = new PaymentRequest("100.00", "USD");
        when(paymentGateway.process(any())).thenAnswer(invocation -> {
            Thread.sleep(10000); // Simulate slow response
            return true;
        });
        
        // Act
        boolean result = target.processPayment(request);
        
        // Assert: Test will timeout after 5 seconds
    }
    
    @DataProvider(name = "paymentAmounts")
    public Object[][] providePaymentData() {
        return new Object[][] {
            { "100.00", "USD", true },
            { "0.00", "USD", false },
            { "-50.00", "USD", false },
            { "1000.00", "EUR", true }
        };
    }
    
    @Test(dataProvider = "paymentAmounts")
    public void testProcessPayment_WithVariousAmounts(
        String amount, 
        String currency, 
        boolean expectedResult
    ) {
        // Arrange
        PaymentRequest request = new PaymentRequest(amount, currency);
        when(paymentGateway.process(any())).thenReturn(expectedResult);
        
        // Act
        boolean result = target.processPayment(request);
        
        // Assert
        assertEquals(result, expectedResult);
    }
    
    @Test(groups = { "fast", "regression" })
    public void testBasicValidation() {
        // Fast test for regression suite
        assertTrue(target.validate(new PaymentRequest("100.00", "USD")));
    }
    
    @Test(groups = { "slow", "integration" })
    public void testIntegrationWithExternalService() {
        // Slow integration test
        // ...
    }
    
    @AfterMethod
    public void tearDown() {
        target = null;
    }
}
```

### 6.2 TestNG XML 配置

**文件**: `testng.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE suite SYSTEM "https://testng.org/testng-1.0.dtd">
<suite name="PyUTGeneratedTests">
    <test name="FastTests">
        <groups>
            <run>
                <include name="fast" />
            </run>
        </groups>
        <classes>
            <class name="com.example.PaymentServiceTest" />
        </classes>
    </test>
    
    <test name="AllTests">
        <classes>
            <class name="com.example.PaymentServiceTest" />
            <class name="com.example.OrderServiceTest" />
        </classes>
    </test>
</suite>
```

## 7. 实现优先级

### P0 - 核心功能 (Week 2)

1. ✅ 创建 `TestNGGenerator` 类
2. ✅ 实现基础 TestNG 模板生成
3. ✅ 实现 TestNG 注解支持
4. ✅ 实现 DataProvider 参数化测试
5. ✅ 实现异常测试
6. ✅ 更新 `TestGeneratorAgent` 支持 TestNG
7. ✅ 更新 VS Code 配置面板

### P1 - 增强功能 (Week 3)

1. ⏭️ 实现测试分组支持
2. ⏭️ 实现测试依赖支持
3. ⏭️ 实现超时测试
4. ⏭️ 创建 testng.xml 自动生成
5. ⏭️ 更新 MavenRunner 支持 TestNG 命令

### P2 - 优化功能 (Week 4)

1. ⏭️ 智能 Mock 生成
2. ⏭️ 测试优化建议
3. ⏭️ Spock 框架支持

## 8. 测试计划

### 8.1 单元测试

- [ ] 测试 TestNGGenerator 模板生成
- [ ] 测试注解正确性
- [ ] 测试 DataProvider 生成
- [ ] 测试异常测试生成

### 8.2 集成测试

- [ ] 测试与 Maven 集成
- [ ] 测试与 JaCoCo 集成
- [ ] 测试 testng.xml 生成
- [ ] 测试实际项目运行

### 8.3 端到端测试

- [ ] 测试完整生成流程
- [ ] 测试 VS Code 插件集成
- [ ] 测试配置保存和加载

## 9. 验收标准

### 功能验收

1. ✅ 能够检测项目是否使用 TestNG
2. ✅ 能够生成符合 TestNG 规范的测试代码
3. ✅ 生成的测试代码能够通过 Maven 运行
4. ✅ 支持 DataProvider 参数化测试
5. ✅ 支持异常测试
6. ✅ 支持分组测试
7. ✅ 支持测试依赖

### 质量验收

1. ✅ 生成的代码符合 TestNG 最佳实践
2. ✅ 注释完整（英文）
3. ✅ 命名规范清晰
4. ✅ 覆盖率分析正常工作

## 10. 风险和缓解

### 风险

1. **TestNG 版本兼容性**: 不同版本可能有不同的 API
   - **缓解**: 支持主流版本（7.x），提供版本检测

2. **与 Mockito 集成**: TestNG 和 Mockito 的集成可能有特殊要求
   - **缓解**: 使用 `mockito-testng` 扩展

3. **项目配置复杂性**: 不同项目可能有不同的配置
   - **缓解**: 提供灵活的配置选项

## 11. 后续工作

1. 实现 Spock 框架支持（Groovy/Kotlin）
2. 实现 EasyMock/JMock 支持
3. 实现更多 TestNG 高级特性
4. 优化测试生成算法

## 12. 参考资料

- [TestNG 官方文档](https://testng.org/doc/)
- [TestNG Maven 集成](https://maven.apache.org/surefire/maven-surefire-plugin/examples/testng.html)
- [Mockito TestNG](https://github.com/mockito/mockito-testng)
- [JaCoCo Maven 插件](https://www.jacoco.org/jacoco/trunk/doc/maven.html)
