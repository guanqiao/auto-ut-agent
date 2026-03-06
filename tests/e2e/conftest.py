"""Test fixtures for E2E tests.

This module provides comprehensive test fixtures for E2E testing:
- Temporary Maven projects
- Mock LLM clients and servers
- Configuration management
- GUI components
"""

import asyncio
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication

from pyutagent.llm.config import LLMConfig, LLMConfigCollection
from pyutagent.core.config import Settings


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_maven_project() -> Generator[Path, None, None]:
    """Create a temporary Maven project with basic structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        
        src_main = project_path / "src" / "main" / "java" / "com" / "example"
        src_main.mkdir(parents=True, exist_ok=True)
        
        src_test = project_path / "src" / "test" / "java" / "com" / "example"
        src_test.mkdir(parents=True, exist_ok=True)
        
        pom_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0-SNAPSHOT</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.1.2</version>
            </plugin>
        </plugins>
    </build>
</project>
'''
        (project_path / "pom.xml").write_text(pom_xml)
        
        calculator_code = '''
package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
    
    public int divide(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("Division by zero");
        }
        return a / b;
    }
    
    public boolean isPositive(int n) {
        return n > 0;
    }
    
    public boolean isNegative(int n) {
        return n < 0;
    }
    
    public int abs(int n) {
        return n < 0 ? -n : n;
    }
}
'''
        (src_main / "Calculator.java").write_text(calculator_code)
        
        yield project_path


@pytest.fixture
def temp_multi_file_project() -> Generator[Path, None, None]:
    """Create a temporary Maven project with multiple interdependent files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        
        src_main = project_path / "src" / "main" / "java" / "com" / "example"
        src_main.mkdir(parents=True, exist_ok=True)
        
        src_test = project_path / "src" / "test" / "java" / "com" / "example"
        src_test.mkdir(parents=True, exist_ok=True)
        
        pom_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>multi-file-project</artifactId>
    <version>1.0-SNAPSHOT</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
'''
        (project_path / "pom.xml").write_text(pom_xml)
        
        model_code = '''
package com.example.model;

public class User {
    private String name;
    private int age;
    
    public User(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public String getName() { return name; }
    public int getAge() { return age; }
    
    public boolean isAdult() {
        return age >= 18;
    }
}
'''
        model_dir = src_main / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "User.java").write_text(model_code)
        
        repository_code = '''
package com.example.repository;

import com.example.model.User;
import java.util.*;

public class UserRepository {
    private Map<String, User> users = new HashMap<>();
    
    public void save(User user) {
        users.put(user.getName(), user);
    }
    
    public User findByName(String name) {
        return users.get(name);
    }
    
    public List<User> findAll() {
        return new ArrayList<>(users.values());
    }
    
    public void delete(String name) {
        users.remove(name);
    }
}
'''
        repo_dir = src_main / "repository"
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "UserRepository.java").write_text(repository_code)
        
        service_code = '''
package com.example.service;

import com.example.model.User;
import com.example.repository.UserRepository;
import java.util.List;

public class UserService {
    private UserRepository repository;
    
    public UserService(UserRepository repository) {
        this.repository = repository;
    }
    
    public void createUser(String name, int age) {
        User user = new User(name, age);
        repository.save(user);
    }
    
    public User getUser(String name) {
        return repository.findByName(name);
    }
    
    public List<User> getAllUsers() {
        return repository.findAll();
    }
    
    public void deleteUser(String name) {
        repository.delete(name);
    }
    
    public boolean isUserAdult(String name) {
        User user = repository.findByName(name);
        return user != null && user.isAdult();
    }
}
'''
        service_dir = src_main / "service"
        service_dir.mkdir(parents=True, exist_ok=True)
        (service_dir / "UserService.java").write_text(service_code)
        
        calculator_code = '''
package com.example.util;

public class Calculator {
    public int add(int a, int b) { return a + b; }
    public int subtract(int a, int b) { return a - b; }
    public int multiply(int a, int b) { return a * b; }
    public int divide(int a, int b) {
        if (b == 0) throw new IllegalArgumentException("Division by zero");
        return a / b;
    }
}
'''
        util_dir = src_main / "util"
        util_dir.mkdir(parents=True, exist_ok=True)
        (util_dir / "Calculator.java").write_text(calculator_code)
        
        string_utils_code = '''
package com.example.util;

public class StringUtils {
    public static boolean isEmpty(String str) {
        return str == null || str.isEmpty();
    }
    
    public static String reverse(String str) {
        if (str == null) return null;
        return new StringBuilder(str).reverse().toString();
    }
    
    public static String capitalize(String str) {
        if (isEmpty(str)) return str;
        return str.substring(0, 1).toUpperCase() + str.substring(1);
    }
}
'''
        (util_dir / "StringUtils.java").write_text(string_utils_code)
        
        yield project_path


@pytest.fixture
def temp_large_project() -> Generator[Path, None, None]:
    """Create a large Maven project with many files for performance testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        
        src_main = project_path / "src" / "main" / "java" / "com" / "example"
        src_main.mkdir(parents=True, exist_ok=True)
        
        src_test = project_path / "src" / "test" / "java" / "com" / "example"
        src_test.mkdir(parents=True, exist_ok=True)
        
        pom_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>large-project</artifactId>
    <version>1.0-SNAPSHOT</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
'''
        (project_path / "pom.xml").write_text(pom_xml)
        
        for i in range(20):
            class_code = f'''
package com.example;

public class Service{i} {{
    public int process(int value) {{
        return value * {i + 1};
    }}
    
    public String getName() {{
        return "Service{i}";
    }}
}}
'''
            (src_main / f"Service{i}.java").write_text(class_code)
        
        yield project_path


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client with realistic responses."""
    client = AsyncMock()
    
    test_code_template = '''
package com.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

class {class_name}Test {{
    private {class_name} instance;
    
    @BeforeEach
    void setUp() {{
        instance = new {class_name}();
    }}
    
    @Test
    void testProcess() {{
        int result = instance.process(5);
        assertTrue(result > 0);
    }}
    
    @Test
    void testGetName() {{
        String name = instance.getName();
        assertNotNull(name);
        assertFalse(name.isEmpty());
    }}
}}
'''
    
    def generate_test_code(prompt: str, **kwargs) -> str:
        if "Calculator" in prompt:
            return '''
package com.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    private Calculator calculator;
    
    @BeforeEach
    void setUp() {
        calculator = new Calculator();
    }
    
    @Test
    void testAdd() {
        assertEquals(5, calculator.add(2, 3));
        assertEquals(0, calculator.add(-1, 1));
        assertEquals(-5, calculator.add(-2, -3));
    }
    
    @Test
    void testSubtract() {
        assertEquals(1, calculator.subtract(3, 2));
        assertEquals(-2, calculator.subtract(-1, 1));
    }
    
    @Test
    void testMultiply() {
        assertEquals(6, calculator.multiply(2, 3));
        assertEquals(-6, calculator.multiply(-2, 3));
    }
    
    @Test
    void testDivide() {
        assertEquals(2, calculator.divide(6, 3));
        assertEquals(0, calculator.divide(1, 2));
    }
    
    @Test
    void testDivideByZero() {
        assertThrows(IllegalArgumentException.class, () -> {
            calculator.divide(1, 0);
        });
    }
    
    @Test
    void testIsPositive() {
        assertTrue(calculator.isPositive(1));
        assertFalse(calculator.isPositive(0));
        assertFalse(calculator.isPositive(-1));
    }
    
    @Test
    void testIsNegative() {
        assertTrue(calculator.isNegative(-1));
        assertFalse(calculator.isNegative(0));
        assertFalse(calculator.isNegative(1));
    }
    
    @Test
    void testAbs() {
        assertEquals(5, calculator.abs(5));
        assertEquals(5, calculator.abs(-5));
        assertEquals(0, calculator.abs(0));
    }
}
'''
        elif "Service" in prompt:
            import re
            match = re.search(r'Service(\d+)', prompt)
            if match:
                num = match.group(1)
                return test_code_template.format(class_name=f"Service{num}")
        return test_code_template.format(class_name="Calculator")
    
    client.agenerate = AsyncMock(side_effect=lambda prompt, **kwargs: generate_test_code(prompt, **kwargs))
    client.astream = AsyncMock(return_value=iter([generate_test_code("Calculator")]))
    
    return client


@pytest.fixture
def mock_llm_server():
    """Create a mock LLM server for testing."""
    from unittest.mock import patch
    import json
    
    class MockLLMServer:
        def __init__(self):
            self.responses = []
            self.requests = []
        
        def add_response(self, response: str):
            self.responses.append(response)
        
        def get_next_response(self):
            if self.responses:
                return self.responses.pop(0)
            return "Default test response"
        
        def record_request(self, request: Dict):
            self.requests.append(request)
    
    server = MockLLMServer()
    yield server


@pytest.fixture
def temp_config(temp_dir: Path) -> Path:
    """Create a temporary configuration directory."""
    config_dir = temp_dir / ".pyutagent"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    llm_config = LLMConfigCollection(
        configs=[
            LLMConfig(
                id="test-openai",
                name="Test OpenAI",
                provider="openai",
                api_key="test-api-key",
                model="gpt-4",
                temperature=0.7,
                max_tokens=4096,
                timeout=300,
                max_retries=3,
                is_default=True
            )
        ]
    )
    
    config_file = config_dir / "llm_config.json"
    with open(config_file, 'w') as f:
        json.dump(llm_config.model_dump(), f, indent=2)
    
    app_config = {
        "log_level": "INFO",
        "target_coverage": 0.8,
        "max_iterations": 10,
        "project_paths": {
            "src_main_java": "src/main/java",
            "src_test_java": "src/test/java"
        }
    }
    
    app_config_file = config_dir / "config.json"
    with open(app_config_file, 'w') as f:
        json.dump(app_config, f, indent=2)
    
    return config_dir


@pytest.fixture
def temp_old_config(temp_dir: Path) -> Path:
    """Create an old version configuration for migration testing."""
    config_dir = temp_dir / ".pyutagent_old"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    old_config = {
        "llm": {
            "provider": "openai",
            "api_key": "old-api-key",
            "model": "gpt-3.5-turbo"
        },
        "settings": {
            "coverage_target": 70,
            "max_attempts": 5
        }
    }
    
    config_file = config_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(old_config, f, indent=2)
    
    return config_dir


@pytest.fixture
def main_window(qtbot):
    """Create a main window instance for GUI testing."""
    from pyutagent.ui.main_window import MainWindow
    
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    window = MainWindow()
    qtbot.addWidget(window)
    
    yield window
    
    window.close()


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
