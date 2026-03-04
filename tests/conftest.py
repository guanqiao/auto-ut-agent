"""Global pytest fixtures for PyUT Agent tests.

This module provides common fixtures used across all test modules.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest


# =============================================================================
# Event Loop Fixture
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Provide a temporary file path."""
    return temp_dir / "test_file.txt"


# =============================================================================
# Mock LLM Client Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing.
    
    Returns:
        Mock: A mock LLM client with async methods
    """
    client = AsyncMock()
    client.agenerate = AsyncMock(return_value="generated test code")
    client.astream = AsyncMock(return_value=iter(["chunk1", "chunk2", "chunk3"]))
    client.test_connection = AsyncMock(return_value=(True, "Connection successful"))
    client.model = "gpt-4"
    return client


@pytest.fixture
def mock_llm_response():
    """Provide a mock LLM response."""
    return """
    @Test
    public void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.add(2, 3));
    }
    """


# =============================================================================
# Java Code Sample Fixtures
# =============================================================================

@pytest.fixture
def sample_java_class():
    """Return a simple Java class for testing."""
    return """
    package com.example;
    
    public class Calculator {
        public int add(int a, int b) {
            return a + b;
        }
        
        public int subtract(int a, int b) {
            return a - b;
        }
    }
    """


@pytest.fixture
def sample_java_class_with_dependencies():
    """Return a Java class with dependencies for testing."""
    return """
    package com.example;
    
    import java.util.List;
    import java.util.ArrayList;
    
    public class DataProcessor {
        private List<String> data;
        
        public DataProcessor() {
            this.data = new ArrayList<>();
        }
        
        public void addData(String item) {
            data.add(item);
        }
        
        public int getCount() {
            return data.size();
        }
    }
    """


@pytest.fixture
def sample_interface():
    """Return a Java interface for testing."""
    return """
    package com.example;
    
    public interface CalculatorService {
        int add(int a, int b);
        int subtract(int a, int b);
        int multiply(int a, int b);
        double divide(int a, int b);
    }
    """


# =============================================================================
# Maven Project Fixtures
# =============================================================================

@pytest.fixture
def maven_project_structure(temp_dir):
    """Create a minimal Maven project structure.
    
    Returns:
        Path: Path to the created Maven project
    """
    # Create directories
    src_main = temp_dir / "src" / "main" / "java" / "com" / "example"
    src_test = temp_dir / "src" / "test" / "java" / "com" / "example"
    src_main.mkdir(parents=True, exist_ok=True)
    src_test.mkdir(parents=True, exist_ok=True)
    
    # Create pom.xml
    pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
</project>
"""
    (temp_dir / "pom.xml").write_text(pom_content)
    
    return temp_dir


@pytest.fixture
def sample_java_file(maven_project_structure):
    """Create a sample Java file in the Maven project."""
    java_file = maven_project_structure / "src" / "main" / "java" / "com" / "example" / "Calculator.java"
    java_content = """package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
}
"""
    java_file.write_text(java_content)
    return java_file


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def mock_settings():
    """Provide mock settings for testing."""
    from pyutagent.core.config import Settings
    
    settings = Mock(spec=Settings)
    settings.coverage.target_coverage = 0.8
    settings.coverage.max_iterations = 3
    settings.coverage.timeout_per_file = 300
    settings.maven.maven_path = "mvn"
    settings.jdk.java_home = "/usr/lib/jvm/java-11"
    return settings


@pytest.fixture
def mock_llm_config():
    """Provide a mock LLM configuration."""
    from pyutagent.core.config import LLMConfig, LLMProvider
    
    return LLMConfig(
        name="test-config",
        provider=LLMProvider.OPENAI,
        endpoint="https://api.openai.com/v1",
        api_key="test-api-key",
        model="gpt-4",
        timeout=300
    )


# =============================================================================
# PyQt Fixtures (for GUI tests)
# =============================================================================

@pytest.fixture
def qtbot(qtbot):
    """Provide a QtBot instance for GUI testing.
    
    This fixture extends the default qtbot with additional utilities.
    """
    return qtbot


@pytest.fixture
def mock_main_window(qtbot):
    """Create a mock main window for GUI testing."""
    from PyQt6.QtWidgets import QMainWindow
    
    window = QMainWindow()
    window.setWindowTitle("Test Window")
    window.resize(800, 600)
    qtbot.addWidget(window)
    return window


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_test_generation_result():
    """Provide a sample test generation result."""
    return {
        "success": True,
        "test_file": "/path/to/TestCalculator.java",
        "coverage": 0.85,
        "iterations": 2,
        "message": "Test generation completed successfully"
    }


@pytest.fixture
def sample_error_result():
    """Provide a sample error result."""
    return {
        "success": False,
        "error": "Compilation failed",
        "message": "Failed to compile generated tests"
    }


# =============================================================================
# Patch Fixtures
# =============================================================================

@pytest.fixture
def patch_settings():
    """Patch settings for testing."""
    with patch("pyutagent.core.config.get_settings") as mock:
        yield mock


@pytest.fixture
def patch_llm_client():
    """Patch LLM client for testing."""
    with patch("pyutagent.llm.client.LLMClient") as mock:
        yield mock


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically clean up after each test."""
    yield
    # Cleanup code runs after each test
    # Add any global cleanup here


# =============================================================================
# Marker Fixtures
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow tests (>5s)")
    config.addinivalue_line("markers", "flaky: Flaky tests")
    config.addinivalue_line("markers", "gui: GUI tests")
