"""Unit tests for IncrementalTestManager."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile
import os

from pyutagent.agent.incremental_manager import (
    IncrementalConfig,
    ExistingTestAnalysis,
    IncrementalContext,
    IncrementalTestManager,
    create_incremental_manager,
)
from pyutagent.agent.partial_success_handler import TestMethodInfo


class TestIncrementalConfig:
    """Tests for IncrementalConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IncrementalConfig()
        
        assert config.enabled is False
        assert config.preserve_passing_tests is True
        assert config.analyze_existing_coverage is True
        assert config.max_preserved_tests == 50
        assert config.min_tests_to_preserve == 1
        assert config.force_regenerate_failed is True
        assert config.skip_analysis is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = IncrementalConfig(
            enabled=True,
            preserve_passing_tests=False,
            max_preserved_tests=100,
            skip_analysis=True
        )
        
        assert config.enabled is True
        assert config.preserve_passing_tests is False
        assert config.max_preserved_tests == 100
        assert config.skip_analysis is True


class TestExistingTestAnalysis:
    """Tests for ExistingTestAnalysis."""
    
    def test_default_analysis(self):
        """Test default analysis values."""
        analysis = ExistingTestAnalysis(test_file_path="path/to/Test.java")
        
        assert analysis.test_file_path == "path/to/Test.java"
        assert analysis.exists is False
        assert analysis.test_methods == []
        assert analysis.passing_tests == []
        assert analysis.failing_tests == []
        assert analysis.current_coverage == 0.0
    
    def test_total_tests(self):
        """Test total_tests property."""
        analysis = ExistingTestAnalysis(
            test_file_path="Test.java",
            test_methods=[
                TestMethodInfo(method_name="test1", content="void test1() {}", start_line=1, end_line=2),
                TestMethodInfo(method_name="test2", content="void test2() {}", start_line=3, end_line=4),
            ]
        )
        
        assert analysis.total_tests == 2
    
    def test_has_tests(self):
        """Test has_tests property."""
        analysis = ExistingTestAnalysis(test_file_path="Test.java")
        assert analysis.has_tests is False
        
        analysis.test_methods = [TestMethodInfo(method_name="test1", content="void test1() {}", start_line=1, end_line=2)]
        assert analysis.has_tests is True
    
    def test_has_passing_tests(self):
        """Test has_passing_tests property."""
        analysis = ExistingTestAnalysis(test_file_path="Test.java")
        assert analysis.has_passing_tests is False
        
        analysis.passing_tests = ["test1"]
        assert analysis.has_passing_tests is True
    
    def test_has_failing_tests(self):
        """Test has_failing_tests property."""
        analysis = ExistingTestAnalysis(test_file_path="Test.java")
        assert analysis.has_failing_tests is False
        
        analysis.failing_tests = ["test1"]
        assert analysis.has_failing_tests is True
        
        analysis.failing_tests = []
        analysis.error_tests = ["test2"]
        assert analysis.has_failing_tests is True
    
    def test_pass_rate(self):
        """Test pass_rate property."""
        analysis = ExistingTestAnalysis(
            test_file_path="Test.java",
            test_methods=[
                TestMethodInfo(method_name="test1", content="void test1() {}", start_line=1, end_line=2),
                TestMethodInfo(method_name="test2", content="void test2() {}", start_line=3, end_line=4),
                TestMethodInfo(method_name="test3", content="void test3() {}", start_line=5, end_line=6),
            ],
            passing_tests=["test1", "test2"]
        )
        
        assert analysis.pass_rate == pytest.approx(2/3)
        
        analysis_empty = ExistingTestAnalysis(test_file_path="Test.java")
        assert analysis_empty.pass_rate == 0.0
    
    def test_should_use_incremental(self):
        """Test should_use_incremental property."""
        analysis = ExistingTestAnalysis(
            test_file_path="Test.java",
            passing_tests=["test1"]
        )
        assert analysis.should_use_incremental is True
        
        analysis.has_compilation_errors = True
        assert analysis.should_use_incremental is False


class TestIncrementalContext:
    """Tests for IncrementalContext."""
    
    def test_default_context(self):
        """Test default context values."""
        context = IncrementalContext()
        
        assert context.existing_tests_code == ""
        assert context.preserved_test_names == []
        assert context.tests_to_fix == []
        assert context.target_coverage_gap == 0.0
    
    def test_has_preserved_tests(self):
        """Test has_preserved_tests property."""
        context = IncrementalContext()
        assert context.has_preserved_tests is False
        
        context.preserved_test_names = ["test1", "test2"]
        assert context.has_preserved_tests is True
    
    def test_has_tests_to_fix(self):
        """Test has_tests_to_fix property."""
        context = IncrementalContext()
        assert context.has_tests_to_fix is False
        
        from pyutagent.agent.partial_success_handler import TestMethodResult, TestStatus
        context.tests_to_fix = [TestMethodResult(method_name="test1", status=TestStatus.FAILED)]
        assert context.has_tests_to_fix is True
    
    def test_needs_additional_coverage(self):
        """Test needs_additional_coverage property."""
        context = IncrementalContext()
        assert context.needs_additional_coverage is False
        
        context.target_coverage_gap = 0.2
        assert context.needs_additional_coverage is True


class TestIncrementalTestManager:
    """Tests for IncrementalTestManager."""
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure."""
        src_main = tmp_path / "src" / "main" / "java" / "com" / "example"
        src_test = tmp_path / "src" / "test" / "java" / "com" / "example"
        
        src_main.mkdir(parents=True)
        src_test.mkdir(parents=True)
        
        java_file = src_main / "MyClass.java"
        java_file.write_text("""
package com.example;

public class MyClass {
    public void method1() {}
    public int method2() { return 42; }
}
""")
        
        return tmp_path
    
    @pytest.fixture
    def manager(self, temp_project):
        """Create an IncrementalTestManager instance."""
        config = IncrementalConfig(enabled=True)
        return IncrementalTestManager(
            project_path=str(temp_project),
            config=config
        )
    
    def test_init(self, temp_project):
        """Test manager initialization."""
        config = IncrementalConfig(enabled=True)
        manager = IncrementalTestManager(
            project_path=str(temp_project),
            config=config
        )
        
        assert manager.config.enabled is True
        assert manager.parser is not None
        assert manager.partial_handler is not None
    
    def test_detect_existing_test_not_found(self, manager, temp_project):
        """Test detecting non-existent test file."""
        target_file = str(temp_project / "src" / "main" / "java" / "com" / "example" / "MyClass.java")
        
        result = manager.detect_existing_test(target_file)
        
        assert result is None
    
    def test_detect_existing_test_found(self, manager, temp_project):
        """Test detecting existing test file."""
        target_file = str(temp_project / "src" / "main" / "java" / "com" / "example" / "MyClass.java")
        test_file = temp_project / "src" / "test" / "java" / "com" / "example" / "MyClassTest.java"
        test_file.write_text("""
package com.example;

class MyClassTest {
    @Test
    void testMethod1() {}
}
""")
        
        result = manager.detect_existing_test(
            target_file,
            {"package": "com.example"}
        )
        
        assert result is not None
        assert "MyClassTest.java" in result
    
    def test_analyze_existing_tests_not_exists(self, manager):
        """Test analyzing non-existent test file."""
        import asyncio
        
        result = asyncio.run(manager.analyze_existing_tests("nonexistent/Test.java"))
        
        assert result.exists is False
    
    def test_analyze_existing_tests_exists(self, manager, temp_project):
        """Test analyzing existing test file."""
        import asyncio
        
        test_file = temp_project / "src" / "test" / "java" / "com" / "example" / "MyClassTest.java"
        test_file.write_text("""
package com.example;

import org.junit.jupiter.api.Test;

class MyClassTest {
    @Test
    void testMethod1() {}
    
    @Test
    void testMethod2() {}
}
""")
        
        result = asyncio.run(manager.analyze_existing_tests(
            str(test_file.relative_to(temp_project)),
            run_tests=False
        ))
        
        assert result.exists is True
        assert len(result.test_methods) == 2
        assert "testMethod1" in [m.method_name for m in result.test_methods]
    
    def test_build_incremental_context(self, manager):
        """Test building incremental context."""
        analysis = ExistingTestAnalysis(
            test_file_path="Test.java",
            exists=True,
            test_methods=[
                TestMethodInfo(method_name="testMethod1", content="@Test void testMethod1() {}", start_line=1, end_line=2),
                TestMethodInfo(method_name="testMethod2", content="@Test void testMethod2() {}", start_line=3, end_line=4),
            ],
            passing_tests=["testMethod1"],
            failing_tests=["testMethod2"],
            current_coverage=0.5,
            test_code="class Test { @Test void testMethod1() {} @Test void testMethod2() {} }"
        )
        
        class_info = {"name": "MyClass", "package": "com.example"}
        source_code = "public class MyClass { public void method1() {} }"
        
        context = manager.build_incremental_context(
            analysis=analysis,
            class_info=class_info,
            source_code=source_code,
            target_coverage=0.8
        )
        
        assert context.existing_tests_code == analysis.test_code
        assert "testMethod1" in context.preserved_test_names
        assert context.target_coverage_gap == pytest.approx(0.3)
    
    def test_should_use_incremental_mode_disabled(self, manager):
        """Test should_use_incremental_mode when disabled."""
        manager.config.enabled = False
        analysis = ExistingTestAnalysis(
            test_file_path="Test.java",
            exists=True,
            passing_tests=["test1"]
        )
        
        result = manager.should_use_incremental_mode(analysis)
        
        assert result is False
    
    def test_should_use_incremental_mode_not_exists(self, manager):
        """Test should_use_incremental_mode when test doesn't exist."""
        analysis = ExistingTestAnalysis(test_file_path="Test.java", exists=False)
        
        result = manager.should_use_incremental_mode(analysis)
        
        assert result is False
    
    def test_should_use_incremental_mode_compilation_errors(self, manager):
        """Test should_use_incremental_mode with compilation errors."""
        analysis = ExistingTestAnalysis(
            test_file_path="Test.java",
            exists=True,
            passing_tests=["test1"],
            has_compilation_errors=True
        )
        
        result = manager.should_use_incremental_mode(analysis)
        
        assert result is False
    
    def test_should_use_incremental_mode_success(self, manager):
        """Test should_use_incremental_mode when conditions are met."""
        analysis = ExistingTestAnalysis(
            test_file_path="Test.java",
            exists=True,
            test_methods=[TestMethodInfo(method_name="test1", content="void test1() {}", start_line=1, end_line=2)],
            passing_tests=["test1"]
        )
        
        result = manager.should_use_incremental_mode(analysis)
        
        assert result is True
    
    def test_merge_tests(self, manager):
        """Test merging preserved and new tests."""
        preserved_code = """
class MyClassTest {
    @Test
    void testExisting() {}
}
"""
        new_code = """
class MyClassTest {
    @Test
    void testNew() {}
}
"""
        
        result = manager.merge_tests(preserved_code, new_code, "MyClass")
        
        assert "testNew" in result
        assert result is not None


class TestCreateIncrementalManager:
    """Tests for create_incremental_manager factory function."""
    
    def test_create_with_defaults(self, tmp_path):
        """Test creating manager with default settings."""
        manager = create_incremental_manager(
            project_path=str(tmp_path),
            incremental_mode=True
        )
        
        assert manager.config.enabled is True
        assert manager.config.preserve_passing_tests is True
    
    def test_create_with_custom_settings(self, tmp_path):
        """Test creating manager with custom settings."""
        manager = create_incremental_manager(
            project_path=str(tmp_path),
            incremental_mode=True,
            preserve_passing_tests=False,
            max_preserved_tests=100,
            skip_analysis=True
        )
        
        assert manager.config.enabled is True
        assert manager.config.preserve_passing_tests is False
        assert manager.config.max_preserved_tests == 100
        assert manager.config.skip_analysis is True
    
    def test_create_disabled(self, tmp_path):
        """Test creating manager with incremental mode disabled."""
        manager = create_incremental_manager(
            project_path=str(tmp_path),
            incremental_mode=False
        )
        
        assert manager.config.enabled is False
