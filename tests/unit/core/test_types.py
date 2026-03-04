"""测试类型系统"""
import pytest
from typing import get_type_hints, List, Optional
from pyutagent.core.types import (
    FilePath,
    ClassName,
    MethodName,
    CoveragePercentage,
    CompilationResultDict,
    TestResultDict,
    ComponentInfo,
    AgentResultDict
)


class TestNewTypes:
    """测试 NewType"""
    
    def test_file_path(self):
        """测试 FilePath"""
        file_path: FilePath = FilePath("/path/to/file.java")
        assert isinstance(file_path, str)
        assert file_path == "/path/to/file.java"
    
    def test_class_name(self):
        """测试 ClassName"""
        class_name: ClassName = ClassName("MyClass")
        assert isinstance(class_name, str)
        assert class_name == "MyClass"
    
    def test_method_name(self):
        """测试 MethodName"""
        method_name: MethodName = MethodName("testMethod")
        assert isinstance(method_name, str)
        assert method_name == "testMethod"
    
    def test_coverage_percentage(self):
        """测试 CoveragePercentage"""
        coverage: CoveragePercentage = CoveragePercentage(0.85)
        assert isinstance(coverage, float)
        assert coverage == 0.85
    
    def test_coverage_percentage_invalid(self):
        """测试 CoveragePercentage 无效值"""
        # 测试范围
        with pytest.raises(ValueError):
            CoveragePercentage(1.5)  # 超过 1.0
        
        with pytest.raises(ValueError):
            CoveragePercentage(-0.1)  # 小于 0


class TestTypedDicts:
    """测试 TypedDict"""
    
    def test_compilation_result_dict(self):
        """测试 CompilationResultDict"""
        result: CompilationResultDict = {
            "success": True,
            "output": "Build successful",
            "errors": [],
            "duration": 1.5
        }
        
        # 验证类型
        hints = get_type_hints(CompilationResultDict)
        assert hints["success"] == bool
        assert hints["output"] == str
        assert hints["errors"] == List[str]
        assert hints["duration"] == float
    
    def test_compilation_result_with_optional(self):
        """测试带可选字段的 CompilationResultDict"""
        result: CompilationResultDict = {
            "success": True,
            "output": "Build successful",
            "errors": ["Warning: deprecated API"],
            "duration": 1.5,
            "is_incremental": True  # 可选字段
        }
        
        assert result["is_incremental"] is True
    
    def test_test_result_dict(self):
        """测试 TestResultDict"""
        result: TestResultDict = {
            "total": 10,
            "passed": 8,
            "failed": 2,
            "errors": 0,
            "duration": 5.3
        }
        
        # 验证类型
        hints = get_type_hints(TestResultDict)
        assert hints["total"] == int
        assert hints["passed"] == int
        assert hints["failed"] == int
        assert hints["errors"] == int
        assert hints["duration"] == float
    
    def test_component_info(self):
        """测试 ComponentInfo"""
        info: ComponentInfo = {
            "name": "TestComponent",
            "type": "generator",
            "version": "1.0.0"
        }
        
        hints = get_type_hints(ComponentInfo)
        assert hints["name"] == str
        assert hints["type"] == str
        assert hints["version"] == str
    
    def test_component_info_with_optional(self):
        """测试带可选字段的 ComponentInfo"""
        info: ComponentInfo = {
            "name": "TestComponent",
            "type": "generator",
            "version": "1.0.0",
            "description": "A test component",  # 可选字段
            "enabled": True  # 可选字段
        }
        
        assert info["description"] == "A test component"
        assert info["enabled"] is True
    
    def test_agent_result_dict(self):
        """测试 AgentResultDict"""
        result: AgentResultDict = {
            "success": True,
            "coverage": 0.85,
            "iterations": 5,
            "test_file": "/path/to/TestFile.java",
            "errors": []
        }
        
        hints = get_type_hints(AgentResultDict)
        assert hints["success"] == bool
        assert hints["coverage"] == float
        assert hints["iterations"] == int
        assert hints["test_file"] == str
        assert hints["errors"] == List[str]


class TestTypeValidation:
    """测试类型验证"""
    
    def test_coverage_range(self):
        """测试覆盖率范围验证"""
        # 有效值
        c1 = CoveragePercentage(0.0)
        assert c1 == 0.0
        
        c2 = CoveragePercentage(1.0)
        assert c2 == 1.0
        
        c3 = CoveragePercentage(0.5)
        assert c3 == 0.5
    
    def test_file_path_format(self):
        """测试文件格式"""
        # Windows 路径
        p1 = FilePath("C:\\path\\to\\file.java")
        assert "file.java" in p1
        
        # Unix 路径
        p2 = FilePath("/path/to/file.java")
        assert p2.endswith("file.java")
        
        # 相对路径
        p3 = FilePath("src/main/java/File.java")
        assert "File.java" in p3
