"""Tests for Tool Abstraction Layer."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pyutagent.tools.core import (
    ToolBase,
    ToolCategory,
    ToolParameter,
    ToolResult,
    ToolError,
    ToolContext,
    ToolRegistry,
    get_tool_registry,
    register_tool,
)
from pyutagent.tools.core.tool_base import tool, ToolMeta
from pyutagent.tools.builtin import (
    ReadFileTool,
    WriteFileTool,
    DeleteFileTool,
    ListFilesTool,
)


class TestToolParameter:
    """Tests for ToolParameter."""
    
    def test_creation(self):
        """Test parameter creation."""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="Test parameter",
            required=True,
        )
        
        assert param.name == "test_param"
        assert param.type == "string"
        assert param.required is True
    
    def test_to_json_schema(self):
        """Test JSON schema generation."""
        param = ToolParameter(
            name="enum_param",
            type="string",
            description="Enum parameter",
            enum=["a", "b", "c"],
        )
        
        schema = param.to_json_schema()
        
        assert schema["type"] == "string"
        assert schema["enum"] == ["a", "b", "c"]


class TestToolResult:
    """Tests for ToolResult."""
    
    def test_ok_factory(self):
        """Test ok factory method."""
        result = ToolResult.ok(
            output="test output",
            data={"key": "value"},
        )
        
        assert result.success is True
        assert result.output == "test output"
        assert result.data["key"] == "value"
    
    def test_fail_factory(self):
        """Test fail factory method."""
        result = ToolResult.fail(
            error="test error",
            code="TEST_ERROR",
        )
        
        assert result.failed is True
        assert result.error.message == "test error"
        assert result.error.code == "TEST_ERROR"
    
    def test_timeout_factory(self):
        """Test timeout factory method."""
        result = ToolResult.timeout(30.0)
        
        assert result.timed_out is True
        assert "30" in result.error.message
    
    def test_cancel_factory(self):
        """Test cancel factory method."""
        result = ToolResult.cancel("user cancelled")
        
        assert result.cancelled is True
        assert result.error.message == "user cancelled"
    
    def test_with_duration(self):
        """Test duration chaining."""
        result = ToolResult.ok().with_duration(100)
        
        assert result.duration_ms == 100
    
    def test_with_metadata(self):
        """Test metadata chaining."""
        result = ToolResult.ok().with_metadata(key="value")
        
        assert result.metadata["key"] == "value"
    
    def test_with_artifacts(self):
        """Test artifacts chaining."""
        result = ToolResult.ok().with_artifacts("/path/to/file")
        
        assert "/path/to/file" in result.artifacts
    
    def test_to_dict_and_from_dict(self):
        """Test serialization."""
        original = ToolResult.ok(
            output="test",
            data={"key": "value"},
        ).with_duration(50)
        
        data = original.to_dict()
        restored = ToolResult.from_dict(data)
        
        assert restored.output == original.output
        assert restored.data == original.data
        assert restored.duration_ms == original.duration_ms
    
    def test_bool_conversion(self):
        """Test boolean conversion."""
        assert bool(ToolResult.ok()) is True
        assert bool(ToolResult.fail("error")) is False


class TestToolContext:
    """Tests for ToolContext."""
    
    def test_creation(self):
        """Test context creation."""
        ctx = ToolContext(project_path="/test/project")
        
        assert ctx.project_path == Path("/test/project")
        assert ctx.cwd == ctx.project_path
    
    def test_resolve_path(self):
        """Test path resolution."""
        ctx = ToolContext(project_path="/test/project")
        
        abs_path = ctx.resolve_path("/absolute/path")
        assert abs_path == Path("/absolute/path")
        
        rel_path = ctx.resolve_path("relative/path")
        assert rel_path == Path("/test/project/relative/path")
    
    def test_relative_path(self):
        """Test relative path calculation."""
        ctx = ToolContext(project_path="/test/project")
        
        rel = ctx.relative_path("/test/project/src/file.java")
        assert rel == Path("src/file.java")
    
    def test_config(self):
        """Test configuration management."""
        ctx = ToolContext(project_path="/test")
        
        ctx.set_config("key", "value")
        assert ctx.get_config("key") == "value"
        assert ctx.get_config("missing", "default") == "default"
    
    def test_env(self):
        """Test environment management."""
        ctx = ToolContext(project_path="/test")
        
        ctx.set_env("VAR", "value")
        assert ctx.get_env("VAR") == "value"
    
    def test_track_modified_files(self):
        """Test file tracking."""
        ctx = ToolContext(project_path="/test")
        
        ctx.track_modified_file("/test/file1.java")
        ctx.track_modified_file("/test/file2.java")
        
        modified = ctx.get_modified_files()
        assert len(modified) == 2
    
    def test_create_child(self):
        """Test child context creation."""
        parent = ToolContext(
            project_path="/test",
            config={"parent_key": "parent_value"},
        )
        
        child = parent.create_child(
            config={"child_key": "child_value"},
        )
        
        assert child.get_config("parent_key") == "parent_value"
        assert child.get_config("child_key") == "child_value"


class TestToolBase:
    """Tests for ToolBase."""
    
    def test_metadata(self):
        """Test tool metadata."""
        meta = ReadFileTool().metadata
        
        assert meta.name == "read_file"
        assert meta.category == ToolCategory.FILE
        assert len(meta.parameters) == 2
    
    def test_get_schema(self):
        """Test schema generation."""
        schema = ReadFileTool().get_schema()
        
        assert schema["name"] == "read_file"
        assert "parameters" in schema
        assert "file_path" in schema["parameters"]["properties"]
    
    def test_validate_parameters_success(self):
        """Test parameter validation success."""
        tool = ReadFileTool()
        
        errors = tool.validate_parameters({"file_path": "/test/file.txt"})
        
        assert len(errors) == 0
    
    def test_validate_parameters_missing_required(self):
        """Test parameter validation with missing required."""
        tool = ReadFileTool()
        
        errors = tool.validate_parameters({})
        
        assert len(errors) > 0
        assert any("file_path" in e for e in errors)
    
    @pytest.mark.asyncio
    async def test_run_with_validation_error(self):
        """Test run with validation error."""
        tool = ReadFileTool()
        
        result = await tool.run({})
        
        assert result.failed is True
        assert "VALIDATION_ERROR" == result.error.code


class TestFileTools:
    """Tests for file tools."""
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory."""
        return tmp_path
    
    @pytest.fixture
    def context(self, temp_dir):
        """Create a tool context."""
        return ToolContext(project_path=temp_dir)
    
    @pytest.mark.asyncio
    async def test_read_file_success(self, temp_dir, context):
        """Test successful file read."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")
        
        tool = ReadFileTool(context=context)
        result = await tool.run({"file_path": str(test_file)})
        
        assert result.success is True
        assert result.output == "Hello, World!"
    
    @pytest.mark.asyncio
    async def test_read_file_not_found(self, context):
        """Test reading non-existent file."""
        tool = ReadFileTool(context=context)
        result = await tool.run({"file_path": "/nonexistent/file.txt"})
        
        assert result.failed is True
        assert "FILE_NOT_FOUND" == result.error.code
    
    @pytest.mark.asyncio
    async def test_write_file_success(self, temp_dir, context):
        """Test successful file write."""
        test_file = temp_dir / "output.txt"
        
        tool = WriteFileTool(context=context)
        result = await tool.run({
            "file_path": str(test_file),
            "content": "Test content",
        })
        
        assert result.success is True
        assert test_file.read_text() == "Test content"
    
    @pytest.mark.asyncio
    async def test_write_file_create_dirs(self, temp_dir, context):
        """Test writing file with directory creation."""
        test_file = temp_dir / "subdir" / "output.txt"
        
        tool = WriteFileTool(context=context)
        result = await tool.run({
            "file_path": str(test_file),
            "content": "Test",
            "create_dirs": True,
        })
        
        assert result.success is True
        assert test_file.exists()
    
    @pytest.mark.asyncio
    async def test_delete_file_success(self, temp_dir, context):
        """Test successful file delete."""
        test_file = temp_dir / "to_delete.txt"
        test_file.write_text("Delete me")
        
        tool = DeleteFileTool(context=context)
        result = await tool.run({"file_path": str(test_file)})
        
        assert result.success is True
        assert not test_file.exists()
    
    @pytest.mark.asyncio
    async def test_list_files_success(self, temp_dir, context):
        """Test successful directory listing."""
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file3.txt").write_text("content3")
        
        tool = ListFilesTool(context=context)
        result = await tool.run({
            "directory": str(temp_dir),
            "pattern": "*.txt",
        })
        
        assert result.success is True
        assert result.data["count"] == 2
    
    @pytest.mark.asyncio
    async def test_list_files_recursive(self, temp_dir, context):
        """Test recursive directory listing."""
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file2.txt").write_text("content2")
        
        tool = ListFilesTool(context=context)
        result = await tool.run({
            "directory": str(temp_dir),
            "pattern": "*.txt",
            "recursive": True,
        })
        
        assert result.success is True
        assert result.data["count"] == 2


class TestToolRegistry:
    """Tests for ToolRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return ToolRegistry()
    
    def test_register_and_get(self, registry):
        """Test tool registration and retrieval."""
        registry.register(ReadFileTool)
        
        assert registry.has("read_file")
        assert registry.get("read_file") == ReadFileTool
    
    def test_unregister(self, registry):
        """Test tool unregistration."""
        registry.register(ReadFileTool)
        
        result = registry.unregister("read_file")
        
        assert result is True
        assert not registry.has("read_file")
    
    def test_aliases(self, registry):
        """Test tool aliases."""
        registry.register(ReadFileTool, aliases=["read", "cat"])
        
        assert registry.has("read")
        assert registry.has("cat")
        assert registry.get("read") == ReadFileTool
    
    def test_list_tools(self, registry):
        """Test listing tools."""
        registry.register(ReadFileTool)
        registry.register(WriteFileTool)
        
        tools = registry.list_tools()
        
        assert "read_file" in tools
        assert "write_file" in tools
    
    def test_list_tools_by_category(self, registry):
        """Test listing tools by category."""
        registry.register(ReadFileTool)
        
        tools = registry.list_tools(category=ToolCategory.FILE)
        
        assert "read_file" in tools
    
    def test_get_schemas(self, registry):
        """Test getting schemas."""
        registry.register(ReadFileTool)
        
        schemas = registry.get_schemas()
        
        assert len(schemas) == 1
        assert schemas[0]["name"] == "read_file"
    
    @pytest.mark.asyncio
    async def test_execute(self, registry, tmp_path):
        """Test tool execution through registry."""
        registry.register(ReadFileTool)
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        context = ToolContext(project_path=tmp_path)
        result = await registry.execute(
            "read_file",
            {"file_path": str(test_file)},
            context=context,
        )
        
        assert result.success is True
        assert result.output == "content"
    
    def test_get_stats(self, registry):
        """Test getting statistics."""
        registry.register(ReadFileTool)
        registry.register(WriteFileTool)
        
        stats = registry.get_stats()
        
        assert stats["total_tools"] == 2
    
    def test_register_tool_decorator(self, registry):
        """Test @register_tool decorator."""
        @register_tool()
        class CustomTool(ToolBase):
            name = "custom_tool"
            description = "A custom tool"
            
            async def execute(self, params, context=None):
                return ToolResult.ok()
        
        assert get_tool_registry().has("custom_tool")


class TestToolDecorator:
    """Tests for @tool decorator."""
    
    @pytest.mark.asyncio
    async def test_tool_decorator(self):
        """Test @tool decorator creates a tool class."""
        @tool(
            name="echo",
            description="Echo a message",
            parameters=[
                ToolParameter("message", "string", "Message to echo")
            ],
        )
        async def echo_tool(params, context):
            return ToolResult.ok(output=params["message"])
        
        tool_instance = echo_tool()
        result = await tool_instance.run({"message": "hello"})
        
        assert result.success is True
        assert result.output == "hello"
