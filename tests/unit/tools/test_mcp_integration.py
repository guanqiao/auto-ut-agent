"""Tests for MCP (Model Context Protocol) integration.

This module tests the comprehensive MCP integration including:
- Protocol types and message handling
- Tool definitions and adaptation
- Server configuration and management
- Client connections (stdio, HTTP, WebSocket)
- Server discovery
- Built-in presets
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import asdict

from pyutagent.tools.mcp_integration import (
    # Protocol Types
    MCPTransportType,
    MCPErrorCode,
    MCPError,
    MCPRequest,
    MCPResponse,
    # Tool Definitions
    MCPToolParameter,
    MCPTool,
    MCPToolResult,
    MCPToolError,
    # Server Configuration
    MCPServerConfig,
    ServerStatus,
    # Tool Adapter
    MCPToolAdapter,
    AdaptedTool,
    # Manager
    MCPManager,
    # Discovery
    MCPServerDiscovery,
    DiscoveredServer,
    # Presets
    MCPServerPresets,
    # Convenience Functions
    create_mcp_manager_with_presets,
)


# ============================================================================
# Protocol Types Tests
# ============================================================================

class TestMCPProtocolTypes:
    """Test MCP protocol type definitions."""

    def test_mcp_transport_type_values(self):
        """Test MCP transport type enum values."""
        assert MCPTransportType.STDIO.value == "stdio"
        assert MCPTransportType.HTTP.value == "http"
        assert MCPTransportType.HTTPS.value == "https"
        assert MCPTransportType.WEBSOCKET.value == "websocket"
        assert MCPTransportType.WEBSOCKET_SECURE.value == "wss"
        assert MCPTransportType.SSE.value == "sse"

    def test_mcp_error_code_values(self):
        """Test MCP error code enum values."""
        assert MCPErrorCode.PARSE_ERROR.value == -32700
        assert MCPErrorCode.INVALID_REQUEST.value == -32600
        assert MCPErrorCode.METHOD_NOT_FOUND.value == -32601
        assert MCPErrorCode.TOOL_NOT_FOUND.value == -32001

    def test_mcp_error_creation(self):
        """Test MCP error creation."""
        error = MCPError(
            code=-32001,
            message="Tool not found",
            data={"tool": "test_tool"}
        )
        assert error.code == -32001
        assert error.message == "Tool not found"
        assert error.data == {"tool": "test_tool"}

    def test_mcp_error_to_dict(self):
        """Test MCP error serialization."""
        error = MCPError(code=-32001, message="Error")
        result = error.to_dict()
        assert result == {"code": -32001, "message": "Error"}

    def test_mcp_request_creation(self):
        """Test MCP request creation."""
        request = MCPRequest(
            id=1,
            method="tools/list",
            params={"filter": "test"}
        )
        assert request.id == 1
        assert request.method == "tools/list"
        assert request.params == {"filter": "test"}
        assert request.jsonrpc == "2.0"

    def test_mcp_request_to_dict(self):
        """Test MCP request serialization."""
        request = MCPRequest(id=1, method="tools/list")
        result = request.to_dict()
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["method"] == "tools/list"

    def test_mcp_request_to_json(self):
        """Test MCP request JSON serialization."""
        request = MCPRequest(id=1, method="tools/list")
        json_str = request.to_json()
        data = json.loads(json_str)
        assert data["method"] == "tools/list"

    def test_mcp_response_from_dict_success(self):
        """Test MCP response parsing for success."""
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": []}
        }
        response = MCPResponse.from_dict(data)
        assert response.id == 1
        assert response.result == {"tools": []}
        assert response.error is None
        assert response.is_success()

    def test_mcp_response_from_dict_error(self):
        """Test MCP response parsing for error."""
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32001,
                "message": "Tool not found"
            }
        }
        response = MCPResponse.from_dict(data)
        assert response.id == 1
        assert response.result is None
        assert response.error is not None
        assert response.error.code == -32001
        assert not response.is_success()


# ============================================================================
# Tool Definition Tests
# ============================================================================

class TestMCPToolDefinitions:
    """Test MCP tool definitions."""

    def test_mcp_tool_parameter_creation(self):
        """Test MCP tool parameter creation."""
        param = MCPToolParameter(
            name="path",
            param_type="string",
            description="File path",
            required=True,
            pattern=r"^[\w\-/\\\.]+$"
        )
        assert param.name == "path"
        assert param.param_type == "string"
        assert param.required is True
        assert param.pattern == r"^[\w\-/\\\.]+$"

    def test_mcp_tool_parameter_to_dict(self):
        """Test MCP tool parameter serialization."""
        param = MCPToolParameter(
            name="count",
            param_type="integer",
            description="Item count",
            minimum=0,
            maximum=100
        )
        result = param.to_dict()
        assert result["type"] == "integer"
        assert result["minimum"] == 0
        assert result["maximum"] == 100

    def test_mcp_tool_creation(self):
        """Test MCP tool creation."""
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            parameters=[
                MCPToolParameter(
                    name="path",
                    param_type="string",
                    description="File path",
                    required=True
                )
            ]
        )
        assert tool.name == "read_file"
        assert len(tool.parameters) == 1

    def test_mcp_tool_to_dict(self):
        """Test MCP tool serialization."""
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            parameters=[
                MCPToolParameter(
                    name="path",
                    param_type="string",
                    description="File path",
                    required=True
                )
            ]
        )
        result = tool.to_dict()
        assert result["name"] == "read_file"
        assert "inputSchema" in result
        assert result["inputSchema"]["required"] == ["path"]

    def test_mcp_tool_from_dict(self):
        """Test MCP tool deserialization."""
        data = {
            "name": "write_file",
            "description": "Write a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "content": {
                        "type": "string",
                        "description": "File content"
                    }
                },
                "required": ["path", "content"]
            }
        }
        tool = MCPTool.from_dict(data)
        assert tool.name == "write_file"
        assert len(tool.parameters) == 2
        assert all(p.required for p in tool.parameters)

    def test_mcp_tool_result_success(self):
        """Test successful MCP tool result."""
        result = MCPToolResult.success_result(
            result="File content",
            execution_time_ms=100,
            metadata={"size": 1024}
        )
        assert result.success is True
        assert result.result == "File content"
        assert result.execution_time_ms == 100

    def test_mcp_tool_result_error(self):
        """Test error MCP tool result."""
        result = MCPToolResult.error_result(
            message="File not found",
            code=-32001,
            execution_time_ms=50
        )
        assert result.success is False
        assert result.error_message == "File not found"
        assert result.error_code == -32001


# ============================================================================
# Server Configuration Tests
# ============================================================================

class TestMCPServerConfig:
    """Test MCP server configuration."""

    def test_server_config_creation(self):
        """Test server config creation."""
        config = MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."],
            timeout=60,
            tags=["filesystem", "official"]
        )
        assert config.name == "filesystem"
        assert config.command == "npx"
        assert config.timeout == 60

    def test_server_config_to_dict(self):
        """Test server config serialization."""
        config = MCPServerConfig(
            name="test",
            command="cmd",
            args=["arg1"],
            transport=MCPTransportType.STDIO
        )
        result = config.to_dict()
        assert result["name"] == "test"
        assert result["transport"] == "stdio"
        assert result["args"] == ["arg1"]

    def test_server_config_from_dict(self):
        """Test server config deserialization."""
        data = {
            "name": "http-server",
            "command": "node",
            "args": ["server.js"],
            "transport": "http",
            "timeout": 45,
            "enabled": False
        }
        config = MCPServerConfig.from_dict(data)
        assert config.name == "http-server"
        assert config.transport == MCPTransportType.HTTP
        assert config.enabled is False


# ============================================================================
# Tool Adapter Tests
# ============================================================================

class TestMCPToolAdapter:
    """Test MCP tool adapter."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock MCP client."""
        client = Mock()
        client.config = Mock()
        client.config.name = "test_server"
        client.config.tags = ["test"]
        return client

    def test_adapt_tool(self, mock_client):
        """Test tool adaptation."""
        adapter = MCPToolAdapter(mock_client)

        mcp_tool = MCPTool(
            name="read_file",
            description="Read a file",
            parameters=[
                MCPToolParameter(
                    name="path",
                    param_type="string",
                    description="File path",
                    required=True
                )
            ]
        )

        adapted = adapter.adapt_tool(mcp_tool)

        assert adapted.name == "mcp_test_server_read_file"
        assert adapted.original_name == "read_file"
        assert adapted.server_name == "test_server"
        assert adapted.source == "mcp"
        assert "mcp" in adapted.tags

    def test_adapt_parameter(self, mock_client):
        """Test parameter adaptation."""
        adapter = MCPToolAdapter(mock_client)

        param = MCPToolParameter(
            name="count",
            param_type="integer",
            description="Count",
            required=True,
            minimum=0,
            maximum=10
        )

        result = adapter._adapt_parameter(param)

        assert result["name"] == "count"
        assert result["type"] == "integer"
        assert result["minimum"] == 0
        assert result["maximum"] == 10

    def test_map_type(self, mock_client):
        """Test type mapping."""
        adapter = MCPToolAdapter(mock_client)

        assert adapter._map_type("string") == "string"
        assert adapter._map_type("integer") == "integer"
        assert adapter._map_type("number") == "number"
        assert adapter._map_type("boolean") == "boolean"
        assert adapter._map_type("object") == "object"
        assert adapter._map_type("array") == "array"
        assert adapter._map_type("unknown") == "string"

    def test_infer_category_filesystem(self, mock_client):
        """Test category inference for filesystem."""
        adapter = MCPToolAdapter(mock_client)

        assert adapter._infer_category("read_file", "filesystem") == "filesystem"
        assert adapter._infer_category("list_dir", "my_server") == "filesystem"

    def test_infer_category_search(self, mock_client):
        """Test category inference for search."""
        adapter = MCPToolAdapter(mock_client)

        assert adapter._infer_category("search", "brave-search") == "search"
        assert adapter._infer_category("query", "my_server") == "search"

    def test_infer_category_web(self, mock_client):
        """Test category inference for web."""
        adapter = MCPToolAdapter(mock_client)

        assert adapter._infer_category("fetch", "web_server") == "web"
        assert adapter._infer_category("http_get", "my_server") == "web"


# ============================================================================
# MCP Manager Tests
# ============================================================================

@pytest.mark.asyncio
class TestMCPManager:
    """Test MCP manager."""

    @pytest.fixture
    def manager(self):
        """Create an MCP manager."""
        return MCPManager()

    @pytest.fixture
    def sample_config(self):
        """Create a sample server config."""
        return MCPServerConfig(
            name="test_server",
            command="echo",
            args=["test"],
            enabled=True
        )

    def test_add_server(self, manager, sample_config):
        """Test adding a server."""
        result = manager.add_server(sample_config)

        assert result is True
        assert "test_server" in manager.configs
        assert "test_server" in manager._status

    def test_add_server_no_name(self, manager):
        """Test adding a server without name."""
        config = MCPServerConfig(name="", command="echo")
        result = manager.add_server(config)

        assert result is False

    def test_remove_server(self, manager, sample_config):
        """Test removing a server."""
        manager.add_server(sample_config)
        result = manager.remove_server("test_server")

        assert result is True
        assert "test_server" not in manager.configs

    def test_get_server_status(self, manager, sample_config):
        """Test getting server status."""
        manager.add_server(sample_config)
        status = manager.get_server_status()

        assert "test_server" in status
        assert status["test_server"].enabled is True
        assert status["test_server"].connected is False

    def test_get_all_tools_empty(self, manager):
        """Test getting tools when none are cached."""
        tools = manager.get_all_tools()
        assert tools == []

    def test_get_tool_not_found(self, manager):
        """Test getting a non-existent tool."""
        tool = manager.get_tool("non_existent")
        assert tool is None


# ============================================================================
# Server Discovery Tests
# ============================================================================

class TestMCPServerDiscovery:
    """Test MCP server discovery."""

    @pytest.fixture
    def discovery(self):
        """Create a discovery instance."""
        return MCPServerDiscovery()

    def test_known_servers_populated(self, discovery):
        """Test that known servers are populated."""
        assert "@modelcontextprotocol/server-filesystem" in discovery.KNOWN_SERVERS
        assert "@modelcontextprotocol/server-github" in discovery.KNOWN_SERVERS
        assert "@upstash/context7-mcp" in discovery.COMMUNITY_SERVERS

    def test_discovered_server_to_config(self, discovery):
        """Test converting discovered server to config."""
        server = DiscoveredServer(
            name="test",
            command="npx",
            args=["-y", "test-pkg"],
            source="npm_global",
            description="Test server",
            tags=["test"]
        )

        config = server.to_config()

        assert config.name == "test"
        assert config.command == "npx"
        assert config.description == "Test server"
        assert config.tags == ["test"]

    @pytest.mark.asyncio
    async def test_discover_all_empty(self, discovery):
        """Test discovery with no servers found."""
        # Mock npm to return empty
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout='{"dependencies": {}}'
            )

            servers = await discovery.discover_all()
            assert servers == []


# ============================================================================
# Server Presets Tests
# ============================================================================

class TestMCPServerPresets:
    """Test MCP server presets."""

    def test_context7_preset(self):
        """Test Context7 preset."""
        config = MCPServerPresets.context7()

        assert config.name == "context7"
        assert config.command == "npx"
        assert "@upstash/context7-mcp" in config.args
        assert "documentation" in config.tags

    def test_playwright_preset(self):
        """Test Playwright preset."""
        config = MCPServerPresets.playwright()

        assert config.name == "playwright"
        assert config.command == "npx"
        assert "@executeautomation/playwright-mcp-server" in config.args
        assert "browser" in config.tags
        assert "automation" in config.tags

    def test_filesystem_preset(self):
        """Test filesystem preset."""
        config = MCPServerPresets.filesystem(allowed_paths=["/tmp", "/home"])

        assert config.name == "filesystem"
        assert "/tmp" in config.args
        assert "/home" in config.args

    def test_filesystem_preset_default(self):
        """Test filesystem preset with default paths."""
        config = MCPServerPresets.filesystem()

        assert config.name == "filesystem"
        assert "." in config.args

    def test_github_preset(self):
        """Test GitHub preset."""
        config = MCPServerPresets.github()

        assert config.name == "github"
        assert "github" in config.tags
        assert "api" in config.tags

    def test_memory_preset(self):
        """Test memory preset."""
        config = MCPServerPresets.memory()

        assert config.name == "memory"
        assert "memory" in config.tags
        assert "knowledge" in config.tags

    def test_brave_search_preset(self):
        """Test Brave Search preset."""
        config = MCPServerPresets.brave_search()

        assert config.name == "brave-search"
        assert "search" in config.tags
        assert "web" in config.tags

    def test_sqlite_preset(self):
        """Test SQLite preset."""
        config = MCPServerPresets.sqlite(db_path="/tmp/test.db")

        assert config.name == "sqlite"
        assert "/tmp/test.db" in config.args
        assert "database" in config.tags

    def test_postgres_preset(self):
        """Test PostgreSQL preset."""
        config = MCPServerPresets.postgres(
            connection_string="postgresql://user:pass@localhost/db"
        )

        assert config.name == "postgres"
        assert "postgresql://user:pass@localhost/db" in config.args
        assert "database" in config.tags

    def test_get_all_presets(self):
        """Test getting all presets."""
        presets = MCPServerPresets.get_all_presets()

        assert "context7" in presets
        assert "playwright" in presets
        assert "filesystem" in presets
        assert "github" in presets
        assert "memory" in presets
        assert "brave-search" in presets
        assert "sqlite" in presets
        assert "postgres" in presets


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
class TestMCPIntegration:
    """Integration tests for MCP."""

    async def test_create_manager_with_presets(self):
        """Test creating manager with presets."""
        manager = await create_mcp_manager_with_presets(
            ["context7", "playwright"]
        )

        assert "context7" in manager.configs
        assert "playwright" in manager.configs

    async def test_create_manager_with_preset_kwargs(self):
        """Test creating manager with preset kwargs."""
        manager = await create_mcp_manager_with_presets(
            ["filesystem"],
            filesystem={"allowed_paths": ["/tmp"]}
        )

        assert "filesystem" in manager.configs
        assert "/tmp" in manager.configs["filesystem"].args


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestMCPErrorHandling:
    """Test MCP error handling."""

    def test_mcp_tool_error(self):
        """Test MCP tool error exception."""
        error = MCPToolError("Test error", code=-32001)

        assert str(error) == "Test error"
        assert error.code == -32001
        assert error.message == "Test error"

    def test_mcp_tool_error_no_code(self):
        """Test MCP tool error without code."""
        error = MCPToolError("Test error")

        assert error.code is None
        assert error.message == "Test error"


# ============================================================================
# Server Status Tests
# ============================================================================

class TestServerStatus:
    """Test server status."""

    def test_server_status_creation(self):
        """Test server status creation."""
        from datetime import datetime

        status = ServerStatus(
            name="test",
            enabled=True,
            connected=True,
            transport="stdio",
            tool_count=5,
            connected_at=datetime.now()
        )

        assert status.name == "test"
        assert status.tool_count == 5

    def test_server_status_to_dict(self):
        """Test server status serialization."""
        status = ServerStatus(
            name="test",
            enabled=True,
            connected=False,
            transport="stdio"
        )

        result = status.to_dict()

        assert result["name"] == "test"
        assert result["enabled"] is True
        assert result["connected"] is False
        assert result["transport"] == "stdio"


# ============================================================================
# AdaptedTool Tests
# ============================================================================

class TestAdaptedTool:
    """Test adapted tool."""

    def test_adapted_tool_creation(self):
        """Test adapted tool creation."""
        async def handler(**kwargs):
            return "result"

        tool = AdaptedTool(
            name="mcp_server_tool",
            original_name="tool",
            description="A tool",
            source="mcp",
            server_name="server",
            parameters=[{"name": "arg", "type": "string"}],
            handler=handler,
            category="test",
            tags=["mcp"]
        )

        assert tool.name == "mcp_server_tool"
        assert tool.category == "test"

    def test_adapted_tool_to_dict(self):
        """Test adapted tool serialization."""
        async def handler(**kwargs):
            return "result"

        tool = AdaptedTool(
            name="mcp_server_tool",
            original_name="tool",
            description="A tool",
            source="mcp",
            server_name="server",
            parameters=[],
            handler=handler
        )

        result = tool.to_dict()

        assert result["name"] == "mcp_server_tool"
        assert result["source"] == "mcp"
        assert result["category"] == "mcp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
