"""Model Context Protocol (MCP) integration for extensible tool support.

This module provides comprehensive integration with MCP servers, enabling dynamic
tool discovery and standardized tool invocation. It implements the full MCP
protocol specification for seamless integration with the MCP ecosystem.

参考设计:
- Cursor MCP: https://docs.cursor.com/context/model-context-protocol
- MCP Specification: https://modelcontextprotocol.io

核心功能:
- 完整的MCP协议支持 (JSON-RPC 2.0)
- MCP Server自动发现 (npm, config files, environment)
- 动态工具加载和热重载
- 标准化工具接口适配
- 支持多种传输协议 (stdio, HTTP, WebSocket)
- 内置常用MCP服务支持 (Context7, Playwright等)
"""

import logging
import json
import asyncio
import os
import subprocess
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Dict, List, Optional, Any, Callable, Union, TypeVar, Generic,
    AsyncIterator, Coroutine, Set, Tuple
)
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# MCP Protocol Types
# ============================================================================

class MCPTransportType(Enum):
    """MCP transport types supported by the protocol."""
    STDIO = "stdio"
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    WEBSOCKET_SECURE = "wss"
    SSE = "sse"  # Server-Sent Events


class MCPErrorCode(Enum):
    """MCP protocol error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000
    TOOL_NOT_FOUND = -32001
    TOOL_EXECUTION_ERROR = -32002


@dataclass
class MCPError:
    """MCP protocol error."""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC error format."""
        result = {"code": self.code, "message": self.message}
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class MCPRequest:
    """MCP JSON-RPC request."""
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method
        }
        if self.params:
            result["params"] = self.params
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class MCPResponse:
    """MCP JSON-RPC response."""
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[MCPError] = None
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResponse":
        """Create from dictionary."""
        error = None
        if "error" in data:
            error_data = data["error"]
            error = MCPError(
                code=error_data.get("code", 0),
                message=error_data.get("message", "Unknown error"),
                data=error_data.get("data")
            )
        return cls(
            id=data.get("id", 0),
            result=data.get("result"),
            error=error,
            jsonrpc=data.get("jsonrpc", "2.0")
        )

    def is_success(self) -> bool:
        """Check if response is successful."""
        return self.error is None


# ============================================================================
# MCP Tool Definitions
# ============================================================================

@dataclass
class MCPToolParameter:
    """MCP tool parameter definition following JSON Schema."""
    name: str
    param_type: str  # string, number, integer, boolean, object, array
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for string validation
    minimum: Optional[float] = None  # For numeric types
    maximum: Optional[float] = None
    min_items: Optional[int] = None  # For array type
    max_items: Optional[int] = None
    items: Optional[Dict[str, Any]] = None  # Array item schema

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        result = {
            "type": self.param_type,
            "description": self.description
        }
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = self.enum
        if self.pattern:
            result["pattern"] = self.pattern
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.min_items is not None:
            result["minItems"] = self.min_items
        if self.max_items is not None:
            result["maxItems"] = self.max_items
        if self.items:
            result["items"] = self.items
        return result


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    parameters: List[MCPToolParameter] = field(default_factory=list)
    returns: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP tool format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_dict()
            if param.required:
                required.append(param.name)

        result = {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

        if self.annotations:
            result["annotations"] = self.annotations

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTool":
        """Create from MCP tool format."""
        schema = data.get("inputSchema", {})
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        parameters = []
        for name, prop in properties.items():
            parameters.append(MCPToolParameter(
                name=name,
                param_type=prop.get("type", "string"),
                description=prop.get("description", ""),
                required=name in required,
                default=prop.get("default"),
                enum=prop.get("enum"),
                pattern=prop.get("pattern"),
                minimum=prop.get("minimum"),
                maximum=prop.get("maximum"),
                min_items=prop.get("minItems"),
                max_items=prop.get("maxItems"),
                items=prop.get("items")
            ))

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=parameters,
            annotations=data.get("annotations")
        )


@dataclass
class MCPToolResult:
    """Result of MCP tool invocation."""
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    error_code: Optional[int] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }

    @classmethod
    def success_result(
        cls,
        result: Any,
        execution_time_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "MCPToolResult":
        """Create a successful result."""
        return cls(
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {}
        )

    @classmethod
    def error_result(
        cls,
        message: str,
        code: int = -32000,
        execution_time_ms: int = 0
    ) -> "MCPToolResult":
        """Create an error result."""
        return cls(
            success=False,
            error_message=message,
            error_code=code,
            execution_time_ms=execution_time_ms
        )


# ============================================================================
# MCP Server Configuration
# ============================================================================

@dataclass
class MCPServerConfig:
    """MCP server configuration."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: MCPTransportType = MCPTransportType.STDIO
    timeout: int = 30
    enabled: bool = True
    auto_connect: bool = True
    retry_count: int = 3
    retry_delay: float = 1.0
    working_dir: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "transport": self.transport.value,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "auto_connect": self.auto_connect,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "working_dir": self.working_dir,
            "description": self.description,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPServerConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            command=data["command"],
            args=data.get("args", []),
            env=data.get("env", {}),
            transport=MCPTransportType(data.get("transport", "stdio")),
            timeout=data.get("timeout", 30),
            enabled=data.get("enabled", True),
            auto_connect=data.get("auto_connect", True),
            retry_count=data.get("retry_count", 3),
            retry_delay=data.get("retry_delay", 1.0),
            working_dir=data.get("working_dir"),
            description=data.get("description"),
            tags=data.get("tags", [])
        )


# ============================================================================
# MCP Client Base Class
# ============================================================================

class MCPClient(ABC):
    """Abstract base class for MCP clients.

    Implements the MCP protocol for communication with MCP servers.
    Supports initialization, tool discovery, and tool invocation.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{config.name}")
        self._available_tools: List[MCPTool] = []
        self._connected = False
        self._connection_info: Optional[Dict[str, Any]] = None
        self._protocol_version: str = "2024-11-05"
        self._request_id = 0

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    @property
    def available_tools(self) -> List[MCPTool]:
        """Get cached list of available tools."""
        return self._available_tools.copy()

    @property
    def connection_info(self) -> Optional[Dict[str, Any]]:
        """Get connection information from server."""
        return self._connection_info

    def _get_request_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to MCP server and initialize session.

        Returns:
            True if connected and initialized successfully
        """
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from MCP server and cleanup resources."""
        pass

    @abstractmethod
    async def _send_request(
        self,
        request: MCPRequest
    ) -> Optional[MCPResponse]:
        """Send JSON-RPC request to server.

        Args:
            request: MCP request

        Returns:
            Response or None if failed
        """
        pass

    async def initialize(self) -> bool:
        """Initialize MCP session with server.

        Returns:
            True if initialized successfully
        """
        init_request = MCPRequest(
            id=self._get_request_id(),
            method="initialize",
            params={
                "protocolVersion": self._protocol_version,
                "capabilities": {
                    "tools": {
                        "listChanged": True
                    },
                    "logging": {}
                },
                "clientInfo": {
                    "name": "pyutagent",
                    "version": "1.0.0"
                }
            }
        )

        response = await self._send_request(init_request)

        if response and response.is_success() and response.result:
            self._connection_info = response.result
            server_info = response.result.get("serverInfo", {})
            self.logger.info(
                f"Initialized with server: {server_info.get('name', 'unknown')} "
                f"v{server_info.get('version', 'unknown')}"
            )
            return True
        else:
            error = response.error if response else None
            self.logger.error(f"Failed to initialize: {error}")
            return False

    async def list_tools(self) -> List[MCPTool]:
        """List available tools from server.

        Returns:
            List of available tools
        """
        if not self._connected:
            self.logger.warning("Not connected to server")
            return []

        request = MCPRequest(
            id=self._get_request_id(),
            method="tools/list"
        )

        response = await self._send_request(request)

        if response and response.is_success() and response.result:
            tools_data = response.result.get("tools", [])
            tools = [MCPTool.from_dict(tool_data) for tool_data in tools_data]
            self._available_tools = tools
            self.logger.info(f"Discovered {len(tools)} tools")
            return tools

        return []

    async def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> MCPToolResult:
        """Invoke a tool on the server.

        Args:
            tool_name: Name of the tool to invoke
            parameters: Tool parameters

        Returns:
            Tool invocation result
        """
        if not self._connected:
            return MCPToolResult.error_result(
                "Not connected to MCP server",
                MCPErrorCode.SERVER_ERROR.value
            )

        import time
        start_time = time.time()

        request = MCPRequest(
            id=self._get_request_id(),
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": parameters
            }
        )

        response = await self._send_request(request)

        execution_time = int((time.time() - start_time) * 1000)

        if not response:
            return MCPToolResult.error_result(
                "No response from MCP server",
                MCPErrorCode.SERVER_ERROR.value,
                execution_time
            )

        if response.error:
            return MCPToolResult.error_result(
                response.error.message,
                response.error.code,
                execution_time
            )

        if response.result:
            result_data = response.result

            # Check for tool error
            if result_data.get("isError"):
                content = result_data.get("content", [])
                error_text = ""
                for item in content:
                    if item.get("type") == "text":
                        error_text += item.get("text", "")
                return MCPToolResult.error_result(
                    error_text or "Tool execution error",
                    MCPErrorCode.TOOL_EXECUTION_ERROR.value,
                    execution_time
                )

            # Extract result content
            content = result_data.get("content", [])
            result_text = ""
            for item in content:
                if item.get("type") == "text":
                    result_text += item.get("text", "")

            return MCPToolResult.success_result(
                result_text,
                execution_time,
                metadata={"content": content}
            )

        return MCPToolResult.error_result(
            "Invalid response from server",
            MCPErrorCode.INTERNAL_ERROR.value,
            execution_time
        )

    async def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a specific tool by name.

        Args:
            tool_name: Tool name

        Returns:
            Tool definition or None
        """
        for tool in self._available_tools:
            if tool.name == tool_name:
                return tool
        return None

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is available.

        Args:
            tool_name: Tool name

        Returns:
            True if tool is available
        """
        return any(tool.name == tool_name for tool in self._available_tools)


# ============================================================================
# STDIO Transport Client
# ============================================================================

class StdioMCPClient(MCPClient):
    """MCP client using stdio transport.

    Communicates with MCP servers through stdin/stdout using JSON-RPC.
    """

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._process: Optional[subprocess.Popen] = None
        self._lock = asyncio.Lock()
        self._buffer = ""

    async def connect(self) -> bool:
        """Connect to MCP server via stdio."""
        async with self._lock:
            if self._connected:
                return True

            try:
                self.logger.info(f"Starting MCP server: {self.config.name}")

                # Prepare environment
                env = os.environ.copy()
                env.update(self.config.env)

                # Prepare working directory
                cwd = self.config.working_dir or os.getcwd()

                # Start process
                self._process = subprocess.Popen(
                    [self.config.command] + self.config.args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    cwd=cwd,
                    bufsize=1  # Line buffered
                )

                # Initialize MCP session
                if await self.initialize():
                    self._connected = True

                    # Load available tools
                    self._available_tools = await self.list_tools()

                    self.logger.info(
                        f"Connected to MCP server: {self.config.name} "
                        f"({len(self._available_tools)} tools)"
                    )
                    return True
                else:
                    await self._cleanup_process()
                    return False

            except Exception as e:
                self.logger.exception(f"Failed to connect to MCP server: {e}")
                await self._cleanup_process()
                return False

    async def disconnect(self):
        """Disconnect from MCP server."""
        async with self._lock:
            await self._cleanup_process()
            self._connected = False
            self._available_tools = []
            self.logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def _cleanup_process(self):
        """Cleanup subprocess."""
        if self._process:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            except Exception as e:
                self.logger.warning(f"Error cleaning up process: {e}")
            finally:
                self._process = None

    async def _send_request(
        self,
        request: MCPRequest
    ) -> Optional[MCPResponse]:
        """Send JSON-RPC request to server via stdio."""
        if not self._process or self._process.poll() is not None:
            self.logger.error("Process not running")
            return None

        try:
            request_json = request.to_json() + "\n"

            # Send request
            self._process.stdin.write(request_json)
            self._process.stdin.flush()

            # Read response with timeout
            response_line = await asyncio.wait_for(
                self._read_line(),
                timeout=self.config.timeout
            )

            if response_line:
                data = json.loads(response_line)
                return MCPResponse.from_dict(data)

            return None

        except asyncio.TimeoutError:
            self.logger.error("Request timeout")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to send request: {e}")
            return None

    async def _read_line(self) -> str:
        """Read a line from stdout asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process.stdout.readline)


# ============================================================================
# HTTP Transport Client
# ============================================================================

class HttpMCPClient(MCPClient):
    """MCP client using HTTP/SSE transport."""

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._session = None
        self._endpoint: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to MCP server via HTTP."""
        try:
            import aiohttp
        except ImportError:
            self.logger.error("aiohttp not installed")
            return False

        try:
            self.logger.info(f"Connecting to HTTP MCP server: {self.config.name}")

            # Parse endpoint from args or use default
            if self.config.args:
                self._endpoint = self.config.args[0]
            else:
                self._endpoint = "http://localhost:3000/mcp"

            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={"Content-Type": "application/json"}
            )

            if await self.initialize():
                self._connected = True
                self._available_tools = await self.list_tools()
                self.logger.info(
                    f"Connected to HTTP MCP server: {self.config.name} "
                    f"({len(self._available_tools)} tools)"
                )
                return True
            else:
                await self._session.close()
                return False

        except Exception as e:
            self.logger.exception(f"Failed to connect: {e}")
            if self._session:
                await self._session.close()
            return False

    async def disconnect(self):
        """Disconnect from HTTP server."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        self.logger.info(f"Disconnected from HTTP MCP server: {self.config.name}")

    async def _send_request(
        self,
        request: MCPRequest
    ) -> Optional[MCPResponse]:
        """Send HTTP request to server."""
        if not self._session:
            return None

        try:
            import aiohttp
        except ImportError:
            return None

        try:
            async with self._session.post(
                self._endpoint,
                json=request.to_dict()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return MCPResponse.from_dict(data)
                else:
                    self.logger.error(f"HTTP error: {response.status}")
                    return None

        except Exception as e:
            self.logger.error(f"HTTP request failed: {e}")
            return None


# ============================================================================
# WebSocket Transport Client
# ============================================================================

class WebSocketMCPClient(MCPClient):
    """MCP client using WebSocket transport."""

    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._websocket = None
        self._uri: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to MCP server via WebSocket."""
        try:
            import websockets
        except ImportError:
            self.logger.error("websockets not installed")
            return False

        try:
            self.logger.info(f"Connecting to WebSocket MCP server: {self.config.name}")

            # Parse URI from args or use default
            if self.config.args:
                self._uri = self.config.args[0]
            else:
                self._uri = "ws://localhost:3000/mcp"

            import websockets

            self._websocket = await websockets.connect(
                self._uri,
                ping_timeout=self.config.timeout
            )

            if await self.initialize():
                self._connected = True
                self._available_tools = await self.list_tools()
                self.logger.info(
                    f"Connected to WebSocket MCP server: {self.config.name} "
                    f"({len(self._available_tools)} tools)"
                )
                return True
            else:
                await self._websocket.close()
                return False

        except Exception as e:
            self.logger.exception(f"Failed to connect: {e}")
            if self._websocket:
                await self._websocket.close()
            return False

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        self._connected = False
        self.logger.info(f"Disconnected from WebSocket MCP server: {self.config.name}")

    async def _send_request(
        self,
        request: MCPRequest
    ) -> Optional[MCPResponse]:
        """Send WebSocket request to server."""
        if not self._websocket:
            return None

        try:
            import websockets
        except ImportError:
            return None

        try:
            await self._websocket.send(request.to_json())

            response_str = await asyncio.wait_for(
                self._websocket.recv(),
                timeout=self.config.timeout
            )

            data = json.loads(response_str)
            return MCPResponse.from_dict(data)

        except asyncio.TimeoutError:
            self.logger.error("WebSocket request timeout")
            return None
        except Exception as e:
            self.logger.error(f"WebSocket request failed: {e}")
            return None


# ============================================================================
# MCP Tool Adapter
# ============================================================================

@dataclass
class AdaptedTool:
    """Internal tool representation adapted from MCP tool."""
    name: str
    original_name: str
    description: str
    source: str
    server_name: str
    parameters: List[Dict[str, Any]]
    handler: Callable
    category: str = "mcp"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "original_name": self.original_name,
            "description": self.description,
            "source": self.source,
            "server_name": self.server_name,
            "parameters": self.parameters,
            "category": self.category,
            "tags": self.tags,
            "metadata": self.metadata
        }


class MCPToolAdapter:
    """Adapter to convert MCP tools to internal tool format.

    Provides seamless integration between MCP tools and the internal
    tool system with parameter transformation and result conversion.
    """

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(self.__class__.__name__)

    def adapt_tool(self, mcp_tool: MCPTool) -> AdaptedTool:
        """Adapt MCP tool to internal format.

        Args:
            mcp_tool: MCP tool definition

        Returns:
            Adapted internal tool representation
        """
        server_name = self.mcp_client.config.name
        adapted_name = f"mcp_{server_name}_{mcp_tool.name}"

        # Build tags from server config and tool name
        tags = list(self.mcp_client.config.tags)
        tags.append("mcp")
        tags.append(server_name)

        # Infer category from tool name or server
        category = self._infer_category(mcp_tool.name, server_name)

        return AdaptedTool(
            name=adapted_name,
            original_name=mcp_tool.name,
            description=mcp_tool.description,
            source="mcp",
            server_name=server_name,
            parameters=[self._adapt_parameter(p) for p in mcp_tool.parameters],
            handler=self._create_handler(mcp_tool.name),
            category=category,
            tags=tags,
            metadata={
                "mcp_tool": mcp_tool.to_dict(),
                "server_config": self.mcp_client.config.to_dict()
            }
        )

    def _adapt_parameter(self, param: MCPToolParameter) -> Dict[str, Any]:
        """Adapt MCP parameter to internal format."""
        result = {
            "name": param.name,
            "type": self._map_type(param.param_type),
            "description": param.description,
            "required": param.required,
            "default": param.default
        }

        if param.enum:
            result["enum"] = param.enum
        if param.pattern:
            result["pattern"] = param.pattern
        if param.minimum is not None:
            result["minimum"] = param.minimum
        if param.maximum is not None:
            result["maximum"] = param.maximum

        return result

    def _map_type(self, mcp_type: str) -> str:
        """Map MCP type to internal type."""
        type_mapping = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "object": "object",
            "array": "array"
        }
        return type_mapping.get(mcp_type, "string")

    def _create_handler(self, tool_name: str) -> Callable:
        """Create handler function for tool."""
        async def handler(**kwargs) -> Any:
            result = await self.mcp_client.invoke_tool(tool_name, kwargs)

            if result.success:
                return result.result
            else:
                raise MCPToolError(
                    result.error_message or "Unknown error",
                    result.error_code
                )

        return handler

    def _infer_category(self, tool_name: str, server_name: str) -> str:
        """Infer tool category from name."""
        name_lower = tool_name.lower()
        server_lower = server_name.lower()

        # Check server name first
        if "file" in server_lower:
            return "filesystem"
        if "search" in server_lower:
            return "search"
        if "web" in server_lower:
            return "web"
        if "db" in server_lower or "sql" in server_lower:
            return "database"
        if "git" in server_lower:
            return "git"

        # Check tool name
        if any(kw in name_lower for kw in ["read", "write", "file", "dir"]):
            return "filesystem"
        if any(kw in name_lower for kw in ["search", "find", "query"]):
            return "search"
        if any(kw in name_lower for kw in ["fetch", "http", "web", "url"]):
            return "web"

        return "mcp"


class MCPToolError(Exception):
    """MCP tool execution error."""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code
        self.message = message


# ============================================================================
# MCP Manager
# ============================================================================

@dataclass
class ServerStatus:
    """MCP server connection status."""
    name: str
    enabled: bool
    connected: bool
    transport: str
    tool_count: int = 0
    last_error: Optional[str] = None
    connected_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "connected": self.connected,
            "transport": self.transport,
            "tool_count": self.tool_count,
            "last_error": self.last_error,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None
        }


class MCPManager:
    """Manager for multiple MCP servers.

    Provides centralized management of MCP server connections,
    tool discovery, and invocation with support for multiple
    transport protocols.
    """

    def __init__(self):
        """Initialize MCP manager."""
        self.clients: Dict[str, MCPClient] = {}
        self.configs: Dict[str, MCPServerConfig] = {}
        self._status: Dict[str, ServerStatus] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._adapters: Dict[str, MCPToolAdapter] = {}
        self._tool_cache: Dict[str, AdaptedTool] = {}

        logger.info("[MCPManager] Initialized")

    def add_server(self, config: MCPServerConfig) -> bool:
        """Add MCP server configuration.

        Args:
            config: Server configuration

        Returns:
            True if added successfully
        """
        if not config.name:
            self.logger.error("Server name is required")
            return False

        self.configs[config.name] = config
        self._status[config.name] = ServerStatus(
            name=config.name,
            enabled=config.enabled,
            connected=False,
            transport=config.transport.value
        )

        self.logger.info(f"[MCPManager] Added server: {config.name}")
        return True

    def remove_server(self, name: str) -> bool:
        """Remove a server configuration.

        Args:
            name: Server name

        Returns:
            True if removed
        """
        if name not in self.configs:
            return False

        # Disconnect if connected
        if name in self.clients:
            asyncio.create_task(self.clients[name].disconnect())
            del self.clients[name]

        del self.configs[name]
        del self._status[name]

        # Remove cached tools from this server
        tools_to_remove = [
            name for name, tool in self._tool_cache.items()
            if tool.server_name == name
        ]
        for tool_name in tools_to_remove:
            del self._tool_cache[tool_name]

        self.logger.info(f"[MCPManager] Removed server: {name}")
        return True

    def load_config_from_file(self, config_path: str) -> int:
        """Load MCP server configurations from file.

        Args:
            config_path: Path to JSON config file

        Returns:
            Number of servers loaded
        """
        try:
            path = Path(config_path)
            if not path.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                return 0

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            count = 0
            servers_data = config_data.get("mcpServers", {})

            # Handle both dict and list formats
            if isinstance(servers_data, dict):
                servers_data = [
                    {"name": name, **data}
                    for name, data in servers_data.items()
                ]

            for server_data in servers_data:
                try:
                    config = MCPServerConfig(
                        name=server_data["name"],
                        command=server_data["command"],
                        args=server_data.get("args", []),
                        env=server_data.get("env", {}),
                        transport=MCPTransportType(
                            server_data.get("transport", "stdio")
                        ),
                        timeout=server_data.get("timeout", 30),
                        enabled=server_data.get("enabled", True),
                        auto_connect=server_data.get("auto_connect", True),
                        retry_count=server_data.get("retry_count", 3),
                        retry_delay=server_data.get("retry_delay", 1.0),
                        working_dir=server_data.get("working_dir"),
                        description=server_data.get("description"),
                        tags=server_data.get("tags", [])
                    )

                    if self.add_server(config):
                        count += 1

                except Exception as e:
                    self.logger.warning(
                        f"[MCPManager] Failed to parse server config: {e}"
                    )

            self.logger.info(f"[MCPManager] Loaded {count} servers from {config_path}")
            return count

        except json.JSONDecodeError as e:
            self.logger.error(f"[MCPManager] Invalid JSON in config: {e}")
            return 0
        except Exception as e:
            self.logger.error(f"[MCPManager] Failed to load config: {e}")
            return 0

    def _create_client(self, config: MCPServerConfig) -> Optional[MCPClient]:
        """Create appropriate client for transport type."""
        if config.transport == MCPTransportType.STDIO:
            return StdioMCPClient(config)
        elif config.transport in (MCPTransportType.HTTP, MCPTransportType.HTTPS):
            return HttpMCPClient(config)
        elif config.transport in (MCPTransportType.WEBSOCKET, MCPTransportType.WEBSOCKET_SECURE):
            return WebSocketMCPClient(config)
        else:
            self.logger.warning(f"Unsupported transport: {config.transport}")
            return None

    async def connect_server(self, name: str) -> bool:
        """Connect to a specific server.

        Args:
            name: Server name

        Returns:
            True if connected successfully
        """
        if name not in self.configs:
            self.logger.error(f"Server not found: {name}")
            return False

        config = self.configs[name]

        if not config.enabled:
            self.logger.info(f"Server {name} is disabled")
            return False

        # Create client
        client = self._create_client(config)
        if not client:
            return False

        # Try to connect with retries
        for attempt in range(config.retry_count):
            try:
                success = await client.connect()

                if success:
                    self.clients[name] = client
                    self._adapters[name] = MCPToolAdapter(client)

                    # Update status
                    self._status[name].connected = True
                    self._status[name].tool_count = len(client.available_tools)
                    self._status[name].connected_at = datetime.now()
                    self._status[name].last_error = None

                    # Cache adapted tools
                    for tool in client.available_tools:
                        adapted = self._adapters[name].adapt_tool(tool)
                        self._tool_cache[adapted.name] = adapted

                    return True

            except Exception as e:
                self.logger.warning(
                    f"Connection attempt {attempt + 1} failed for {name}: {e}"
                )
                self._status[name].last_error = str(e)

                if attempt < config.retry_count - 1:
                    await asyncio.sleep(config.retry_delay)

        return False

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all enabled MCP servers.

        Returns:
            Dictionary of server name to connection status
        """
        results = {}

        for name, config in self.configs.items():
            if not config.enabled:
                results[name] = False
                continue

            if not config.auto_connect:
                results[name] = False
                continue

            results[name] = await self.connect_server(name)

        connected_count = sum(1 for v in results.values() if v)
        self.logger.info(
            f"[MCPManager] Connected to {connected_count}/{len(results)} servers"
        )

        return results

    async def disconnect_server(self, name: str):
        """Disconnect from a specific server."""
        if name in self.clients:
            await self.clients[name].disconnect()
            del self.clients[name]

            if name in self._adapters:
                del self._adapters[name]

            self._status[name].connected = False
            self._status[name].tool_count = 0

            # Remove cached tools
            tools_to_remove = [
                tool_name for tool_name, tool in self._tool_cache.items()
                if tool.server_name == name
            ]
            for tool_name in tools_to_remove:
                del self._tool_cache[tool_name]

            self.logger.info(f"[MCPManager] Disconnected from {name}")

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for name in list(self.clients.keys()):
            await self.disconnect_server(name)

        self.logger.info("[MCPManager] Disconnected from all servers")

    def get_all_tools(self) -> List[AdaptedTool]:
        """Get all tools from all connected MCP servers.

        Returns:
            List of adapted tools
        """
        return list(self._tool_cache.values())

    def get_tools_by_server(self, server_name: str) -> List[AdaptedTool]:
        """Get tools from a specific server.

        Args:
            server_name: MCP server name

        Returns:
            List of adapted tools
        """
        return [
            tool for tool in self._tool_cache.values()
            if tool.server_name == server_name
        ]

    def get_tool(self, tool_name: str) -> Optional[AdaptedTool]:
        """Get a specific tool by name.

        Args:
            tool_name: Tool name (adapted or original)

        Returns:
            Tool or None
        """
        # Try exact match first
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]

        # Try matching original name
        for tool in self._tool_cache.values():
            if tool.original_name == tool_name:
                return tool

        return None

    async def invoke_tool(
        self,
        server_name: str,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> MCPToolResult:
        """Invoke a tool on a specific server.

        Args:
            server_name: MCP server name
            tool_name: Tool name
            parameters: Tool parameters

        Returns:
            Tool invocation result
        """
        if server_name not in self.clients:
            return MCPToolResult.error_result(
                f"Server not found: {server_name}",
                MCPErrorCode.TOOL_NOT_FOUND.value
            )

        client = self.clients[server_name]
        return await client.invoke_tool(tool_name, parameters)

    async def invoke_tool_by_name(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> MCPToolResult:
        """Invoke a tool by its adapted name.

        Args:
            tool_name: Adapted tool name (mcp_{server}_{tool})
            parameters: Tool parameters

        Returns:
            Tool invocation result
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return MCPToolResult.error_result(
                f"Tool not found: {tool_name}",
                MCPErrorCode.TOOL_NOT_FOUND.value
            )

        return await self.invoke_tool(
            tool.server_name,
            tool.original_name,
            parameters
        )

    def get_server_status(self) -> Dict[str, ServerStatus]:
        """Get status of all MCP servers.

        Returns:
            Dictionary of server status
        """
        return self._status.copy()

    def get_server_status_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get server status as dictionaries.

        Returns:
            Dictionary of server status dictionaries
        """
        return {
            name: status.to_dict()
            for name, status in self._status.items()
        }

    @asynccontextmanager
    async def session(self):
        """Async context manager for MCP session.

        Automatically connects to all enabled servers on enter
        and disconnects on exit.

        Example:
            async with manager.session():
                tools = manager.get_all_tools()
                # Use tools...
        """
        await self.connect_all()
        try:
            yield self
        finally:
            await self.disconnect_all()


# ============================================================================
# MCP Server Discovery
# ============================================================================

@dataclass
class DiscoveredServer:
    """A discovered MCP server."""
    name: str
    command: str
    args: List[str]
    source: str
    config_path: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_config(self) -> MCPServerConfig:
        """Convert to server configuration."""
        return MCPServerConfig(
            name=self.name,
            command=self.command,
            args=self.args,
            description=self.description,
            tags=self.tags
        )


class MCPServerDiscovery:
    """MCP Server auto-discovery.

    Discovers MCP servers from various sources:
    - npm global packages
    - Local node_modules
    - Config files (~/.config/mcp/*.json)
    - Project mcp.json
    - Environment variables
    """

    # Known official MCP server packages
    KNOWN_SERVERS = {
        "@modelcontextprotocol/server-filesystem": {
            "name": "filesystem",
            "description": "File system operations",
            "args_template": ["{path}"],
            "tags": ["filesystem", "file", "official"]
        },
        "@modelcontextprotocol/server-github": {
            "name": "github",
            "description": "GitHub API integration",
            "args_template": [],
            "tags": ["github", "git", "api", "official"]
        },
        "@modelcontextprotocol/server-brave-search": {
            "name": "brave-search",
            "description": "Brave Search API",
            "args_template": [],
            "tags": ["search", "web", "brave", "official"]
        },
        "@modelcontextprotocol/server-slack": {
            "name": "slack",
            "description": "Slack integration",
            "args_template": [],
            "tags": ["slack", "messaging", "official"]
        },
        "@modelcontextprotocol/server-postgres": {
            "name": "postgres",
            "description": "PostgreSQL database access",
            "args_template": ["{connection_string}"],
            "tags": ["database", "postgres", "sql", "official"]
        },
        "@modelcontextprotocol/server-sqlite": {
            "name": "sqlite",
            "description": "SQLite database access",
            "args_template": ["{db_path}"],
            "tags": ["database", "sqlite", "sql", "official"]
        },
        "@modelcontextprotocol/server-memory": {
            "name": "memory",
            "description": "Knowledge graph memory",
            "args_template": [],
            "tags": ["memory", "knowledge", "official"]
        },
        "@modelcontextprotocol/server-time": {
            "name": "time",
            "description": "Time and timezone utilities",
            "args_template": [],
            "tags": ["time", "utility", "official"]
        },
        "@modelcontextprotocol/server-jupyter": {
            "name": "jupyter",
            "description": "Jupyter notebook integration",
            "args_template": [],
            "tags": ["jupyter", "notebook", "data", "official"]
        },
        "@modelcontextprotocol/server-puppeteer": {
            "name": "puppeteer",
            "description": "Browser automation with Puppeteer",
            "args_template": [],
            "tags": ["browser", "automation", "puppeteer", "official"]
        },
    }

    # Community servers
    COMMUNITY_SERVERS = {
        "@upstash/context7-mcp": {
            "name": "context7",
            "description": "Context7 documentation search",
            "args_template": [],
            "tags": ["documentation", "search", "context7", "community"]
        },
        "@executeautomation/playwright-mcp-server": {
            "name": "playwright",
            "description": "Playwright browser automation",
            "args_template": [],
            "tags": ["browser", "automation", "playwright", "testing", "community"]
        },
        "@kazuph/mcp-fetch": {
            "name": "fetch",
            "description": "HTTP fetch operations",
            "args_template": [],
            "tags": ["http", "fetch", "web", "community"]
        },
    }

    def __init__(self):
        """Initialize MCP server discovery."""
        self._discovered_servers: List[DiscoveredServer] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    async def discover_all(self) -> List[DiscoveredServer]:
        """Discover all MCP servers from all sources.

        Returns:
            List of discovered servers
        """
        self._discovered_servers.clear()

        await self._discover_npm_global()
        await self._discover_local_npm()
        await self._discover_config_files()
        await self._discover_project_config()
        await self._discover_environment()

        self.logger.info(
            f"[MCPServerDiscovery] Discovered {len(self._discovered_servers)} MCP servers"
        )
        return self._discovered_servers

    async def _discover_npm_global(self):
        """Discover MCP servers from npm global packages."""
        try:
            result = subprocess.run(
                ["npm", "list", "-g", "--depth=0", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                dependencies = data.get("dependencies", {})

                for pkg_name, pkg_info in {**self.KNOWN_SERVERS, **self.COMMUNITY_SERVERS}.items():
                    if pkg_name in dependencies:
                        server = DiscoveredServer(
                            name=pkg_info["name"],
                            command="npx",
                            args=["-y", pkg_name] + pkg_info.get("args_template", []),
                            source="npm_global",
                            description=pkg_info.get("description"),
                            tags=pkg_info.get("tags", [])
                        )
                        self._discovered_servers.append(server)
                        self.logger.info(f"[MCPServerDiscovery] Found npm global: {pkg_name}")

        except Exception as e:
            self.logger.warning(f"[MCPServerDiscovery] npm discovery failed: {e}")

    async def _discover_local_npm(self):
        """Discover MCP servers from local node_modules."""
        local_paths = [
            Path("node_modules"),
            Path.cwd() / "node_modules",
        ]

        for base_path in local_paths:
            if not base_path.exists():
                continue

            for pkg_name in {**self.KNOWN_SERVERS, **self.COMMUNITY_SERVERS}:
                pkg_path = base_path / pkg_name
                if pkg_path.exists():
                    pkg_info = {**self.KNOWN_SERVERS, **self.COMMUNITY_SERVERS}[pkg_name]
                    server = DiscoveredServer(
                        name=pkg_info["name"],
                        command="npx",
                        args=[pkg_name] + pkg_info.get("args_template", []),
                        source=f"local_npm:{base_path}",
                        description=pkg_info.get("description"),
                        tags=pkg_info.get("tags", [])
                    )
                    self._discovered_servers.append(server)
                    self.logger.info(f"[MCPServerDiscovery] Found local npm: {pkg_name}")

    async def _discover_config_files(self):
        """Discover MCP servers from user config files."""
        config_paths = [
            Path.home() / ".config" / "mcp",
            Path.home() / ".mcp",
            Path.home() / ".cursor" / "mcp.json",  # Cursor IDE config
        ]

        for config_path in config_paths:
            if config_path.is_file():
                # Single file
                await self._parse_config_file(config_path, "user_config")
            elif config_path.is_dir():
                # Directory with config files
                for config_file in config_path.glob("*.json"):
                    await self._parse_config_file(config_file, "user_config")

    async def _discover_project_config(self):
        """Discover MCP servers from project config files."""
        project_configs = [
            Path("mcp.json"),
            Path(".mcp.json"),
            Path(".mcp") / "mcp.json",
            Path(".vscode") / "mcp.json",
            Path(".cursor") / "mcp.json",
        ]

        for config_file in project_configs:
            if config_file.exists():
                await self._parse_config_file(config_file, "project")

    async def _discover_environment(self):
        """Discover MCP servers from environment variables."""
        # Check for MCP_CONFIG_PATH
        config_path = os.environ.get("MCP_CONFIG_PATH")
        if config_path and Path(config_path).exists():
            await self._parse_config_file(Path(config_path), "environment")

        # Check for individual server configs
        for key, value in os.environ.items():
            if key.startswith("MCP_SERVER_") and key.endswith("_COMMAND"):
                name = key[11:-8].lower()  # Extract server name
                args_str = os.environ.get(f"MCP_SERVER_{name.upper()}_ARGS", "")
                args = args_str.split() if args_str else []

                server = DiscoveredServer(
                    name=name,
                    command=value,
                    args=args,
                    source="environment"
                )
                self._discovered_servers.append(server)
                self.logger.info(f"[MCPServerDiscovery] Found from env: {name}")

    async def _parse_config_file(self, config_file: Path, source: str):
        """Parse a config file and extract servers."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Handle different config formats
            servers = config.get("mcpServers", {})

            for server_name, server_config in servers.items():
                command = server_config.get("command", "")
                args = server_config.get("args", [])

                server = DiscoveredServer(
                    name=server_name,
                    command=command,
                    args=args,
                    source=source,
                    config_path=str(config_file),
                    description=server_config.get("description"),
                    tags=server_config.get("tags", [])
                )
                self._discovered_servers.append(server)
                self.logger.info(f"[MCPServerDiscovery] Found config server: {server_name}")

        except Exception as e:
            self.logger.warning(f"[MCPServerDiscovery] Failed to parse {config_file}: {e}")

    def get_discovered_servers(self) -> List[DiscoveredServer]:
        """Get list of discovered servers."""
        return self._discovered_servers.copy()

    def get_servers_by_tag(self, tag: str) -> List[DiscoveredServer]:
        """Get servers by tag."""
        return [s for s in self._discovered_servers if tag in s.tags]

    def get_server_by_name(self, name: str) -> Optional[DiscoveredServer]:
        """Get server by name."""
        for server in self._discovered_servers:
            if server.name == name:
                return server
        return None


# ============================================================================
# Built-in MCP Server Configurations
# ============================================================================

class MCPServerPresets:
    """Built-in MCP server configurations for popular services."""

    @staticmethod
    def context7() -> MCPServerConfig:
        """Context7 MCP server for documentation search.

        Provides access to Context7 documentation search capabilities.
        Requires CONTEXT7_API_KEY environment variable.
        """
        return MCPServerConfig(
            name="context7",
            command="npx",
            args=["-y", "@upstash/context7-mcp"],
            description="Context7 documentation search and retrieval",
            tags=["documentation", "search", "context7", "ai"],
            env={}
        )

    @staticmethod
    def playwright() -> MCPServerConfig:
        """Playwright MCP server for browser automation.

        Provides browser automation capabilities for testing and
        web scraping using Playwright.
        """
        return MCPServerConfig(
            name="playwright",
            command="npx",
            args=["-y", "@executeautomation/playwright-mcp-server"],
            description="Playwright browser automation and testing",
            tags=["browser", "automation", "playwright", "testing", "scraping"],
            env={}
        )

    @staticmethod
    def filesystem(allowed_paths: Optional[List[str]] = None) -> MCPServerConfig:
        """Filesystem MCP server.

        Args:
            allowed_paths: List of paths the server can access
        """
        paths = allowed_paths or ["."]
        return MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"] + paths,
            description="File system operations",
            tags=["filesystem", "file", "official"]
        )

    @staticmethod
    def github() -> MCPServerConfig:
        """GitHub MCP server.

        Requires GITHUB_TOKEN environment variable.
        """
        return MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            description="GitHub API integration",
            tags=["github", "git", "api", "official"],
            env={}
        )

    @staticmethod
    def memory() -> MCPServerConfig:
        """Memory MCP server for knowledge graph."""
        return MCPServerConfig(
            name="memory",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            description="Knowledge graph memory storage",
            tags=["memory", "knowledge", "graph", "official"]
        )

    @staticmethod
    def brave_search() -> MCPServerConfig:
        """Brave Search MCP server.

        Requires BRAVE_API_KEY environment variable.
        """
        return MCPServerConfig(
            name="brave-search",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            description="Brave Search API",
            tags=["search", "web", "brave", "official"],
            env={}
        )

    @staticmethod
    def sqlite(db_path: str = "./data.db") -> MCPServerConfig:
        """SQLite MCP server.

        Args:
            db_path: Path to SQLite database file
        """
        return MCPServerConfig(
            name="sqlite",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-sqlite", db_path],
            description="SQLite database access",
            tags=["database", "sqlite", "sql", "official"]
        )

    @staticmethod
    def postgres(connection_string: str) -> MCPServerConfig:
        """PostgreSQL MCP server.

        Args:
            connection_string: PostgreSQL connection string
        """
        return MCPServerConfig(
            name="postgres",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres", connection_string],
            description="PostgreSQL database access",
            tags=["database", "postgres", "sql", "official"]
        )

    @classmethod
    def get_all_presets(cls) -> Dict[str, Callable]:
        """Get all available presets."""
        return {
            "context7": cls.context7,
            "playwright": cls.playwright,
            "filesystem": cls.filesystem,
            "github": cls.github,
            "memory": cls.memory,
            "brave-search": cls.brave_search,
            "sqlite": cls.sqlite,
            "postgres": cls.postgres,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

async def discover_mcp_tools(
    config_path: Optional[str] = None,
    auto_discover_servers: bool = True
) -> List[AdaptedTool]:
    """Discover all available MCP tools.

    Args:
        config_path: Path to MCP config file
        auto_discover_servers: Whether to auto-discover servers

    Returns:
        List of available tools
    """
    manager = MCPManager()

    if config_path:
        manager.load_config_from_file(config_path)

    if auto_discover_servers and not manager.configs:
        discovery = MCPServerDiscovery()
        servers = await discovery.discover_all()

        for server in servers:
            manager.add_server(server.to_config())

    async with manager.session():
        return manager.get_all_tools()


def create_mcp_tool_wrapper(
    server_name: str,
    tool_name: str,
    manager: MCPManager
) -> Callable:
    """Create a wrapper function for an MCP tool.

    Args:
        server_name: MCP server name
        tool_name: Tool name
        manager: MCP manager instance

    Returns:
        Async wrapper function
    """
    async def wrapper(**kwargs) -> Any:
        result = await manager.invoke_tool(server_name, tool_name, kwargs)

        if result.success:
            return result.result
        else:
            raise MCPToolError(
                f"MCP tool error: {result.error_message}",
                result.error_code
            )

    return wrapper


async def create_mcp_manager_with_presets(
    presets: List[str],
    **preset_kwargs
) -> MCPManager:
    """Create MCP manager with preset servers.

    Args:
        presets: List of preset names to enable
        **preset_kwargs: Keyword arguments for preset configurations

    Returns:
        Configured MCPManager
    """
    manager = MCPManager()
    available_presets = MCPServerPresets.get_all_presets()

    for preset_name in presets:
        if preset_name in available_presets:
            try:
                preset_func = available_presets[preset_name]
                # Get kwargs for this preset
                kwargs = preset_kwargs.get(preset_name, {})
                config = preset_func(**kwargs) if kwargs else preset_func()
                manager.add_server(config)
                logger.info(f"Added preset server: {preset_name}")
            except Exception as e:
                logger.warning(f"Failed to add preset {preset_name}: {e}")
        else:
            logger.warning(f"Unknown preset: {preset_name}")

    return manager
