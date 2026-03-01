"""Model Context Protocol (MCP) integration for extensible tool support.

This module provides integration with MCP servers, enabling dynamic
tool discovery and standardized tool invocation.
"""

import logging
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import subprocess
import os

logger = logging.getLogger(__name__)


class MCPTransportType(Enum):
    """MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"


@dataclass
class MCPToolParameter:
    """MCP tool parameter definition."""
    name: str
    param_type: str  # string, number, boolean, object, array
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    parameters: List[MCPToolParameter] = field(default_factory=list)
    returns: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.param_type,
                    "description": p.description,
                    "required": p.required,
                    **({"default": p.default} if p.default is not None else {}),
                    **({"enum": p.enum} if p.enum else {})
                }
                for p in self.parameters
            ]
        }


@dataclass
class MCPToolResult:
    """Result of MCP tool invocation."""
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    execution_time_ms: int = 0


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


class MCPClient(ABC):
    """Abstract base class for MCP clients."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{config.name}")
        self._available_tools: List[MCPTool] = []
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to MCP server."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from MCP server."""
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools from server."""
        pass
    
    @abstractmethod
    async def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> MCPToolResult:
        """Invoke a tool on the server."""
        pass
    
    @property
    def available_tools(self) -> List[MCPTool]:
        """Get cached list of available tools."""
        return self._available_tools


class StdioMCPClient(MCPClient):
    """MCP client using stdio transport."""
    
    def __init__(self, config: MCPServerConfig):
        super().__init__(config)
        self._process: Optional[subprocess.Popen] = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to MCP server via stdio."""
        try:
            self.logger.info(f"Connecting to MCP server: {self.config.name}")
            
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.env)
            
            # Start process
            self._process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Initialize MCP session
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "pyutagent",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = await self._send_request(init_request)
            
            if response and "result" in response:
                self._connected = True
                self.logger.info(f"Connected to MCP server: {self.config.name}")
                
                # Load available tools
                self._available_tools = await self.list_tools()
                
                return True
            else:
                self.logger.error(f"Failed to initialize MCP server: {response}")
                return False
                
        except Exception as e:
            self.logger.exception(f"Failed to connect to MCP server: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except:
                self._process.kill()
            finally:
                self._process = None
                self._connected = False
                self.logger.info(f"Disconnected from MCP server: {self.config.name}")
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools."""
        if not self._connected:
            return []
        
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        response = await self._send_request(request)
        
        if response and "result" in response:
            tools_data = response["result"].get("tools", [])
            tools = []
            
            for tool_data in tools_data:
                parameters = []
                for param_data in tool_data.get("parameters", []):
                    parameters.append(MCPToolParameter(
                        name=param_data["name"],
                        param_type=param_data["type"],
                        description=param_data.get("description", ""),
                        required=param_data.get("required", False),
                        default=param_data.get("default"),
                        enum=param_data.get("enum")
                    ))
                
                tools.append(MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    parameters=parameters
                ))
            
            return tools
        
        return []
    
    async def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> MCPToolResult:
        """Invoke a tool."""
        if not self._connected:
            return MCPToolResult(
                success=False,
                error_message="Not connected to MCP server"
            )
        
        import time
        start_time = time.time()
        
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }
        
        response = await self._send_request(request)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        if response:
            if "error" in response:
                return MCPToolResult(
                    success=False,
                    error_message=response["error"].get("message", "Unknown error"),
                    execution_time_ms=execution_time
                )
            
            if "result" in response:
                result_data = response["result"]
                
                # Check for tool error
                if result_data.get("isError"):
                    return MCPToolResult(
                        success=False,
                        error_message=result_data.get("content", [{}])[0].get("text", "Tool error"),
                        execution_time_ms=execution_time
                    )
                
                # Extract result content
                content = result_data.get("content", [])
                result_text = ""
                for item in content:
                    if item.get("type") == "text":
                        result_text += item.get("text", "")
                
                return MCPToolResult(
                    success=True,
                    result=result_text,
                    execution_time_ms=execution_time
                )
        
        return MCPToolResult(
            success=False,
            error_message="No response from MCP server",
            execution_time_ms=execution_time
        )
    
    async def _send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send JSON-RPC request to server."""
        if not self._process:
            return None
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self._process.stdin.write(request_json)
            self._process.stdin.flush()
            
            # Read response
            response_line = self._process.stdout.readline()
            
            if response_line:
                return json.loads(response_line)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to send request: {e}")
            return None


class MCPToolAdapter:
    """Adapter to convert MCP tools to internal tool format."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def adapt_tool(self, mcp_tool: MCPTool) -> Dict[str, Any]:
        """Adapt MCP tool to internal format.
        
        Args:
            mcp_tool: MCP tool definition
            
        Returns:
            Internal tool format
        """
        return {
            "name": f"mcp_{self.mcp_client.config.name}_{mcp_tool.name}",
            "original_name": mcp_tool.name,
            "description": mcp_tool.description,
            "source": "mcp",
            "mcp_client": self.mcp_client.config.name,
            "parameters": [self._adapt_parameter(p) for p in mcp_tool.parameters],
            "handler": self._create_handler(mcp_tool.name)
        }
    
    def _adapt_parameter(self, param: MCPToolParameter) -> Dict[str, Any]:
        """Adapt MCP parameter to internal format."""
        return {
            "name": param.name,
            "type": self._map_type(param.param_type),
            "description": param.description,
            "required": param.required,
            "default": param.default
        }
    
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
                raise Exception(result.error_message)
        
        return handler


class MCPManager:
    """Manager for multiple MCP servers."""
    
    def __init__(self):
        """Initialize MCP manager."""
        self.clients: Dict[str, MCPClient] = {}
        self.configs: Dict[str, MCPServerConfig] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        logger.info("[MCPManager] Initialized")
    
    def add_server(self, config: MCPServerConfig):
        """Add MCP server configuration.
        
        Args:
            config: Server configuration
        """
        self.configs[config.name] = config
        self.logger.info(f"[MCPManager] Added server: {config.name}")
    
    def load_config_from_file(self, config_path: str):
        """Load MCP server configurations from file.
        
        Args:
            config_path: Path to JSON config file
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for server_data in config_data.get("mcpServers", []):
                config = MCPServerConfig(
                    name=server_data["name"],
                    command=server_data["command"],
                    args=server_data.get("args", []),
                    env=server_data.get("env", {}),
                    transport=MCPTransportType(server_data.get("transport", "stdio")),
                    timeout=server_data.get("timeout", 30),
                    enabled=server_data.get("enabled", True)
                )
                self.add_server(config)
                
        except Exception as e:
            self.logger.error(f"Failed to load MCP config: {e}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all enabled MCP servers.
        
        Returns:
            Dictionary of server name to connection status
        """
        results = {}
        
        for name, config in self.configs.items():
            if not config.enabled:
                continue
            
            # Create client based on transport
            if config.transport == MCPTransportType.STDIO:
                client = StdioMCPClient(config)
            else:
                self.logger.warning(f"Unsupported transport: {config.transport}")
                continue
            
            # Connect
            success = await client.connect()
            
            if success:
                self.clients[name] = client
            
            results[name] = success
        
        return results
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for name, client in self.clients.items():
            await client.disconnect()
        
        self.clients.clear()
        self.logger.info("[MCPManager] Disconnected from all servers")
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all connected MCP servers.
        
        Returns:
            List of adapted tools
        """
        all_tools = []
        
        for name, client in self.clients.items():
            adapter = MCPToolAdapter(client)
            
            for mcp_tool in client.available_tools:
                adapted = adapter.adapt_tool(mcp_tool)
                all_tools.append(adapted)
        
        return all_tools
    
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
            return MCPToolResult(
                success=False,
                error_message=f"Server not found: {server_name}"
            )
        
        client = self.clients[server_name]
        return await client.invoke_tool(tool_name, parameters)
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all MCP servers.
        
        Returns:
            Dictionary of server status
        """
        status = {}
        
        for name, config in self.configs.items():
            connected = name in self.clients
            
            status[name] = {
                "enabled": config.enabled,
                "connected": connected,
                "transport": config.transport.value,
                "tool_count": len(self.clients[name].available_tools) if connected else 0
            }
        
        return status


# Convenience functions

async def discover_mcp_tools(config_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover all available MCP tools.
    
    Args:
        config_path: Path to MCP config file
        
    Returns:
        List of available tools
    """
    manager = MCPManager()
    
    if config_path:
        manager.load_config_from_file(config_path)
    
    # Add default servers if no config
    if not manager.configs:
        # Example: Add filesystem MCP server
        manager.add_server(MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/"],
            enabled=False  # Disabled by default
        ))
    
    await manager.connect_all()
    tools = manager.get_all_tools()
    await manager.disconnect_all()
    
    return tools


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
        Wrapper function
    """
    async def wrapper(**kwargs) -> Any:
        result = await manager.invoke_tool(server_name, tool_name, kwargs)
        
        if result.success:
            return result.result
        else:
            raise Exception(f"MCP tool error: {result.error_message}")
    
    return wrapper
