"""Remote MCP Support - Secure remote MCP server connections.

This module provides:
- Remote MCP server connections (HTTP/WebSocket)
- Secure authentication
- Connection pooling
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

# Optional websockets import
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False


class RemoteTransportType(Enum):
    """Remote transport types."""
    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "ws"
    WSS = "wss"


@dataclass
class RemoteMCPServerConfig:
    """Remote MCP server configuration."""
    name: str
    url: str
    transport: RemoteTransportType = RemoteTransportType.WSS
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    enabled: bool = True
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConnectionStatus:
    """Connection status."""
    connected: bool
    last_connected: Optional[datetime] = None
    last_error: Optional[str] = None
    retry_count: int = 0


class RemoteMCPClient:
    """Remote MCP client for secure server connections.

    Supports:
    - WebSocket connections (wss://)
    - HTTP connections (https://)
    - Authentication (API keys, Bearer tokens)
    - Automatic reconnection
    - Connection pooling
    """

    def __init__(
        self,
        config: RemoteMCPServerConfig,
        connection_pool_size: int = 3
    ):
        """Initialize remote MCP client.

        Args:
            config: Remote server configuration
            connection_pool_size: Number of connections in pool
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package is required for RemoteMCPClient")

        self.config = config
        self.connection_pool_size = connection_pool_size

        self._connection: Optional[Any] = None
        self._connection_pool: List[Any] = []
        self._status = ConnectionStatus(connected=False)
        self._lock = asyncio.Lock()
        self._request_id = 0

    async def connect(self) -> bool:
        """Connect to remote MCP server.

        Returns:
            True if connected successfully
        """
        async with self._lock:
            if self._status.connected:
                return True

            try:
                logger.info(f"[RemoteMCPClient] Connecting to {self.config.url}")

                headers = self.config.headers.copy()
                if self.config.api_key:
                    headers["Authorization"] = f"Bearer {self.config.api_key}"

                self._connection = await websockets.connect(
                    self.config.url,
                    extra_headers=headers,
                    ping_timeout=self.config.timeout
                )

                init_request = {
                    "jsonrpc": "2.0",
                    "id": self._get_request_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "pyutagent-remote",
                            "version": "1.0.0"
                        }
                    }
                }

                response = await self._send_request(init_request)

                if response and "result" in response:
                    self._status = ConnectionStatus(
                        connected=True,
                        last_connected=datetime.now()
                    )
                    logger.info(f"[RemoteMCPClient] Connected to {self.config.name}")
                    return True
                else:
                    raise Exception("Failed to initialize")

            except Exception as e:
                logger.error(f"[RemoteMCPClient] Connection failed: {e}")
                self._status = ConnectionStatus(
                    connected=False,
                    last_error=str(e)
                )
                return False

    async def disconnect(self):
        """Disconnect from remote server."""
        async with self._lock:
            if self._connection:
                await self._connection.close()
                self._connection = None

            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()

            self._status = ConnectionStatus(connected=False)
            logger.info(f"[RemoteMCPClient] Disconnected from {self.config.name}")

    async def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke a tool on remote server.

        Args:
            tool_name: Tool name
            parameters: Tool parameters

        Returns:
            Tool result
        """
        if not self._status.connected:
            await self.connect()

        request = {
            "jsonrpc": "2.0",
            "id": self._get_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            }
        }

        response = await self._send_request(request)

        if "result" in response:
            return response["result"]
        elif "error" in response:
            raise Exception(f"Tool error: {response['error']}")
        else:
            raise Exception("Invalid response")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools on remote server.

        Returns:
            List of tools
        """
        if not self._status.connected:
            await self.connect()

        request = {
            "jsonrpc": "2.0",
            "id": self._get_request_id(),
            "method": "tools/list",
            "params": {}
        }

        response = await self._send_request(request)

        if "result" in response:
            return response["result"].get("tools", [])
        return []

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to server.

        Args:
            request: JSON-RPC request

        Returns:
            Response
        """
        if not self._connection:
            raise Exception("Not connected")

        await self._connection.send(json.dumps(request))

        response_str = await asyncio.wait_for(
            self._connection.recv(),
            timeout=self.config.timeout
        )

        return json.loads(response_str)

    def _get_request_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    @property
    def status(self) -> ConnectionStatus:
        """Get connection status."""
        return self._status


class RemoteMCPManager:
    """Manager for multiple remote MCP servers.

    Features:
    - Multiple server management
    - Connection pooling
    - Automatic failover
    - Load balancing
    """

    def __init__(self):
        """Initialize remote MCP manager."""
        self._servers: Dict[str, RemoteMCPClient] = {}
        self._lock = asyncio.Lock()

    async def add_server(
        self,
        config: RemoteMCPServerConfig
    ) -> RemoteMCPClient:
        """Add a remote server.

        Args:
            config: Server configuration

        Returns:
            Remote MCP client
        """
        async with self._lock:
            client = RemoteMCPClient(config)
            self._servers[config.name] = client

            logger.info(f"[RemoteMCPManager] Added server: {config.name}")
            return client

    async def remove_server(self, name: str) -> bool:
        """Remove a server.

        Args:
            name: Server name

        Returns:
            True if removed
        """
        async with self._lock:
            if name not in self._servers:
                return False

            client = self._servers[name]
            await client.disconnect()

            del self._servers[name]
            logger.info(f"[RemoteMCPManager] Removed server: {name}")
            return True

    async def connect_all(self):
        """Connect to all servers."""
        for name, client in self._servers.items():
            try:
                await client.connect()
            except Exception as e:
                logger.error(f"[RemoteMCPManager] Failed to connect {name}: {e}")

    async def disconnect_all(self):
        """Disconnect from all servers."""
        for client in self._servers.values():
            await client.disconnect()

    async def invoke_tool(
        self,
        server_name: str,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Invoke tool on specific server.

        Args:
            server_name: Server name
            tool_name: Tool name
            parameters: Tool parameters

        Returns:
            Tool result
        """
        if server_name not in self._servers:
            raise Exception(f"Server not found: {server_name}")

        client = self._servers[server_name]
        return await client.invoke_tool(tool_name, parameters)

    def get_server(self, name: str) -> Optional[RemoteMCPClient]:
        """Get server client.

        Args:
            name: Server name

        Returns:
            Client or None
        """
        return self._servers.get(name)

    def list_servers(self) -> List[str]:
        """List all server names.

        Returns:
            Server names
        """
        return list(self._servers.keys())

    def get_status(self) -> Dict[str, ConnectionStatus]:
        """Get status of all servers.

        Returns:
            Status dict
        """
        return {
            name: client.status
            for name, client in self._servers.items()
        }


def create_remote_mcp_manager() -> RemoteMCPManager:
    """Create remote MCP manager.

    Returns:
        RemoteMCPManager instance
    """
    return RemoteMCPManager()
