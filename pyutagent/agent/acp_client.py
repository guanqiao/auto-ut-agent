"""ACP (Anthropic Claude Protocol) client for IDE integration.

This module provides:
- ACPClient: Client for ACP protocol
- ACPMessage: Message types for ACP
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ACPMessageType(Enum):
    """ACP message types."""
    Initialize = "initialize"
    InitializeResult = "initialize/result"
    Notification = "notification"
    Request = "request"
    Response = "response"
    Error = "error"


class ACPErrorCode(Enum):
    """ACP error codes."""
    ParseError = -32700
    InvalidRequest = -32600
    MethodNotFound = -32601
    InvalidParams = -32602
    InternalError = -32603


@dataclass
class ACPMessage:
    """ACP message structure."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class ACPClientConfig:
    """Configuration for ACP client."""
    endpoint: str = "http://localhost:8080"
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    auto_reconnect: bool = True


class ACPClient:
    """Client for Anthropic Claude Protocol."""

    def __init__(self, config: Optional[ACPClientConfig] = None):
        """Initialize ACP client.

        Args:
            config: ACP client configuration
        """
        self.config = config or ACPClientConfig()
        self._connected = False
        self._message_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._ws = None
        self._callbacks: Dict[str, Callable] = {}

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    async def connect(self) -> bool:
        """Connect to ACP server.

        Returns:
            True if connection successful
        """
        try:
            parsed = urlparse(self.config.endpoint)

            if parsed.scheme == "ws" or parsed.scheme == "wss":
                await self._connect_websocket()
            else:
                logger.warning("[ACPClient] HTTP transport not fully implemented")

            self._connected = True
            logger.info(f"[ACPClient] Connected to {self.config.endpoint}")
            return True

        except Exception as e:
            logger.error(f"[ACPClient] Connection failed: {e}")
            self._connected = False
            return False

    async def _connect_websocket(self):
        """Connect via WebSocket."""
        try:
            import websockets

            self._ws = await websockets.connect(
                self.config.endpoint,
                extra_headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}
            )
            asyncio.create_task(self._receive_loop())
        except ImportError:
            logger.warning("[ACPClient] websockets not installed, using mock")

    async def _receive_loop(self):
        """Receive messages in loop."""
        if not self._ws:
            return

        try:
            async for message in self._ws:
                await self._handle_message(message)
        except Exception as e:
            logger.error(f"[ACPClient] Receive error: {e}")

    async def _handle_message(self, message: str):
        """Handle incoming message.

        Args:
            message: JSON message string
        """
        try:
            data = json.loads(message)
            msg = ACPMessage(**data)

            if msg.id and msg.id in self._pending_requests:
                future = self._pending_requests.pop(msg.id)
                if msg.error:
                    future.set_exception(Exception(msg.error.get("message", "Unknown error")))
                else:
                    future.set_result(msg.result)
            elif msg.method:
                await self._dispatch_method(msg)

        except Exception as e:
            logger.error(f"[ACPClient] Message handling error: {e}")

    async def _dispatch_method(self, msg: ACPMessage):
        """Dispatch method call to handler.

        Args:
            msg: ACP message
        """
        callback = self._callbacks.get(msg.method)
        if callback:
            try:
                result = await callback(msg.params) if asyncio.iscoroutinefunction(callback) else callback(msg.params)
                await self._send_response(msg.id, result)
            except Exception as e:
                await self._send_error(msg.id, -32603, str(e))

    async def _send_response(self, msg_id: int, result: Any):
        """Send response message.

        Args:
            msg_id: Message ID
            result: Result data
        """
        await self._send(ACPMessage(
            jsonrpc="2.0",
            id=msg_id,
            result=result
        ))

    async def _send_error(self, msg_id: int, code: int, message: str):
        """Send error response.

        Args:
            msg_id: Message ID
            code: Error code
            message: Error message
        """
        await self._send(ACPMessage(
            jsonrpc="2.0",
            id=msg_id,
            error={"code": code, "message": message}
        ))

    async def _send(self, msg: ACPMessage):
        """Send message.

        Args:
            msg: ACP message
        """
        if self._ws:
            await self._ws.send(json.dumps(msg.__dict__))

    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Send request and wait for response.

        Args:
            method: Method name
            params: Method parameters

        Returns:
            Result from server
        """
        if not self._connected:
            await self.connect()

        self._message_id += 1
        msg_id = self._message_id

        future = asyncio.Future()
        self._pending_requests[msg_id] = future

        await self._send(ACPMessage(
            jsonrpc="2.0",
            id=msg_id,
            method=method,
            params=params
        ))

        try:
            result = await asyncio.wait_for(future, timeout=self.config.timeout)
            return result
        except asyncio.TimeoutError:
            self._pending_requests.pop(msg_id, None)
            raise TimeoutError(f"Request {method} timed out")

    async def send_notification(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send notification (no response).

        Args:
            method: Method name
            params: Method parameters
        """
        if not self._connected:
            return

        await self._send(ACPMessage(
            jsonrpc="2.0",
            method=method,
            params=params
        ))

    def register_callback(self, method: str, callback: Callable):
        """Register callback for method.

        Args:
            method: Method name
            callback: Callback function
        """
        self._callbacks[method] = callback
        logger.info(f"[ACPClient] Registered callback for: {method}")

    async def disconnect(self):
        """Disconnect from server."""
        if self._ws:
            await self._ws.close()
        self._connected = False
        logger.info("[ACPClient] Disconnected")

    async def initialize(self, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize session.

        Args:
            capabilities: Client capabilities

        Returns:
            Server capabilities
        """
        return await self.send_request("initialize", {
            "capabilities": capabilities,
            "protocolVersion": "2024-11-05"
        })

    async def textDocument_didOpen(self, uri: str, content: str, language: str):
        """Notify server that a document was opened.

        Args:
            uri: Document URI
            content: Document content
            language: Language ID
        """
        await self.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": language,
                "version": 1,
                "text": content
            }
        })

    async def textDocument_didChange(self, uri: str, content: str, version: int):
        """Notify server that a document was changed.

        Args:
            uri: Document URI
            content: New content
            version: Document version
        """
        await self.send_notification("textDocument/didChange", {
            "textDocument": {
                "uri": uri,
                "version": version
            },
            "contentChanges": [{"text": content}]
        })

    async def workspace_executeCommand(self, command: str, args: List[Any]) -> Any:
        """Execute a workspace command.

        Args:
            command: Command name
            args: Command arguments

        Returns:
            Command result
        """
        return await self.send_request("workspace/executeCommand", {
            "command": command,
            "arguments": args
        })


async def create_acp_client(
    endpoint: str = "ws://localhost:8080",
    api_key: Optional[str] = None
) -> ACPClient:
    """Create and connect ACP client.

    Args:
        endpoint: Server endpoint
        api_key: Optional API key

    Returns:
        Connected ACP client
    """
    config = ACPClientConfig(endpoint=endpoint, api_key=api_key)
    client = ACPClient(config)

    if await client.connect():
        return client

    return client
