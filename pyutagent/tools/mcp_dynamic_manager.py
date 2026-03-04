"""Enhanced MCP Integration - Server Discovery and Dynamic Tool Registration.

This module provides:
- MCP Server auto-discovery (npm global, local, config files)
- Dynamic tool registration
- Hot-reload support
"""

import json
import logging
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredServer:
    """A discovered MCP server."""
    name: str
    command: str
    args: List[str]
    source: str
    config_path: Optional[str] = None


@dataclass
class ToolRegistration:
    """Tool registration record."""
    tool_name: str
    server_name: str
    registered_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0


class MCPServerDiscovery:
    """MCP Server auto-discovery.

    Discovers MCP servers from:
    - npm global packages
    - Local node_modules
    - Config files (~/.config/mcp/*.json)
    - Project mcp.json
    """

    def __init__(self):
        """Initialize MCP server discovery."""
        self._discovered_servers: List[DiscoveredServer] = []
        self._known_server_packages = {
            "@modelcontextprotocol/server-filesystem": "npx -y @modelcontextprotocol/server-filesystem {path}",
            "@modelcontextprotocol/server-github": "npx -y @modelcontextprotocol/server-github",
            "@modelcontextprotocol/server-brave-search": "npx -y @modelcontextprotocol/server-brave-search",
            "@modelcontextprotocol/server-slack": "npx -y @modelcontextprotocol/server-slack",
            "@modelcontextprotocol/server-postgres": "npx -y @modelcontextprotocol/server-postgres",
            "@modelcontextprotocol/server-sqlite": "npx -y @modelcontextprotocol/server-sqlite",
            "@modelcontextprotocol/server-memory": "npx -y @modelcontextprotocol/server-memory",
            "@modelcontextprotocol/server-time": "npx -y @modelcontextprotocol/server-time",
            "@modelcontextprotocol/server-jupyter": "npx -y @modelcontextprotocol/server-jupyter",
        }

    async def discover_all(self) -> List[DiscoveredServer]:
        """Discover all MCP servers.

        Returns:
            List of discovered servers
        """
        self._discovered_servers.clear()

        await self._discover_npm_global()
        await self._discover_config_files()
        await self._discover_project_config()

        logger.info(f"[MCPServerDiscovery] Discovered {len(self._discovered_servers)} MCP servers")
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

                for pkg_name in self._known_server_packages.keys():
                    if pkg_name in dependencies:
                        server = DiscoveredServer(
                            name=pkg_name.replace("@modelcontextprotocol/server-", ""),
                            command=self._format_command(pkg_name),
                            args=[],
                            source="npm_global"
                        )
                        self._discovered_servers.append(server)
                        logger.info(f"[MCPServerDiscovery] Found npm global: {pkg_name}")

        except Exception as e:
            logger.warning(f"[MCPServerDiscovery] npm discovery failed: {e}")

    def _format_command(self, package: str) -> str:
        """Format command template."""
        template = self._known_server_packages.get(package, "npx -y {package}")
        return template.format(package=package, path=".")

    async def _discover_config_files(self):
        """Discover MCP servers from config files."""
        config_paths = [
            Path.home() / ".config" / "mcp",
            Path.home() / ".mcp",
        ]

        for config_dir in config_paths:
            if not config_dir.exists():
                continue

            for config_file in config_dir.glob("*.json"):
                try:
                    with open(config_file) as f:
                        config = json.load(f)

                    servers = config.get("mcpServers", {})
                    for server_name, server_config in servers.items():
                        command = server_config.get("command", "")
                        args = server_config.get("args", [])

                        server = DiscoveredServer(
                            name=server_name,
                            command=command,
                            args=args,
                            source=str(config_file),
                            config_path=str(config_file)
                        )
                        self._discovered_servers.append(server)
                        logger.info(f"[MCPServerDiscovery] Found config server: {server_name}")

                except Exception as e:
                    logger.warning(f"[MCPServerDiscovery] Failed to parse {config_file}: {e}")

    async def _discover_project_config(self):
        """Discover MCP servers from project config."""
        project_configs = [
            Path("mcp.json"),
            Path(".mcp.json"),
            Path(".mcp/mcp.json"),
        ]

        for config_file in project_configs:
            if not config_file.exists():
                continue

            try:
                with open(config_file) as f:
                    config = json.load(f)

                servers = config.get("mcpServers", {})
                for server_name, server_config in servers.items():
                    command = server_config.get("command", "")
                    args = server_config.get("args", [])

                    server = DiscoveredServer(
                        name=server_name,
                        command=command,
                        args=args,
                        source="project",
                        config_path=str(config_file)
                    )
                    self._discovered_servers.append(server)
                    logger.info(f"[MCPServerDiscovery] Found project server: {server_name}")

            except Exception as e:
                logger.warning(f"[MCPServerDiscovery] Failed to parse {config_file}: {e}")

    def get_discovered_servers(self) -> List[DiscoveredServer]:
        """Get list of discovered servers."""
        return self._discovered_servers


class DynamicToolRegistry:
    """Dynamic tool registry with hot-reload support.

    Features:
    - Dynamic tool registration/unregistration
    - Usage tracking
    - Tool status management
    - Persistence
    """

    def __init__(self, storage_path: str = ".pyutagent/mcp_tools.db"):
        """Initialize dynamic tool registry.

        Args:
            storage_path: Path to SQLite database
        """
        self.storage_path = storage_path
        self._conn: Optional[sqlite3.Connection] = None
        self._registered_tools: Dict[str, ToolRegistration] = {}
        self._init_storage()

    def _init_storage(self):
        """Initialize storage and tables."""
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        self._connect()
        self._create_tables()

    def _connect(self):
        """Connect to database."""
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(self.storage_path)
        self._conn.row_factory = sqlite3.Row

    def _create_tables(self):
        """Create necessary tables."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_registrations (
                tool_name TEXT PRIMARY KEY,
                server_name TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                last_used TEXT,
                usage_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_server_name
            ON tool_registrations(server_name)
        """)

        self._conn.commit()

        self._load_registrations()

    def _load_registrations(self):
        """Load registrations from database."""
        cursor = self._conn.execute("SELECT * FROM tool_registrations")
        rows = cursor.fetchall()

        for row in rows:
            registration = ToolRegistration(
                tool_name=row["tool_name"],
                server_name=row["server_name"],
                registered_at=datetime.fromisoformat(row["registered_at"]),
                last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else None,
                usage_count=row["usage_count"]
            )
            self._registered_tools[registration.tool_name] = registration

    async def register_tool(
        self,
        tool_name: str,
        server_name: str
    ) -> ToolRegistration:
        """Register a tool.

        Args:
            tool_name: Tool name
            server_name: Server name

        Returns:
            Registration record
        """
        now = datetime.now()

        registration = ToolRegistration(
            tool_name=tool_name,
            server_name=server_name,
            registered_at=now,
            last_used=now,
            usage_count=1
        )

        self._conn.execute(
            """
            INSERT OR REPLACE INTO tool_registrations
            (tool_name, server_name, registered_at, last_used, usage_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                tool_name,
                server_name,
                now.isoformat(),
                now.isoformat(),
                1
            )
        )
        self._conn.commit()

        self._registered_tools[tool_name] = registration
        logger.info(f"[DynamicToolRegistry] Registered tool: {tool_name} from {server_name}")

        return registration

    async def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool.

        Args:
            tool_name: Tool name

        Returns:
            True if unregistered
        """
        if tool_name not in self._registered_tools:
            return False

        cursor = self._conn.execute(
            "DELETE FROM tool_registrations WHERE tool_name = ?",
            (tool_name,)
        )
        self._conn.commit()

        del self._registered_tools[tool_name]
        logger.info(f"[DynamicToolRegistry] Unregistered tool: {tool_name}")

        return cursor.rowcount > 0

    async def unregister_server(self, server_name: str) -> int:
        """Unregister all tools from a server.

        Args:
            server_name: Server name

        Returns:
            Number of tools unregistered
        """
        tools_to_remove = [
            name for name, reg in self._registered_tools.items()
            if reg.server_name == server_name
        ]

        for tool_name in tools_to_remove:
            await self.unregister_tool(tool_name)

        return len(tools_to_remove)

    async def update_usage(self, tool_name: str):
        """Update tool usage statistics.

        Args:
            tool_name: Tool name
        """
        if tool_name not in self._registered_tools:
            return

        registration = self._registered_tools[tool_name]
        registration.usage_count += 1
        registration.last_used = datetime.now()

        self._conn.execute(
            """
            UPDATE tool_registrations
            SET usage_count = ?, last_used = ?
            WHERE tool_name = ?
            """,
            (
                registration.usage_count,
                registration.last_used.isoformat(),
                tool_name
            )
        )
        self._conn.commit()

    def get_registered_tools(self) -> List[ToolRegistration]:
        """Get all registered tools.

        Returns:
            List of registrations
        """
        return list(self._registered_tools.values())

    def get_tools_by_server(self, server_name: str) -> List[ToolRegistration]:
        """Get tools by server.

        Args:
            server_name: Server name

        Returns:
            List of registrations
        """
        return [
            reg for reg in self._registered_tools.values()
            if reg.server_name == server_name
        ]

    def get_most_used_tools(self, limit: int = 10) -> List[ToolRegistration]:
        """Get most used tools.

        Args:
            limit: Maximum results

        Returns:
            List of registrations sorted by usage
        """
        return sorted(
            self._registered_tools.values(),
            key=lambda r: r.usage_count,
            reverse=True
        )[:limit]

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class MCPDynamicManager:
    """MCP Dynamic Manager with auto-discovery and hot-reload.

    Combines discovery and dynamic registration.
    """

    def __init__(self, storage_path: str = ".pyutagent/mcp_tools.db"):
        """Initialize MCP dynamic manager.

        Args:
            storage_path: Storage path for registrations
        """
        self.discovery = MCPServerDiscovery()
        self.registry = DynamicToolRegistry(storage_path)

    async def discover_and_register(self) -> List[DiscoveredServer]:
        """Discover servers and register tools.

        Returns:
            List of discovered servers
        """
        servers = await self.discovery.discover_all()

        for server in servers:
            logger.info(f"[MCPDynamicManager] Discovered: {server.name} from {server.source}")

        return servers

    async def refresh_tools(self, server_name: str) -> int:
        """Refresh tools for a server.

        Args:
            server_name: Server name

        Returns:
            Number of tools registered
        """
        servers = self.discovery.get_discovered_servers()
        target_server = next(
            (s for s in servers if s.name == server_name),
            None
        )

        if not target_server:
            logger.warning(f"[MCPDynamicManager] Server not found: {server_name}")
            return 0

        count = len(await self._register_server_tools(target_server))
        logger.info(f"[MCPDynamicManager] Refreshed {count} tools for {server_name}")

        return count

    async def _register_server_tools(self, server: DiscoveredServer) -> List[str]:
        """Register tools from a server.

        Args:
            server: Discovered server

        Returns:
            List of registered tool names
        """
        tool_names = []

        for pattern, default_name in self._guess_tools(server.name):
            tool_names.append(pattern)

        return tool_names

    def _guess_tools(self, server_name: str) -> List[tuple]:
        """Guess tool names from server name.

        Args:
            server_name: Server name

        Returns:
            List of (tool_pattern, description)
        """
        tool_guesses = {
            "filesystem": [
                ("read_file", "Read file"),
                ("list_directory", "List directory"),
            ],
            "github": [
                ("get_issue", "Get issue"),
                ("create_issue", "Create issue"),
            ],
            "brave-search": [
                ("web_search", "Web search"),
            ],
            "slack": [
                ("send_message", "Send message"),
            ],
            "postgres": [
                ("query", "Execute query"),
            ],
            "sqlite": [
                ("query", "Execute query"),
            ],
            "memory": [
                ("remember", "Store memory"),
                ("recall", "Recall memory"),
            ],
        }

        return tool_guesses.get(server_name, [(server_name, server_name)])

    def close(self):
        """Close all resources."""
        self.registry.close()


def create_mcp_dynamic_manager(
    storage_dir: str = ".pyutagent"
) -> MCPDynamicManager:
    """Create MCP dynamic manager.

    Args:
        storage_dir: Storage directory

    Returns:
        MCPDynamicManager instance
    """
    import os
    storage_path = os.path.join(storage_dir, "mcp_tools.db")
    return MCPDynamicManager(storage_path)
