"""Enhanced MCP Integration - Auto-discovery and dynamic tool loading.

This module extends the existing MCP integration with:
- Auto-discovery of MCP servers
- Dynamic tool loading
- MCP tool adapter improvements
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..tools.mcp_integration import (
    MCPManager,
    MCPServerConfig,
    MCPTransportType,
    MCPTool,
    MCPToolAdapter,
)

logger = logging.getLogger(__name__)


class EnhancedMCPManager(MCPManager):
    """Enhanced MCP Manager with auto-discovery and dynamic loading.
    
    Features:
    - Auto-discover MCP servers from environment
    - Load MCP tools dynamically
    - Config file support
    - Tool caching
    """
    
    def __init__(self):
        """Initialize enhanced MCP manager."""
        super().__init__()
        self._tool_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._config_paths: List[str] = []
    
    def add_config_path(self, config_path: str):
        """Add a configuration file path.
        
        Args:
            config_path: Path to MCP config file
        """
        if config_path not in self._config_paths:
            self._config_paths.append(config_path)
            logger.info(f"[EnhancedMCPManager] Added config path: {config_path}")
    
    async def auto_discover(self) -> Dict[str, bool]:
        """Auto-discover MCP servers from common locations.
        
        Searches in:
        - Environment variables (MCP_CONFIG_PATH)
        - ~/.mcp/config.json
        - ./.mcp.json
        - Project-specific configs
        
        Returns:
            Dictionary of discovered servers and connection status
        """
        discovered = {}
        
        env_config = os.environ.get("MCP_CONFIG_PATH")
        if env_config:
            self.add_config_path(env_config)
        
        home_mcp = Path.home() / ".mcp" / "config.json"
        if home_mcp.exists():
            self.add_config_path(str(home_mcp))
        
        local_mcp = Path.cwd() / ".mcp.json"
        if local_mcp.exists():
            self.add_config_path(str(local_mcp))
        
        project_mcp = Path.cwd() / "mcp-config.json"
        if project_mcp.exists():
            self.add_config_path(str(project_mcp))
        
        for config_path in self._config_paths:
            try:
                self.load_config_from_file(config_path)
                logger.info(f"[EnhancedMCPManager] Loaded config from: {config_path}")
            except Exception as e:
                logger.warning(f"[EnhancedMCPManager] Failed to load {config_path}: {e}")
        
        if self.configs:
            return await self.connect_all()
        
        return discovered
    
    async def discover_and_load_tools(self) -> List[Dict[str, Any]]:
        """Discover MCP servers and load all available tools.
        
        Returns:
            List of loaded tools
        """
        await self.auto_discover()
        
        all_tools = self.get_all_tools()
        
        for tool in all_tools:
            tool_name = tool.get("name", "")
            if tool_name:
                if tool_name not in self._tool_cache:
                    self._tool_cache[tool_name] = []
                self._tool_cache[tool_name].append(tool)
        
        logger.info(f"[EnhancedMCPManager] Loaded {len(all_tools)} MCP tools")
        return all_tools
    
    def get_tools_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get tools filtered by capability.
        
        Args:
            capability: Capability to filter by (e.g., "filesystem", "web")
        
        Returns:
            List of tools with matching capability
        """
        results = []
        
        for name, tools in self._tool_cache.items():
            if capability.lower() in name.lower():
                results.extend(tools)
        
        return results
    
    def get_cached_tools(self) -> List[str]:
        """Get list of cached tool names.
        
        Returns:
            List of cached tool names
        """
        return list(self._tool_cache.keys())
    
    async def refresh_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """Refresh tools for a specific server.
        
        Args:
            server_name: Name of the server
        
        Returns:
            List of refreshed tools
        """
        if server_name in self.clients:
            client = self.clients[server_name]
            await client.disconnect()
        
        success = await self.connect_all()
        
        if server_name in success and success[server_name]:
            return self.get_all_tools()
        
        return []
    
    def get_server_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """Get tools from a specific server.
        
        Args:
            server_name: Name of the MCP server
        
        Returns:
            List of tools from the server
        """
        if server_name not in self.clients:
            return []
        
        client = self.clients[server_name]
        adapter = MCPToolAdapter(client)
        
        tools = []
        for mcp_tool in client.available_tools:
            adapted = adapter.adapt_tool(mcp_tool)
            tools.append(adapted)
        
        return tools
    
    def create_tool_wrapper(self, server_name: str, tool_name: str):
        """Create a wrapper function for an MCP tool.
        
        Args:
            server_name: MCP server name
            tool_name: Tool name
        
        Returns:
            Async wrapper function
        """
        async def wrapper(**kwargs) -> Any:
            result = await self.invoke_tool(server_name, tool_name, kwargs)
            
            if result.success:
                return result.result
            else:
                raise Exception(f"MCP tool error: {result.error_message}")
        
        return wrapper
    
    async def close(self):
        """Clean up resources."""
        await self.disconnect_all()
        self._tool_cache.clear()
        logger.info("[EnhancedMCPManager] Closed")


class MCPConfigLoader:
    """Loader for MCP configurations from various sources."""
    
    @staticmethod
    def from_env() -> Dict[str, MCPServerConfig]:
        """Load MCP config from environment variables.
        
        Environment variables:
        - MCP_SERVER_{name}_COMMAND
        - MCP_SERVER_{name}_ARGS
        - MCP_SERVER_{name}_ENABLED
        
        Returns:
            Dictionary of server configurations
        """
        configs = {}
        
        for key, value in os.environ.items():
            if key.startswith("MCP_SERVER_") and key.endswith("_COMMAND"):
                name = key[12:-8]
                
                command = value
                args_str = os.environ.get(f"MCP_SERVER_{name}_ARGS", "")
                args = args_str.split() if args_str else []
                enabled = os.environ.get(f"MCP_SERVER_{name}_ENABLED", "true").lower() == "true"
                
                config = MCPServerConfig(
                    name=name,
                    command=command,
                    args=args,
                    enabled=enabled
                )
                
                configs[name] = config
                logger.info(f"[MCPConfigLoader] Loaded from env: {name}")
        
        return configs
    
    @staticmethod
    def from_json_file(file_path: str) -> Dict[str, MCPServerConfig]:
        """Load MCP config from JSON file.
        
        Args:
            file_path: Path to JSON config file
        
        Returns:
            Dictionary of server configurations
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        configs = {}
        
        for name, server_data in data.get("mcpServers", {}).items():
            config = MCPServerConfig(
                name=name,
                command=server_data["command"],
                args=server_data.get("args", []),
                env=server_data.get("env", {}),
                transport=MCPTransportType(server_data.get("transport", "stdio")),
                timeout=server_data.get("timeout", 30),
                enabled=server_data.get("enabled", True)
            )
            configs[name] = config
        
        return configs
    
    @staticmethod
    def from_yaml_file(file_path: str) -> Dict[str, MCPServerConfig]:
        """Load MCP config from YAML file.
        
        Args:
            file_path: Path to YAML config file
        
        Returns:
            Dictionary of server configurations
        """
        try:
            import yaml
        except ImportError:
            logger.warning("[MCPConfigLoader] PyYAML not installed, cannot load YAML")
            return {}
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        configs = {}
        
        for name, server_data in data.get("mcpServers", {}).items():
            config = MCPServerConfig(
                name=name,
                command=server_data["command"],
                args=server_data.get("args", []),
                env=server_data.get("env", {}),
                transport=MCPTransportType(server_data.get("transport", "stdio")),
                timeout=server_data.get("timeout", 30),
                enabled=server_data.get("enabled", True)
            )
            configs[name] = config
        
        return configs


async def create_enhanced_mcp_manager(
    config_paths: Optional[List[str]] = None,
    auto_discover: bool = True
) -> EnhancedMCPManager:
    """Create an enhanced MCP manager.
    
    Args:
        config_paths: Optional list of config file paths
        auto_discover: Whether to auto-discover servers
    
    Returns:
        Configured EnhancedMCPManager
    """
    manager = EnhancedMCPManager()
    
    if config_paths:
        for path in config_paths:
            manager.add_config_path(path)
    
    if auto_discover:
        await manager.auto_discover()
        await manager.connect_all()
    
    return manager
