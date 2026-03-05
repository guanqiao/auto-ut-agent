"""MCP (Model Context Protocol) 演示脚本

演示如何使用 MCP 客户端连接服务器、发现工具并调用工具。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from typing import Dict, Any

from pyutagent.tools import (
    MCPManager,
    EnhancedMCPManager,
    MCPServerConfig,
    MCPTransportType,
    MCPConfigLoader,
    create_enhanced_mcp_manager,
)


def demo_mcp_config():
    """演示 MCP 服务器配置"""
    print("\n" + "="*60)
    print("演示 1: MCP 服务器配置")
    print("="*60)
    
    # 创建文件系统 MCP 服务器配置
    filesystem_config = MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        transport=MCPTransportType.STDIO,
        timeout=30,
        enabled=True
    )
    
    print(f"\n服务器名称: {filesystem_config.name}")
    print(f"命令: {filesystem_config.command}")
    print(f"参数: {filesystem_config.args}")
    print(f"传输类型: {filesystem_config.transport.value}")
    print(f"超时: {filesystem_config.timeout}秒")
    print(f"启用状态: {filesystem_config.enabled}")
    
    # 创建多个服务器配置
    configs = [
        MCPServerConfig(
            name="fetch",
            command="uvx",
            args=["mcp-server-fetch"],
            enabled=False  # 默认禁用
        ),
        MCPServerConfig(
            name="sqlite",
            command="uvx",
            args=["mcp-server-sqlite", "--db-path", "/tmp/test.db"],
            enabled=False
        ),
    ]
    
    print(f"\n已创建 {len(configs)} 个额外配置")
    for config in configs:
        status = "✅ 启用" if config.enabled else "⏸️  禁用"
        print(f"  - {config.name}: {status}")


def demo_mcp_manager():
    """演示 MCP 管理器"""
    print("\n" + "="*60)
    print("演示 2: MCP 管理器")
    print("="*60)
    
    manager = MCPManager()
    
    # 添加服务器配置
    manager.add_server(MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        enabled=False  # 演示时不实际连接
    ))
    
    manager.add_server(MCPServerConfig(
        name="fetch",
        command="uvx",
        args=["mcp-server-fetch"],
        enabled=False
    ))
    
    print(f"\n已添加 {len(manager.configs)} 个服务器配置")
    
    # 显示服务器状态
    status = manager.get_server_status()
    print("\n服务器状态:")
    for name, info in status.items():
        enabled = "✅" if info["enabled"] else "⏸️"
        connected = "🟢" if info["connected"] else "🔴"
        print(f"  {enabled} {name}: {connected} 连接, {info['tool_count']} 个工具")


def demo_config_loader():
    """演示配置加载器"""
    print("\n" + "="*60)
    print("演示 3: MCP 配置加载器")
    print("="*60)
    
    loader = MCPConfigLoader()
    
    # 从环境变量加载配置示例
    print("\n环境变量配置格式示例:")
    print("  MCP_SERVER_FILESYSTEM_COMMAND=npx")
    print("  MCP_SERVER_FILESYSTEM_ARGS=-y @modelcontextprotocol/server-filesystem /tmp")
    print("  MCP_SERVER_FILESYSTEM_ENABLED=true")
    
    # JSON 配置示例
    json_config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "transport": "stdio",
                "enabled": True
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"],
                "transport": "stdio",
                "enabled": False
            }
        }
    }
    
    print("\nJSON 配置示例:")
    print(json.dumps(json_config, indent=2, ensure_ascii=False))


async def demo_enhanced_mcp_manager():
    """演示增强型 MCP 管理器"""
    print("\n" + "="*60)
    print("演示 4: 增强型 MCP 管理器")
    print("="*60)
    
    manager = EnhancedMCPManager()
    
    # 添加配置路径
    manager.add_config_path("~/.mcp/config.json")
    manager.add_config_path("./.mcp.json")
    
    print(f"\n已添加 {len(manager._config_paths)} 个配置路径")
    for path in manager._config_paths:
        print(f"  - {path}")
    
    # 显示自动发现功能
    print("\n自动发现功能:")
    print("  搜索位置:")
    print("    - 环境变量: MCP_CONFIG_PATH")
    print("    - ~/.mcp/config.json")
    print("    - ./.mcp.json")
    print("    - ./mcp-config.json")
    
    # 模拟工具缓存
    mock_tools = [
        {"name": "mcp_filesystem_read", "description": "读取文件内容"},
        {"name": "mcp_filesystem_write", "description": "写入文件"},
        {"name": "mcp_filesystem_list", "description": "列出目录"},
    ]
    
    for tool in mock_tools:
        manager._tool_cache[tool["name"]] = [tool]
    
    print(f"\n已缓存 {len(manager.get_cached_tools())} 个工具:")
    for tool_name in manager.get_cached_tools():
        print(f"  - {tool_name}")
    
    # 按能力筛选工具
    filesystem_tools = manager.get_tools_by_capability("filesystem")
    print(f"\n文件系统相关工具: {len(filesystem_tools)} 个")


def demo_mcp_tool_structure():
    """演示 MCP 工具结构"""
    print("\n" + "="*60)
    print("演示 5: MCP 工具结构")
    print("="*60)
    
    from pyutagent.tools import MCPTool, MCPToolParameter
    
    # 创建一个 MCP 工具定义
    read_file_tool = MCPTool(
        name="read_file",
        description="读取文件内容",
        parameters=[
            MCPToolParameter(
                name="path",
                param_type="string",
                description="文件路径",
                required=True
            ),
            MCPToolParameter(
                name="encoding",
                param_type="string",
                description="文件编码",
                required=False,
                default="utf-8"
            ),
        ]
    )
    
    print(f"\n工具名称: {read_file_tool.name}")
    print(f"描述: {read_file_tool.description}")
    print(f"\n参数列表:")
    for param in read_file_tool.parameters:
        req = "(必填)" if param.required else "(可选)"
        default = f", 默认值: {param.default}" if param.default else ""
        print(f"  - {param.name}: {param.param_type} {req}{default}")
        print(f"    描述: {param.description}")
    
    # 转换为字典格式
    tool_dict = read_file_tool.to_dict()
    print(f"\n工具定义 (JSON):")
    print(json.dumps(tool_dict, indent=2, ensure_ascii=False))


def demo_integration_with_agent():
    """演示与 Agent 的集成"""
    print("\n" + "="*60)
    print("演示 6: 与 Agent 集成")
    print("="*60)
    
    print("""
MCP 与 Agent 集成方式:

1. 工具注册
   ```python
   from pyutagent.tools import MCPManager
   
   mcp_manager = MCPManager()
   mcp_manager.add_server(config)
   await mcp_manager.connect_all()
   
   # 获取所有 MCP 工具
   mcp_tools = mcp_manager.get_all_tools()
   
   # 注册到工具注册表
   for tool in mcp_tools:
       tool_registry.register(tool)
   ```

2. 在 Agent 中使用
   ```python
   from pyutagent.agent import UniversalCodingAgent
   
   agent = UniversalCodingAgent()
   
   # 启用 MCP 工具
   agent.enable_mcp_tools([
       "filesystem",
       "fetch",
       "sqlite"
   ])
   
   # Agent 可以自动使用 MCP 工具
   result = await agent.execute("读取 /tmp/config.json 文件")
   ```

3. Skills 中使用 MCP
   ```python
   from pyutagent.agent.skills import Skill
   
   class FileSearchSkill(Skill):
       required_tools = ["mcp_filesystem_glob"]
       
       async def execute(self, query):
           # 使用 MCP 工具
           result = await self.use_tool("mcp_filesystem_glob", {
               "pattern": query
           })
           return result
   ```
""")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("MCP (Model Context Protocol) 演示")
    print("模型上下文协议 - 扩展工具生态")
    print("="*60)
    
    demo_mcp_config()
    demo_mcp_manager()
    demo_config_loader()
    await demo_enhanced_mcp_manager()
    demo_mcp_tool_structure()
    demo_integration_with_agent()
    
    print("\n" + "="*60)
    print("MCP 演示完成!")
    print("="*60)
    print("""
下一步:
1. 安装 MCP 服务器: npx -y @modelcontextprotocol/server-filesystem /tmp
2. 配置 MCP: 创建 ~/.mcp/config.json
3. 在 Agent 中启用 MCP 工具

参考文档: https://modelcontextprotocol.io/
""")


if __name__ == "__main__":
    asyncio.run(main())
