"""Tools for PyUT Agent."""

from .java_parser import JavaCodeParser, JavaClass, JavaMethod
from .maven_tools import (
    MavenRunner,
    CoverageAnalyzer,
    ProjectScanner,
    CoverageReport,
    FileCoverage
)
from .aider_integration import (
    AiderCodeFixer,
    AiderTestGenerator,
    FixResult,
    apply_diff_edit,
)
from .code_editor import CodeEditor
from .edit_formats import EditFormat, EditFormatRegistry
from .edit_validator import EditValidator
from .error_analyzer import CompilationErrorAnalyzer, ErrorAnalysis
from .failure_analyzer import TestFailureAnalyzer, FailureAnalysis
from .architect_editor import ArchitectMode
from .tool import (
    Tool,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolExecutor,
    create_tool_parameter,
)
from .tool_registry import (
    ToolRegistry,
    RegistryConfig,
    get_registry,
    set_registry,
    register_tool,
    get_tool,
    list_tools,
    ToolNotFoundError,
)
from .standard_tools import (
    ReadTool,
    WriteTool,
    EditTool,
    GlobTool,
    GrepTool,
    BashTool,
)
from .tool_use import (
    ToolUseAgent,
    ToolUseState,
    ToolCall,
    ToolUseTurn,
    create_tool_use_agent,
)
from .git_tools import (
    GitStatusTool,
    GitDiffTool,
    GitCommitTool,
    GitBranchTool,
    GitLogTool,
)
from .search_tools import (
    WebSearchTool,
    WebFetchTool,
    get_all_search_tools,
)
from .analysis_tools import (
    CodeStructureTool,
    DependencyGraphTool,
    get_all_analysis_tools,
)
from .tool_cache import (
    ToolResultCache,
    CacheEntry,
    CachedToolExecutor,
    create_result_cache,
)
# MCP Integration - Full Protocol Support
from .mcp_integration import (
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
    # Clients
    MCPClient,
    StdioMCPClient,
    HttpMCPClient,
    WebSocketMCPClient,
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
    discover_mcp_tools,
    create_mcp_tool_wrapper,
    create_mcp_manager_with_presets,
)
from .enhanced_mcp import (
    EnhancedMCPManager,
    MCPConfigLoader,
    create_enhanced_mcp_manager,
)
from .remote_mcp import (
    RemoteTransportType,
    RemoteMCPServerConfig,
    ConnectionStatus,
    RemoteMCPClient,
    RemoteMCPManager,
    create_remote_mcp_manager,
)
from .mcp_dynamic_manager import (
    MCPDynamicManager,
    DynamicToolRegistry,
    ToolRegistration,
    create_mcp_dynamic_manager,
)
from .multi_file_editor import (
    FileNode,
    MultiFileEditResult,
    DependencyAnalyzer,
    MultiFileEditor,
)

__all__ = [
    # Java Tools
    "JavaCodeParser",
    "JavaClass",
    "JavaMethod",
    # Maven Tools
    "MavenRunner",
    "CoverageAnalyzer",
    "ProjectScanner",
    "CoverageReport",
    "FileCoverage",
    # Aider Integration
    "AiderCodeFixer",
    "AiderTestGenerator",
    "FixResult",
    "apply_diff_edit",
    # Code Editor
    "CodeEditor",
    "EditFormat",
    "EditFormatRegistry",
    "EditValidator",
    # Error Analysis
    "CompilationErrorAnalyzer",
    "ErrorAnalysis",
    "TestFailureAnalyzer",
    "FailureAnalysis",
    # Architect Mode
    "ArchitectMode",
    # Base Tool System
    "Tool",
    "ToolCategory",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    "ToolExecutor",
    "create_tool_parameter",
    # Tool Registry
    "ToolRegistry",
    "RegistryConfig",
    "get_registry",
    "set_registry",
    "register_tool",
    "get_tool",
    "list_tools",
    "ToolNotFoundError",
    # Standard Tools
    "ReadTool",
    "WriteTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "BashTool",
    # Tool Use
    "ToolUseAgent",
    "ToolUseState",
    "ToolCall",
    "ToolUseTurn",
    "create_tool_use_agent",
    # Git Tools
    "GitStatusTool",
    "GitDiffTool",
    "GitCommitTool",
    "GitBranchTool",
    "GitLogTool",
    # Search Tools
    "WebSearchTool",
    "WebFetchTool",
    "get_all_search_tools",
    # Analysis Tools
    "CodeStructureTool",
    "DependencyGraphTool",
    "get_all_analysis_tools",
    # Tool Cache
    "ToolResultCache",
    "CacheEntry",
    "CachedToolExecutor",
    "create_result_cache",
    # MCP Integration - Full Protocol Support
    "MCPTransportType",
    "MCPErrorCode",
    "MCPError",
    "MCPRequest",
    "MCPResponse",
    "MCPToolParameter",
    "MCPTool",
    "MCPToolResult",
    "MCPToolError",
    "MCPServerConfig",
    "ServerStatus",
    "MCPClient",
    "StdioMCPClient",
    "HttpMCPClient",
    "WebSocketMCPClient",
    "MCPToolAdapter",
    "AdaptedTool",
    "MCPManager",
    "MCPServerDiscovery",
    "DiscoveredServer",
    "MCPServerPresets",
    "discover_mcp_tools",
    "create_mcp_tool_wrapper",
    "create_mcp_manager_with_presets",
    # Enhanced MCP
    "EnhancedMCPManager",
    "MCPConfigLoader",
    "create_enhanced_mcp_manager",
    # Remote MCP
    "RemoteTransportType",
    "RemoteMCPServerConfig",
    "ConnectionStatus",
    "RemoteMCPClient",
    "RemoteMCPManager",
    "create_remote_mcp_manager",
    # MCP Dynamic Manager
    "MCPDynamicManager",
    "DynamicToolRegistry",
    "ToolRegistration",
    "create_mcp_dynamic_manager",
    # Multi-file Editor
    "FileNode",
    "MultiFileEditResult",
    "DependencyAnalyzer",
    "MultiFileEditor",
]
