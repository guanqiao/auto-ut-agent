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
from .mcp_integration import (
    MCPTransportType,
    MCPToolParameter,
    MCPTool,
    MCPToolResult,
    MCPServerConfig,
    MCPClient,
    StdioMCPClient,
    MCPToolAdapter,
    MCPManager,
    discover_mcp_tools,
    create_mcp_tool_wrapper,
)
from .enhanced_mcp import (
    EnhancedMCPManager,
    MCPConfigLoader,
    create_enhanced_mcp_manager,
)
from .multi_file_editor import (
    FileNode,
    MultiFileEditResult,
    DependencyAnalyzer,
    MultiFileEditor,
)

__all__ = [
    "JavaCodeParser",
    "JavaClass",
    "JavaMethod",
    "MavenRunner",
    "CoverageAnalyzer",
    "ProjectScanner",
    "CoverageReport",
    "FileCoverage",
    "AiderCodeFixer",
    "AiderTestGenerator",
    "FixResult",
    "apply_diff_edit",
    "CodeEditor",
    "EditFormat",
    "EditFormatRegistry",
    "EditValidator",
    "CompilationErrorAnalyzer",
    "ErrorAnalysis",
    "TestFailureAnalyzer",
    "FailureAnalysis",
    "ArchitectMode",
    "Tool",
    "ToolCategory",
    "ToolDefinition",
    "ToolParameter",
    "ToolResult",
    "ToolExecutor",
    "create_tool_parameter",
    "ToolRegistry",
    "RegistryConfig",
    "get_registry",
    "set_registry",
    "register_tool",
    "get_tool",
    "list_tools",
    "ToolNotFoundError",
    "ReadTool",
    "WriteTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "BashTool",
    "ToolUseAgent",
    "ToolUseState",
    "ToolCall",
    "ToolUseTurn",
    "create_tool_use_agent",
    "GitStatusTool",
    "GitDiffTool",
    "GitCommitTool",
    "GitBranchTool",
    "GitLogTool",
    "WebSearchTool",
    "WebFetchTool",
    "get_all_search_tools",
    "CodeStructureTool",
    "DependencyGraphTool",
    "get_all_analysis_tools",
    "ToolResultCache",
    "CacheEntry",
    "CachedToolExecutor",
    "create_result_cache",
    # MCP Integration
    "MCPTransportType",
    "MCPToolParameter",
    "MCPTool",
    "MCPToolResult",
    "MCPServerConfig",
    "MCPClient",
    "StdioMCPClient",
    "MCPToolAdapter",
    "MCPManager",
    "discover_mcp_tools",
    "create_mcp_tool_wrapper",
    # Enhanced MCP
    "EnhancedMCPManager",
    "MCPConfigLoader",
    "create_enhanced_mcp_manager",
    # Multi-file Editor
    "FileNode",
    "MultiFileEditResult",
    "DependencyAnalyzer",
    "MultiFileEditor",
]
